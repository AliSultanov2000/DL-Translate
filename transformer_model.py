import torch
import math
import torch.nn as nn

from torch import tensor

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, n_heads: int, d_model: int):
        super().__init__()

        self.softmax = nn.Softmax(dim=3)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.n_heads
                
        # Weights for Keys, Queries, Values
        self.W_K = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.W_Q = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.W_V = nn.Linear(in_features=self.d_model, out_features=self.d_model)

        self.W_O = nn.Linear(in_features=d_model, out_features=d_model)

        assert d_model % n_heads == 0, 'd_model % n_heads Error'
    

    def _self_attention(self, key: tensor, query: tensor, value: tensor, mask):
        # def input: batch x n_head x seq_len x d_model
        attention_score = query @ key.transpose(3, 2) // self.d_model ** 0.5    # batch x n_head x seq_len x seq_len
        
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)

        attention_score = self.softmax(attention_score)                         # batch x n_head x seq_len x seq_len

        x = attention_score @ value  
        return x, attention_score   # batch x n_head x seq_len x d_k | batch x n_head x seq_len x seq_len


    def forward(self, key: tensor, query: tensor, value: tensor, mask):
        # batch_size x seq_len x d_model
        key = self.W_K(key)   
        query = self.W_Q(query)    
        value = self.W_V(value)    

        # batch x n_heads x seq_len x d_k
        key = key.view(key.shape[0], self.n_heads, key.shape[1], self.d_k)
        query = query.view(query.shape[0], self.n_heads, query.shape[1], self.d_k)
        value = value.view(value.shape[0], self.n_heads, value.shape[1], self.d_k)

        # x: batch x n_heads x seq_len x d_k | attention_score: batch x n_heads x seq_len x seq_len
        x, attention_score = self._self_attention(key, query, value, mask)
        x = x.view(x.shape[0], x.shape[2], self.n_heads * self.d_k)  # batch x seq_len x d_model

        x = self.W_O(x)  # batch x seq_len x d_model

        return x



class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)


    def forward(self, x):
        # x: batch_size x seq_len
        return self.embedding(x) * (self.d_model ** 0.5)  # batch_size x seq_len x d_model




class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter


    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim = True)   # (batch, seq_len, 1)
        # Eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias



class ResidualConnection(nn.Module):
    def __init__(self, features, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)


    def forward(self, x, sublayer):
        """sublayer: PyTorch class"""
        return x + self.dropout(sublayer(self.norm(x)))



class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2


    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, features: int, multi_head_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()

        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        self.multi_head_attention_block = multi_head_attention_block
        self.feed_forward_block = feed_forward_block


    def forward(self, x: torch.tensor, mask=None):
        x = self.residual_connection[0](x, lambda x: self.multi_head_attention_block(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.layers = layers  # Layers of encoder block


    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)   # Batch x seq_len x d_model
        return self.norm(x)
