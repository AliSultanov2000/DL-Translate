import torch
import math
import torch.nn as nn

from torch import tensor


class MultiHeadAttentionBlock(nn.Module):
    # Implementation of MHSA
    def __init__(self, n_heads: int, d_model: int, dropout: float):
        super().__init__()

        self.n_heads = n_heads                    # Count of heads
        self.d_model = d_model                    # Embedding vector dize
        self.d_k = self.d_model // self.n_heads   # Separate d_model into n_heads of d_k
        
        self.dropout = nn.Dropout(dropout)

        # Weights for Keys, Queries, Values
        self.W_K = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.W_Q = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.W_V = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.W_O = nn.Linear(in_features=d_model, out_features=d_model)
    
        assert d_model % n_heads == 0, 'd_model is not divisible by h'
    

    def _self_attention(self, key: tensor, query: tensor, value: tensor, mask: tensor, dropout: nn.Dropout):
        # def input: batch x n_head x seq_len x d_model
        attention_score = (query @ key.transpose(3, 2)) / self.d_model ** 0.5    # batch x n_head x seq_len x seq_len
        
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_score.masked_fill_(mask == 0, float('-inf'))

        attention_score = attention_score.softmax(dim=-1)  # batch x n_head x seq_len x seq_len
        
        if dropout is not None:
            attention_score = dropout(attention_score)
        
        x = attention_score @ value  # batch x n_head x seq_len x d_k 
        return x, attention_score    # batch x n_head x seq_len x d_k | batch x n_head x seq_len x seq_len


    def forward(self, key: tensor, query: tensor, value: tensor, mask):
        # batch_size x seq_len x d_model
        key = self.W_K(key)      # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        query = self.W_Q(query)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model) 
        value = self.W_V(value)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  

        # batch x n_heads x seq_len x d_k
        key = key.view(key.shape[0], self.n_heads, key.shape[1], self.d_k)
        query = query.view(query.shape[0], self.n_heads, query.shape[1], self.d_k)
        value = value.view(value.shape[0], self.n_heads, value.shape[1], self.d_k)

        # x: batch x n_heads x seq_len x d_k | attention_score: batch x n_heads x seq_len x seq_len
        x, self.attention_score = self._self_attention(key, query, value, mask, self.dropout)
        # Return to batch x seq_len x d_model 
        x = x.contiguous().view(x.shape[0], x.shape[2], self.n_heads * self.d_k)  # batch x seq_len x d_model

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
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
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
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)


    def forward(self, x, sublayer):
        """sublayer: PyTorch class"""
        return x + self.dropout(sublayer(self.norm(x)))
    


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.ff(x)



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
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.layers = layers    # Layers of encoder block


    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)   # Batch x seq_len x d_model
        return self.norm(x)
    


class DecoderBlock(nn.Module):
    def __init__(self, features: int, masked_self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.masked_self_attention_block = masked_self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])


    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Masked self attention + Residual
        x = self.residual_connections[0](x, lambda x: self.masked_self_attention_block(x, x, x, tgt_mask))
        # Cross self attention + Residual
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(encoder_output, x, encoder_output, src_mask))
        # Feed forward + Residual
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    
    def forward(self, x):
        # batch x seq_len x d_model ---> batch x seq_len x vocab_size
        return self.proj(x)
    
    

class Transformer(nn.Module):
    def __init__(self,
                encoder: Encoder,
                decoder: Decoder,
                src_embed: InputEmbedding,
                tgt_embed: InputEmbedding,
                src_pos: PositionalEncoding,
                tgt_pos: PositionalEncoding,
                projection_layer: ProjectionLayer):
        
        super().__init__()
        self.src_embed = src_embed                # Input Embedding layer
        self.tgt_embed = tgt_embed                # Target Embedding layer
        self.src_pos = src_pos                    # Input Positional encoding layer
        self.tgt_pos = tgt_pos                    # Target Positional encoding layer
        self.encoder = encoder                    # Encoder
        self.decoder = decoder                    # Decoder
        self.projection_layer = projection_layer  # Projection layer


    def encode(self, src: torch.tensor, src_mask: torch.tensor):
        src = self.src_embed(src)            # Embeddings
        src = self.src_pos(src)              # Positional encoding
        # return batch x seq_len x d_model
        return self.encoder(src, src_mask)   # Encoding
    

    def decode(self, encoder_output: torch.tensor, src_mask: torch.tensor, tgt: torch.tensor, tgt_mask: torch.tensor):
        tgt =  self.tgt_embed(tgt)                                    # Embeddings 
        tgt = self.tgt_pos(tgt)                                       # Positional encoding
        # return batch x seq_len x d_model
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # Decoder
    

    def project(self, x):
        # return batch x seq_len x vocab_size
        return self.projection_layer(x)



def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, n_heads: int=8, dropout: float=0.1, d_ff=2048) -> Transformer:
    """The function returns Transformer model without pretrained weights"""
    
    # Create the embedding layers
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_multi_head_attention_block = MultiHeadAttentionBlock(n_heads, d_model, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        # Create the encoder block
        encoder_block = EncoderBlock(d_model, encoder_multi_head_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_multi_head_attention_block = MultiHeadAttentionBlock(n_heads, d_model, dropout)  # Masked self attention
        decoder_cross_attention_block = MultiHeadAttentionBlock(n_heads, d_model, dropout)       # Cross attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        # Create the decoder block
        decoder_block = DecoderBlock(d_model, decoder_multi_head_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and the decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
