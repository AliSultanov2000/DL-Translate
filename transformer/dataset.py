import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        """Length of dataset"""
        return len(self.ds)


    def __getitem__(self, idx: int) -> dict:
        """A fully data preparing to dataloader"""
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]  # str
        tgt_text = src_target_pair['translation'][self.tgt_lang]  # str

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos, padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('The sentence is too long')
        
        # Add SOS, EOS, PAD tokens
        encoder_input = torch.cat([
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),], dim=0)
        
        # Add SOS, PAD tokens
        decoder_input = torch.cat([
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),], dim=0)

        # Add EOS, PAD tokens
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),], dim=0)
        
        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len, 'Encoder input size error'
        assert decoder_input.size(0) == self.seq_len, 'Decoder input size error'
        assert label.size(0) == self.seq_len, 'Label input size error'

        return {
            'encoder_input': encoder_input,  # seq_len
            'decoder_input': decoder_input,  # seq_len
            'label': label,                  # seq_len
            'src_text': src_text,            # str
            'tgt_text': tgt_text,            # str
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            }
    

def causal_mask(size: int) -> torch.tensor:
    """Mask to Masked Selfed Attention in decoder"""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0