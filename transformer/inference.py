from transformer_model import *
from config import get_config, latest_weights_file_path

from pathlib import Path
from tokenizers import Tokenizer


@torch.no_grad()
def translate(sentence: str) -> str:
    """Text translation using a transformer"""
    config = get_config()
    temperature = config['temperature']  # Using in softmax
    seq_len = config['seq_len']    
    device = torch.device(config['device'])
    # Tokenizers
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    # Idxs to src
    src_sos_idx, src_eos_idx = tokenizer_src.token_to_id('[SOS]'), tokenizer_src.token_to_id('[EOS]')
    src_pad_idx = tokenizer_src.token_to_id('[PAD]')
    # Idxs to tgt 
    tgt_sos_idx, tgt_eos_idx = tokenizer_tgt.token_to_id('[SOS]'), tokenizer_tgt.token_to_id('[EOS]')
    
    # Transformer architecture
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'], config['seq_len']).to(device)
    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    if model_filename:
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])  # Weights
    
    model.eval()
    encoder_input = tokenizer_src.encode(sentence)
    encoder_input = torch.cat([
        torch.tensor([src_sos_idx], dtype=torch.int64), 
        torch.tensor(encoder_input.ids, dtype=torch.int64),
        torch.tensor([src_eos_idx], dtype=torch.int64),
        torch.tensor([src_pad_idx] * (seq_len - len(encoder_input.ids) - 2), dtype=torch.int64),
        ], dim=0).unsqueeze(0).to(device)
    
    encoder_mask = (encoder_input != src_pad_idx).unsqueeze(0).unsqueeze(0).int().to(device)
    encoder_output = model.encode(encoder_input, encoder_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.tensor([tgt_sos_idx]).unsqueeze(0).to(device)
    # Generate the translation word by word
    while decoder_input.size(1) < seq_len:
        decoding_outputs = model.decode(encoder_output, encoder_mask, decoder_input, None)  # seq_len x d_model
        # d_model
        output = decoding_outputs[:, -1, :]
        # Project next token
        logits = model.project(output)
        # Temperature softmax
        soft_logits = (logits / temperature).softmax(dim=-1)
        # Next token 
        next_token = soft_logits.argmax(dim=-1)
        # Concat decoder input
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]])], dim=1)
        # Break if we predict the end of sentence token
        if next_token.item() == tgt_eos_idx:
            break

    return tokenizer_tgt.decode(decoder_input[0].tolist())



if __name__ == '__main__':
    print(translate('How are you?'))