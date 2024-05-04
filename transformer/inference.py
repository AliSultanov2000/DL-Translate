from transformer_model import *
from config import get_config, latest_weights_file_path

from pathlib import Path
from tokenizers import Tokenizer


@torch.no_grad()
def translate(sentence: str) -> str:
    """Text translation using a transformer"""
    config = get_config()    
    device = config['device']
    device = torch.device(device)

    seq_len = config['seq_len']
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    # Transformer acrhitecture
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'], config['seq_len'])
    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    if model_filename:
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])  # Weights
    
    model.eval()
    source = tokenizer_src.encode(sentence)
    source = torch.cat([
        torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
        torch.tensor(source.ids, dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)],
        dim=0).unsqueeze(0).to(device)
        
    encoder_output = model.encode(source, None)
    decoder_input = torch.tensor([tokenizer_src.token_to_id('[SOS]')]).unsqueeze(0)
    # Generate the translation word by word
    while decoder_input.size(1) < seq_len:
        decoding_outputs = model.decode(encoder_output, None, decoder_input, None)  # seq_len x d_model
        # d_model
        output = decoding_outputs[:, -1, :]
        # Project next token
        logits = model.project(output)
        # Next token 
        next_token = logits.argmax(dim=-1)
        
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]])], dim=1)
        # Break if we predict the end of sentence token
        if next_token.item() == tokenizer_tgt.token_to_id('[EOS]'):
            break

    return tokenizer_tgt.decode(decoder_input[0].tolist())   



if __name__ == '__main__':
    print(translate('How are you?'))
