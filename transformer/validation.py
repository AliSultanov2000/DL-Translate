import torch
import torchmetrics

from transformer_model import *
from tokenizers import Tokenizer



def greedy_valid_decode(model: Transformer, encoder_input: torch.tensor, encoder_mask: torch.tensor, tokenizer_tgt: Tokenizer, device, max_len: int, temperature: float):
    """Greedy decode (tgt) of prepared sentence from Valid DataLoader"""
    # Sos, eos to decoder
    tgt_sos_idx, tgt_eos_idx = tokenizer_tgt.token_to_id('[SOS]'), tokenizer_tgt.token_to_id('[EOS]')
    # Perecompute the encoder output and reuse it for every step
    encoder_output = model.encode(encoder_input, encoder_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.tensor([tgt_sos_idx]).unsqueeze(0).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoding_outputs = model.decode(encoder_output, encoder_mask, decoder_input, None)  # seq_len x d_model
        # d_model
        output = decoding_outputs[:, -1, :]
        # Project next token
        logits = model.project(output)
        # Temperature softmax
        soft_logits = (logits / temperature).softmax(dim=-1)
        # Next token 
        next_token = soft_logits.argmax(dim=-1)
        # Concat the decoder input
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]])], dim=1)
        # Break if we predict the end of sentence token
        if next_token.item() == tgt_eos_idx:
            break

    return decoder_input.squeeze(0)



@torch.no_grad()
def run_validation(config: dict, model: Transformer, tokenizer_tgt: Tokenizer, validation_dataloader, print_msg, epoch: int, writer):
    """Running validation at the end of each epoch"""
    device = torch.deivce(config['device'])
    max_len = config['max_len']
    num_examples = config['num_examples']
    temperature = config['temperature']
    console_width = config['console_width']
    # To metric calculate
    source_texts = []
    expected = []
    predicted = []

    count = 0
    model.eval()
    for batch in validation_dataloader:
        count += 1
        encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
        encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
        # Check that the batch size is 1
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
        # Output for one example
        model_out = greedy_valid_decode(model, encoder_input, encoder_mask, tokenizer_tgt, device, max_len, temperature)  

        # model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
        source_text = batch['src_text'][0]
        target_text = batch['tgt_text'][0]
        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        source_texts.append(source_text)
        expected.append(target_text)
        predicted.append(model_out_text)
        # Print the source, target and model output
        print_msg('=' * console_width)
        print_msg(f"{f'SOURCE: ':>12}{source_text}")
        print_msg(f"{f'TARGET: ':>12}{target_text}")
        print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
        if count == num_examples:
            print_msg('=' * console_width)
            break
        
    # Log the data
    if writer:
        # Metric for all validation dataset
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected).item()
        writer.add_scalar('validation_cer', cer, epoch)
        writer.flush()
        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected).item()
        writer.add_scalar('validation_wer', wer, epoch)
        writer.flush()
        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected).item()
        writer.add_scalar('validation_bleu', bleu, epoch)
        writer.flush()