import torch
import torchmetrics

from transformer_model import *
from inferece import translate


@torch.no_grad()
def run_validation(model: Transformer, validation_dataloader, device, print_msg, num_examples, epoch, writer):
    """Running validation at the end of each epoch"""
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []

    console_width = 130
    
    for batch in validation_dataloader:
        count += 1
        encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
        encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
        # check that the batch size is 1
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
        # Output for one example
        # model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
        source_text = batch['src_text'][0]
        target_text = batch['tgt_text'][0]
        model_out_text = translate(source_text)

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
