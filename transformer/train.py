import torch
import torch.nn as nn
import warnings

from validation import greedy_valid_decode, run_validation
from transformer_model import *
from config import get_config, latest_weights_file_path, get_weights_file_path
from dataset import BilingualDataset, causal_mask

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter


seed = get_config()['seed']
torch.seed = seed
torch.cuda.seed = seed 


def get_all_sentences(ds, lang: str):
    """The function is needed for tokenizer training"""
    for item in ds:
        yield item['translation'][lang]



def get_or_build_tokenizer(config: dict, ds, lang: str) -> Tokenizer:
    """The tokenizer will be trained on the HuggingFace dataset for src, tgt lang"""
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer



def get_max_length(ds_raw, config: dict, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer) -> None:
    """Function find the max length of each sentence: src, tgt"""
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')



def get_model(config: dict, vocab_src_len: int, vocab_tgt_len: int) -> Transformer:
    """Function returns untrained Transformer"""
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'])
    return model



def train_val_ds_split(ds_raw):
    """Random split to train, val datasets"""
    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    return train_ds_raw, val_ds_raw



def get_ds(config: dict) -> DataLoader | DataLoader | Tokenizer | Tokenizer:
    """Src, tgt DataLoader, Tokenizer preparing for training loop"""
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Find the maximum length of each sentence in the source and target sentence
    get_max_length(ds_raw, config, tokenizer_src, tokenizer_tgt)
    
    # Random data split to training, validation
    train_ds_raw, val_ds_raw = train_val_ds_split(ds_raw)
    
    # Creata datasets
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # Create dataloader
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



def save_state(model_filename: str, epoch: int, global_step: int,  model: Transformer, optimizer):
    """State saving at the end of each epochs"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        }, model_filename)



def load_saved_state(model_filename: str, model: Transformer, optimizer: torch.optim.Adam):
    """Load saved state at the beginning of model training"""

    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    initial_epoch = state['epoch'] + 1
    global_step = state['global_step']
    return model, optimizer, initial_epoch, global_step



def number_train_params(model: nn.Module) -> int:
    """The number of parameters to be trained"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def check_train_device(config: dict):
    """Check torch device to model train"""
    device = config['device']
    print('Using device to train:', device)

    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    device = torch.device(device)
    return device



def train_model(config: dict) -> None:
    """Training loop for Transformer model"""
    # Define the device
    device = check_train_device(config)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Optimizer, loss function to Transformer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    # If there is saved model, optimizer, initial_epoch, global_step then load it
    if model_filename:
        model, optimizer, initial_epoch, global_step = load_saved_state(model_filename, model, optimizer)
    else:
        print('No model to preload, starting from scratch')

    # Check train params
    print(f'The number of model parameters for training: {number_train_params(model)}')
    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)
            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)
            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()
            # Backpropagate the loss
            loss.backward()
            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Update global state
            global_step += 1

        # Run validation at the end of every epoch
        run_validation(config, model, tokenizer_src, tokenizer_tgt, val_dataloader, config['max_len'], lambda msg: batch_iterator.write(msg), 1, epoch, writer)    
        # Save the state at the end of each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        save_state(model_filename, epoch, global_step, model, optimizer)

    writer.close()



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
