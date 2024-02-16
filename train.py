import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_weights_file_path, get_config
import warnings

from pathlib import Path
from tqdm import tqdm

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw  = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # Batch size 1 because we want to load one at a time
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    # Sticking with most defaults
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # Define the device on which to store tensors
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print(f"Using device {device}")
    
    # Make sure that the weights folder is created
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # Transfer model to device
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard - to visualize loss
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    initial_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    # We don't want the padding token to contribute to the loss
    # Using label smoothing
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch: 02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (Batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (Batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (Batch, 1, 1, seq_len)
            
            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (Batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, decoder_mask)  # (Batch, seq_len, d_model)
            
            # need to map back to the vocabulary
            # This is the output of the model
            proj_output = model.project(decoder_output)  # (Batch, seq_len, tgt_vocab_size)
            
            # Now we need to compare the output to the label
            label = batch['label'].to(device)  # (Batch, seq_len)
            
            # (Batch, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            # Update the progress bar with the loss
            batch_iterator.set_postfix(f"loss: {loss.item():6.3f}")
            
            # Log the loss
            writer.add_scalar('train loss', loss.item, global_step)
            writer.flush()
            
            # Backpropagate
            loss.backward()
            
            # Update the weights of the model
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            
        # Save the model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            # Save the state of the optimizer as well as the state of the model (good practice if you want your training to be resumable)
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
            
            