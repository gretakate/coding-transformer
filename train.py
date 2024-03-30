import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchtext.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from dataset import BilingualDataset, causal_mask
from model import build_transformer

import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_weights_file_path, get_config, latest_weights_file_path
import warnings

from pathlib import Path
from tqdm import tqdm

# To get around bad huggingface SSL certificate error
import os
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['REQUESTS_CA_BUNDLE'] = ''

# Because mps does not currently support backward()
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    """
    Greedy decoding function for sequence generation using a transformer-based model.

    Args:
    - model (torch.nn.Module): The transformer model for sequence generation.
    - source (torch.Tensor): The input source sequence tensor.
    - source_mask (torch.Tensor): The mask tensor for the source sequence.
    - tokenizer_tgt (Tokenizer): The target sequence tokenizer.
    - max_len (int): Maximum length of the generated sequence.
    - device (str): Device ('cpu' or 'cuda') to run the model.

    Returns:
    - decoded_sequence (torch.Tensor): The decoded sequence tensor.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    """
    Run validation on a dataset using a transformer-based model and calculate evaluation metrics.

    Args:
    - model (torch.nn.Module): The Transformer model to validate.
    - validation_ds (torch.utils.data.Dataset): The validation dataset.
    - tokenizer_tgt (Tokenizer): The target sequence tokenizer.
    - max_len (int): Maximum length of the generated sequence.
    - device (str): Device ('cpu' or 'cuda') to run the model.
    - print_msg (function): Function for printing messages.
    - global_step (int): Current global step for tensorboard logging.
    - writer (SummaryWriter): Tensorboard SummaryWriter for logging.
    - num_examples (int): Number of examples to print during validation (default is 2).

    Returns:
    - None
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
        
        
def get_all_sentences(ds, lang):
    """
    Generator function to extract all sentences of a specific language from a dataset.

    Args:
    - ds (Iterable): Iterable dataset containing translation pairs.
    - lang (str): Language code specifying the target language.

    Yields:
    - sentence (str): A sentence in the specified language.
    """
    for item in ds:
        yield item['translation'][lang]

def build_tokenizer(dataset, lang):
    """
    Build a tokenizer for a specific language.

    Args:
    - dataset (Iterable): Iterable dataset containing text samples.
    - lang (str): Language code specifying the language of the text samples.

    Returns:
    - tokenizer (Tokenizer): The tokenizer object.
    """
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
    return tokenizer

def get_or_build_tokenizer(config, dataset, lang):
    """
    Get or build a tokenizer for a specific language.

    If the tokenizer file does not exist, a new tokenizer is built and trained using the provided dataset.
    Otherwise, the existing tokenizer is loaded from the file.

    Args:
    - config (dict): Configuration dictionary containing tokenizer file paths.
    - dataset (Iterable): Iterable dataset containing text samples.
    - lang (str): Language code specifying the language of the text samples.

    Returns:
    - tokenizer (Tokenizer): The tokenizer object.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Make sure the directory is created
        tokenizer_directory, _ = os.path.split(tokenizer_path)
        Path(tokenizer_directory).mkdir(parents=True, exist_ok=True)
        
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    """
    Create training and validation dataloaders along with tokenizers from a given configuration.

    Args:
    - config (dict): Configuration dictionary containing dataset and model parameters.

    Returns:
    - train_dataloader (DataLoader): Dataloader for the training dataset.
    - val_dataloader (DataLoader): Dataloader for the validation dataset.
    - tokenizer_src (Tokenizer): Tokenizer for the source language.
    - tokenizer_tgt (Tokenizer): Tokenizer for the target language.
    """
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    
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
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # Batch size 1 because we want to load one at a time
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Create a transformer model based on the provided configuration and vocabulary sizes.

    Args:
    - config (dict): Configuration dictionary containing model parameters.
    - vocab_src_len (int): Size of the source language vocabulary.
    - vocab_tgt_len (int): Size of the target language vocabulary.

    Returns:
    - model: Transformer model.
    """
    # Sticking with most defaults
    # TODO: Allow for more configurations
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    """
    Train a transformer model based on the provided configuration. 
    Save model weights and tokenizers to directories specified in the configuration. 
    Training can be resumed from checkpoints specified in the configuration.

    Args:
    - config (dict): Configuration dictionary containing training parameters.

    Returns:
    - None
    """
    # Define the device on which to store tensors
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    # device = 'cpu'  # Having to do this because of https://github.com/pytorch/pytorch/issues/77764 with MPS
    print(f"Using device {device}")
    device = torch.device(device)
    
    # Make sure that the weights folder is created
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # Transfer model to device
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard - to visualize loss
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    initial_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f"Loading model {model_filename}")
        state = torch.load(model_filename, map_location=torch.device(device))
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
        
    # We don't want the padding token to contribute to the loss- use tgt PAD token because both proj_output and label are in the target domain
    # Using label smoothing
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch: 02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (Batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (Batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (Batch, 1, 1, seq_len)
            
            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (Batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (Batch, seq_len, d_model)
            
            # need to map back to the vocabulary
            # This is the output of the model
            proj_output = model.project(decoder_output)  # (Batch, seq_len, tgt_vocab_size)
            
            # Now we need to compare the output to the label
            label = batch['label'].to(device)  # (Batch, seq_len)
            
            # (Batch, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            # Update the progress bar with the loss
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            # Backpropagate
            loss.backward()
            
            # Update the weights of the model
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            
        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
            
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
            
            