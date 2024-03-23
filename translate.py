from pathlib import Path
from config import get_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
import torch

class Translator():
    
    def __init__(self):
        self.model = None
        
    def load_model(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
        self.tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
        self.model = build_transformer(self.tokenizer_src.get_vocab_size(), self.tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model']).to(self.device)
        self.seq_len = config['seq_len']
        self.datasource = config['datasource']
        self.lang_src = config['lang_src']
        self.lang_tgt = config['lang_tgt']
        

        # Load the pretrained weights
        model_filename = get_weights_file_path(config, config['preload'])
        state = torch.load(model_filename, map_location=torch.device(self.device))
        self.model.load_state_dict(state['model_state_dict'])

    def translate(self, sentence: str):
        """
        Translate the sentence.
        """
        
        # Sets the model in eval mode.
        self.model.eval()
        with torch.no_grad():
            # Precompute the encoder output and reuse it for every generation step
            source = self.tokenizer_src.encode(sentence)
            source = torch.cat([
                torch.tensor([self.tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([self.tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
                torch.tensor([self.tokenizer_src.token_to_id('[PAD]')] * (self.seq_len - len(source.ids) - 2), dtype=torch.int64)
            ], dim=0).to(self.device)
            source_mask = (source != self.tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(self.device)
            encoder_output = self.model.encode(source, source_mask)

            # Initialize the decoder input with the sos token
            decoder_input = torch.empty(1, 1).fill_(self.tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(self.device)

            # Generate the translation word by word
            while decoder_input.size(1) < self.seq_len:
                # build mask for target and calculate output
                decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(self.device)
                out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

                # project next token
                prob = self.model.project(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(self.device)], dim=1)

                # break if we predict the end of sentence token
                if next_word == self.tokenizer_tgt.token_to_id('[EOS]'):
                    break

        # convert ids to tokens
        return self.tokenizer_tgt.decode(decoder_input[0].tolist())
