import torch
from transformers import PreTrainedModel
from .configuration_scratch_transformer import ScratchTransformerConfig
from .model import build_transformer
from typing import Optional, Tuple
from transformers.modeling_outputs import Seq2SeqLMOutput


# Write a custom model
class ScratchTransformerModel(PreTrainedModel):
    config_class = ScratchTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = build_transformer(src_vocab_size=self.config.src_vocab_size, tgt_vocab_size=self.config.tgt_vocab_size, src_seq_len=self.config.src_seq_len, tgt_seq_len=self.config.tgt_seq_len)
        self.config.decoder_start_token_id = None  # Needs to be set before running the model
        self.config.pad_token_id = None  # Needs to be set before running the model
        self.config.eos_token_id = None  # Needs to be set before running the model        
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
    ) -> Seq2SeqLMOutput:
        """
        Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py
        - input_ids: Optional[torch.LongTensor]: Tokenized input sequence
        - attention_mask: Optional[torch.FloatTensor] = None: Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`
        - decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*): Indices of decoder input sequence tokens in the vocabulary. 
        - decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
        """
        # TODO: Add some checks to make sure padding token ids and vocab size have been set for the model before running

        # Encode
        if encoder_outputs is None:
            encoder_output = self.model.encode(input_ids, attention_mask)  # (Batch, seq_len, d_model)

        # Decode
        if decoder_input_ids is None:
            # No decoder inputs are given, runs inference only
            # Initialize the decoder input with the sos token
            decoder_input = torch.empty(1, 1).fill_(self.config.decoder_start_token_id).type_as(input_ids)
            
            # Generate the translation word by word
            while decoder_input.size(1) < self.config.seq_len:
                # build mask for target and calculate output
                decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(attention_mask)
                out = self.model.decode(encoder_output, attention_mask, decoder_input, decoder_mask)

                # project next token
                prob = self.model.project(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(input_ids).fill_(next_word.item())], dim=1)

                # break if we predict the end of sentence token
                if next_word == self.config.eos_token_id:
                    break
            return Seq2SeqLMOutput(logits=decoder_input)
                
        else:
            # Decoder inputs are given, runs inference and computes loss
            if decoder_attention_mask is None:
                decoder_attention_mask = decoder_input_ids.new_tensor(decoder_input_ids != self.config.pad_token_id)

            decoder_output = self.model.decode(encoder_output, attention_mask, decoder_input_ids, decoder_attention_mask)  # (Batch, seq_len, d_model)

            # This is the output of the model; maps back to the vocabulary
            logits = self.model.project(decoder_output)

            # We don't want the padding token to contribute to the loss
            # Using label smoothing
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, label_smoothing=0.1)

            # (Batch, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(logits.view(-1, self.config.tgt_vocab_size), decoder_input_ids.view(-1))

            return Seq2SeqLMOutput(
                loss=loss,
                logits=logits
            )
                    
