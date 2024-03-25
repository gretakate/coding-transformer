from transformers import PretrainedConfig

# Write a custom configuration
class ScratchTransformerConfig(PretrainedConfig):
  model_type = "scratch-transformer"

  def __init__(self, 
               src_vocab_size: int = 25000, 
               tgt_vocab_size: int = 25000, 
               seq_len: int = 350,
               src_seq_len: int = 350,
               tgt_seq_len: int = 350,
               d_model: int = 512, 
               N: int = 6, 
               h: int=8, 
               dropout=0.1, 
               d_ff: int = 2048,
               **kwargs):
    self.src_vocab_size = src_vocab_size
    self.tgt_vocab_size = tgt_vocab_size
    self.seq_len = seq_len
    self.src_seq_len = src_seq_len
    self.tgt_seq_len = tgt_seq_len
    self.d_model = d_model
    self.N = N
    self.h = h
    self.dropout = dropout
    self.d_ff = d_ff
    super().__init__(**kwargs)