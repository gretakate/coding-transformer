import pytest
import torch.nn as nn
from model import InputEmbeddings, PositionalEncoding, Encoder, EncoderBlock, Decoder, DecoderBlock, ProjectionLayer, Transformer, MultiHeadAttentionBlock, FeedForwardBlock
from train import build_transformer

def test_build_transformer(config, tokenizer_src, tokenizer_tgt):
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    src_seq_len = config['seq_len']
    tgt_seq_len = config['seq_len']
    d_model = config['d_model']
    N = config['N']
    h = config['h']
    dropout = config['dropout']
    d_ff = config['d_ff']
    
    transformer = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, N, h, dropout, d_ff)
    
    # Assert that transformer is the expected architecture
    assert transformer is not None
    assert isinstance(transformer, Transformer)
    assert isinstance(transformer.encoder, Encoder)
    assert isinstance(transformer.decoder, Decoder)
    assert isinstance(transformer.projection_layer, ProjectionLayer)
    assert isinstance(transformer.src_embed, InputEmbeddings)
    assert isinstance(transformer.tgt_embed, InputEmbeddings)
    assert isinstance(transformer.src_pos, PositionalEncoding)
    assert isinstance(transformer.tgt_pos, PositionalEncoding)
    
    # Transformer has N layers in encoder and decoder
    assert len(transformer.encoder.layers) == N
    assert len(transformer.decoder.layers) == N
    
    # Make sure modules are initialized with correct attributes and sizes
    assert transformer.src_embed.d_model == d_model
    assert transformer.src_embed.vocab_size == src_vocab_size
    
    assert transformer.tgt_embed.d_model == d_model
    assert transformer.tgt_embed.vocab_size == tgt_vocab_size
    
    assert transformer.src_pos.d_model == d_model
    assert transformer.src_pos.seq_len == src_seq_len
    assert transformer.src_pos.dropout.p == dropout
    
    assert transformer.tgt_pos.d_model == d_model
    assert transformer.tgt_pos.seq_len == tgt_seq_len
    assert transformer.tgt_pos.dropout.p == dropout
    
    assert isinstance(transformer.encoder.layers[0], EncoderBlock)
    assert isinstance(transformer.decoder.layers[0], DecoderBlock)
    
    assert isinstance(transformer.encoder.layers[0].self_attention_block, MultiHeadAttentionBlock) 
    assert isinstance(transformer.decoder.layers[0].self_attention_block, MultiHeadAttentionBlock) 
    assert isinstance(transformer.decoder.layers[0].cross_attention_block, MultiHeadAttentionBlock) 
    
    assert transformer.encoder.layers[0].self_attention_block.h == h
    assert transformer.decoder.layers[0].self_attention_block.h == h
    assert transformer.decoder.layers[0].cross_attention_block.h == h
    
    assert isinstance(transformer.encoder.layers[0].feed_forward_block, FeedForwardBlock) 
    assert isinstance(transformer.decoder.layers[0].feed_forward_block, FeedForwardBlock) 
    
    assert transformer.encoder.layers[0].feed_forward_block.linear_1.in_features == d_model
    assert transformer.encoder.layers[0].feed_forward_block.linear_1.out_features == d_ff
    assert transformer.encoder.layers[0].feed_forward_block.linear_2.in_features == d_ff
    assert transformer.encoder.layers[0].feed_forward_block.linear_2.out_features == d_model
    
    assert transformer.projection_layer.proj.in_features == d_model
    assert transformer.projection_layer.proj.out_features == tgt_vocab_size