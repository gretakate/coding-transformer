import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        """
        d_model: Dimension of the model
        vocab_size: How many words are in the vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # PyTorch provides an embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        d_model: Dimension of the model
        seq_len: Maximum length of the sentence
        dropout: To help the model not overfit 
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of length (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Using a log in the div term has equivalent result to the formula for positional encoding in the transformers paper, but more numerically stable
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to the even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to the odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension to positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        
        # Save the tensor along with the model by registering it as a buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add this positional encoding to every word in the sentence.
        Need to tell the model not to learn the positional encoding because it is a fixed value. (denoted with requires_grad(False))
        Remember to include dropout.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        
        
class LayerNormalization(nn.Module):
    
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        """
        eps: Very small number used for numerical stability (prevent denominator from being zero in the case that sigma^2 is 0)
        """
        super().__init__()
        self.eps = eps
        # Multiplicative learnable parameter, initialized to 1
        self.alpha = nn.Parameter(torch.ones(features))
        # Additive learnable parameter, initialized to 0
        self.bias = nn.Parameter(torch.zeros(features))
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1 (bias is true by default in nn.Linear)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
        
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        # Remember to apply dropout in between the layers!
        # Use ReLU as activation function
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        d_model: Dimension of the model
        h: Number of attention heads -> required that d_model is divisible by h to get evenly split embedding vectors for each attention head
        dropout: Dropout value
        """
        super().__init__()
        self.d_model = d_model
        self.h = h 
        assert d_model % h == 0,  "d_model is not divisible by h"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # @ means matrix multiplication in PyTorch
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # before applying softmax, we need to apply the mask
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim = -1)  # (Batch, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        # We output attention_scores so we can use it for visualization
        return attention_scores @ value, attention_scores
        
    def forward(self, q, k, v, mask):
        """
        mask: Used for when we want some words to not interact with other words
        """
        query = self.w_q(q)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        
        # view method of PyTorch keeps the batch dimension (We don't want to split the sentence, just the embedding)
        # Transpose because we prefer to have the h dimension as the second dimension. Each head will watch (seq_len, d_k)
        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        
        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # Multiply by output matrix
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    
    
class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    def forward(self, x, sublayer):
        """
        sublayer: Previous layer
        """
        return x + self.dropout(sublayer(self.norm(x)))
        
        
class EncoderBlock(nn.Module):
    
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        self_attention_block: Called self attention because the same input is applied to the role of query, key, and value
        feed_forward_block: Feed forward block
        dropout: Dropout
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # ModuleList is a way to organize a list of modules
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        """
        src_mask: So we don't let the padding words interact with the other words
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
                
        
class Encoder(nn.Module):
    # Made up of many EncoderBlocks
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x: input to the decoder
        encoder_outout: output of the encoder
        src_mask: mask that is applied to the encoder
        tgt_mask: mask that is applied to the decoder
        """
        # First calculate the self attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        
        # Next calculate the cross attention
        # query comes from the decoder, keys and values come from the encoder, uses the mask of the encoder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        
        # Finally, calculate the feedforward block
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
        
            
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Projects the embedding back into the vocabulary.
        d_model: Dimension of the model
        vocab_size: Size of the vocabulary
        """
        super().__init__()   
        self.proj = nn.Linear(d_model, vocab_size)     
        
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        # Using lgsoftmax instead of softmax for numerical stability
        return self.proj(x)
    
    
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """
        The full transformer block
        encoder: Encoder
        decoder: Decoder
        src_embed: Embeddings of the src language
        tgt_embed: Embeddings of the tgt language
        src_pos: Source position embeddings
        tgt_pos: Target position embeddings
        projection_layer: Projection layer
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int=8, dropout=0.1, d_ff: int = 2048) -> Transformer:
    """
    Function that builds a transformer given a set of hyperparameters.
    src_vocab_size: Vocabulary size of the source language
    tgt_vocab_size: Vocabulary size of the target language
    src_seq_len: Sequence length of the source text
    tgt_seq_len: Sequence length of the target text
    d_model: Size of the model
    N: Number of layers (N encoder blocks and N decoder blocks)
    h: Number of heads
    dropout: 
    d_ff: Dimension of the hidden layer of the feedforward netword
    """
    # First create the embedding layers
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)
    
    # Next create the positional encoding layers
    src_pos = PositionalEncoding(d_model=d_model, seq_len=src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(d_model=d_model, seq_len=tgt_seq_len, dropout=dropout)
    
    # Create the Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    # Create the Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # Create the Encoder and the Decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the ProjectionLayer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer