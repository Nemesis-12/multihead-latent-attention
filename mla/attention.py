# Adapted from Sebastian Raschka's LLMs from Scratch
# https://github.com/rasbt/LLMs-from-scratch

import torch
import torch.nn as nn
from rope import precompute_rope, RoPE

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False, latent_dim=None):
        super().__init__()
        assert d_out % num_heads == 0, f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_in
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout_rate = dropout
        
        self.latent_dim = latent_dim if latent_dim is not None else max(16, d_out // 8)

        # Projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_dkv = nn.Linear(d_in, self.latent_dim, bias=qkv_bias)
        self.W_uk = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)
        self.W_uv = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
        # RoPE embeddings
        cos, sin = precompute_rope(dim_head=self.head_dim, max_seq_len=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Project queries and latent KV
        queries = self.W_query(x)
        latent_kv = self.W_dkv(x)

        # Up-project to keys and values
        keys = self.W_uk(latent_kv)
        values = self.W_uv(latent_kv)

        # Reshape for multi-head attention
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos_slice = self.cos[:num_tokens, :]
        sin_slice = self.sin[:num_tokens, :]
        queries = RoPE(queries, cos_slice, sin_slice)
        keys = RoPE(keys, cos_slice, sin_slice)

        # Compute attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # Apply causal mask
        mask_slice = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask_slice, float('-inf'))

        # Attention weights and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        self.attn_weights = attn_weights.detach()
        attn_weights = self.dropout(attn_weights)

        # Compute context vector
        context_vec = torch.matmul(attn_weights, values)
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(b, num_tokens, self.num_heads * self.head_dim)
        context_vec = self.out_proj(context_vec)

        return context_vec