# Implementation of Multi-head Latent Attention (MLA) from DeepSeek-V2.

# Original Paper: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
# Author: DeepSeek-AI
# arXiv: https://arxiv.org/abs/2405.04434

# This is an independent implementation adapted from concepts introduced in the above paper. 
# All credits for the theoretical formulation and architecture design go to the original authors.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Copyright (c) 2021 Phil Wang
# Licensed under MIT License
# https://github.com/lucidrains/rotary-embedding-torch
from rotary_embedding_torch import RotaryEmbedding

# Define MLA config
@dataclass
class MLAConfig:
    d_model: int
    num_heads: int
    kv_compressed_dim: int
    q_compressed_dim: int
    rope_dim: int
    seq_len: int
    dropout: float = 0.0
    use_bias: bool = False

# Define MLA
class MLA(nn.Module):
    def __init__(self, config: MLAConfig):
        super(MLA, self).__init__()

        # Initialize config values
        self.config = config
        
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads
        self.kv_lora = config.kv_compressed_dim
        self.q_lora = config.q_compressed_dim
        self.rope_dim = config.rope_dim
        self.seq_len = config.seq_len
        self.dropout = config.dropout
        self.use_bias = config.use_bias

        self.compress_q = nn.Linear(self.d_model, self.q_lora, bias=self.use_bias)
        self.decompress_q = nn.Linear(self.q_lora, self.head_dim * self.num_heads, bias=self.use_bias)
        self.decompress_q_rope = nn.Linear(self.q_lora, self.rope_dim * self.num_heads, bias=self.use_bias)

        self.compress_kv = nn.Linear(self.d_model, self.kv_lora, bias=self.use_bias)
        self.decompress_k = nn.Linear(self.kv_lora, self.head_dim * self.num_heads, bias=self.use_bias)
        self.project_k_rope = nn.Linear(self.d_model, self.rope_dim, bias=self.use_bias)

        self.decompress_v = nn.Linear(self.kv_lora, self.head_dim * self.num_heads, bias=self.use_bias)
        self.output = nn.Linear(self.head_dim * self.num_heads, self.d_model, bias=self.use_bias)

        self.rope = RotaryEmbedding(dim=self.rope_dim)
        self.q_norm = nn.RMSNorm(self.q_lora)
        self.kv_norm = nn.RMSNorm(self.kv_lora)
        self.res_dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        batch, seq, d_model = x.shape

        q_compressed = self.compress_q(x)
        q_compressed = self.q_norm(q_compressed)
        q_decompressed = self.decompress_q(q_compressed)

        kv_compressed = self.compress_kv(x)
        kv_compressed = self.kv_norm(kv_compressed)
        k_decompressed = self.decompress_k(kv_compressed)
        v_decompressed = self.decompress_v(kv_compressed)

        q_decompressed = q_decompressed.view(batch, seq, self.num_heads, self.head_dim)
        k_decompressed = k_decompressed.view(batch, seq, self.num_heads, self.head_dim)
        v_decompressed = v_decompressed.view(batch, seq, self.num_heads, self.head_dim)

        q_rope = self.decompress_q_rope(q_compressed)
        q_rope = q_rope.view(batch, seq, self.num_heads, self.rope_dim)
        q_rope = self.rope.rotate_queries_or_keys(q_rope)

        k_rope = self.project_k_rope(x)
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        k_rope = self.rope.rotate_queries_or_keys(k_rope)

        q = torch.cat([q_decompressed, q_rope], dim=-1)
        k = torch.cat([k_decompressed, k_rope], dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v_decompressed.permute(0, 2, 1, 3)

        output_head = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        output_head = output_head.permute(0, 2, 1, 3).reshape(batch, seq, -1)
        output = self.output(output_head)
        output = self.res_dropout(output)

        return output