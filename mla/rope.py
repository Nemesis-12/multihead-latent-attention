# Adapted from Sebastian Raschka's LLMs from Scratch
# https://github.com/rasbt/LLMs-from-scratch

import torch

def precompute_rope(dim_head, theta_base=10_000, max_seq_len=4096):
    assert dim_head % 2 == 0, "Embedding dimension must be divisible by 2"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, dim_head, 2)[: (dim_head // 2)].float() / dim_head))

    # Generate position indices
    positions = torch.arange(max_seq_len)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (max_seq_len, dim_head // 2)

    # Expand angles to match the dim_head
    angles = torch.cat([angles, angles], dim=1)  # Shape: (max_seq_len, dim_head)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def RoPE(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)