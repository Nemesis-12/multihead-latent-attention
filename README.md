# Multi-head Latent Attention (MLA)

## Overview
This repository attempts to implement Multi-head Latent Attention (MLA), a novel attention mechanism introduced by DeepSeek in their paper ["DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"](https://arxiv.org/abs/2405.04434), which eliminates the bottleneck caused by Key-Value (KV) cache in Multi-Head Attention (MHA).

## Install
```bash
pip install multihead-latent-attention
```

## Usage
### Standalone Usage
```python
import torch
from mla import MLA, MLAConfig

# Initialize configuration for MLA
config = MLAConfig(
    d_model=256,
    num_heads=8,
    kv_compressed_dim=64,
    q_compressed_dim=128,
    rope_dim=32,
    seq_len=256
)

# Initialize attention
attention = MLA(config)

# Dummy values - dimensions should be (batch_size, seq_len, d_model)
x = torch.randn(2, 256, 256)

# Output of the attention mechanism
output = attention(x)

print("Output shape:", output.shape)
```

### Usage in a Transformer Block
MLA can be integrated into a transformer block as shown below:
```python
@dataclass
class Config():
    d_model: int
    num_heads: int
    kv_compressed_dim: int
    q_compressed_dim: int
    rope_dim: int
    seq_len: int
    vocab_size: int
    ff_hidden: int
    hidden_dim: int = 128
    action_embed: int = 64
    dropout: float = 0.0
    use_bias: bool = False

class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super(TransformerBlock, self).__init__()

        self.config = config
        self.d_model = config.d_model
        self.ff_hidden = config.ff_hidden
        self.dropout = config.dropout

        self.norm1 = nn.RMSNorm(self.d_model)
        self.norm2 = nn.RMSNorm(self.d_model)
        self.res_dropout = nn.Dropout(p=self.dropout)

        # Initialize MLA
        mla_config = MLAConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            kv_compressed_dim=config.kv_compressed_dim,
            q_compressed_dim=config.q_compressed_dim,
            rope_dim=config.rope_dim,
            seq_len=config.seq_len
        )
        self.attn = MLA(mla_config)

        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.ff_hidden),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.ff_hidden, self.d_model)
        )

    def forward(self, x):
        norm1 = self.norm1(x)
        # Add the attention to your forward pass
        attn_out = self.attn(norm1)
        x = x + self.res_dropout(attn_out)

        norm2 = self.norm2(x)
        ff_out = self.ff(norm2)
        x = x + self.res_dropout(ff_out)

        return x
```

## Contributing
This project is an educational tool and welcomes contributions. To contribute:
- Fork the repository.
- Make your changes.
- Submit a pull request with a description of your updates.

Feel free to open an issue for suggestions or bugs.

## Citation
```bibtex 
@article{deepseek2024,
  title   = {DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model},
  author  = {DeepSeek-AI},
  journal = {arXiv},
  year    = {2024},
  url     = {https://arxiv.org/abs/2405.04434}
}
```

## License
This project is licensed under the [MIT License](LICENSE).
