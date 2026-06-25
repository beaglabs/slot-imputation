import math
import os
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        d_ff: int = 2048,
        n_heads: int = 12,
        n_layers: int = 12,
        max_seq_len: int = 1024,
        use_mlp: bool = True,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_mlp = use_mlp
        self.d_head = d_model // n_heads

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        return cls(**d)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_out = attn_weights @ v
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.W1(x))
        return self.W2(h)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config.d_model, config.d_ff) if config.use_mlp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT2(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, use_mlp: Optional[bool] = None
    ) -> torch.Tensor:
        B, T = input_ids.shape

        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        for block in self.blocks:
            if use_mlp is False and block.mlp is not None:
                x = x + block.attn(block.ln1(x))
            else:
                x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_embed.weight.numel()
            total -= self.pos_embed.weight.numel()
        return total

    @staticmethod
    def save(model: "MiniGPT2", path: str, metadata: dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "metadata": metadata,
        }
        torch.save(data, path)

    @staticmethod
    def load(
        path: str, map_location: str = "cpu"
    ) -> Tuple["MiniGPT2", dict]:
        data = torch.load(path, map_location=map_location, weights_only=False)
        config = TransformerConfig.from_dict(data["config"])
        model = MiniGPT2(config)
        model.load_state_dict(data["state_dict"])
        return model, data["metadata"]