"""A small decoder-only Transformer built from basic PyTorch tensor operations."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    """A minimal character-level tokenizer with a deterministic vocabulary."""

    def __init__(self, chars: Iterable[str]):
        self.chars = sorted(set(chars))
        if not self.chars:
            raise ValueError("璇嶈〃涓嶈兘涓虹┖")
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"鏂囨湰鍖呭惈璇嶈〃澶栧瓧绗? {unknown[:10]}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def to_dict(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        return cls(data["chars"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")


class Linear(nn.Module):
    """Linear projection implemented explicitly as x @ W + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        scale = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(in_features, out_features).uniform_(-scale, scale))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight
        return y + self.bias if self.bias is not None else y


class TokenEmbedding(nn.Module):
    """Embedding lookup implemented by indexing a learnable parameter matrix."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine positional encoding from the original Transformer."""

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].to(dtype=x.dtype, device=x.device)


class LayerNorm(nn.Module):
    """Layer normalization implemented with mean and variance tensor ops."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return self.gamma * (x - mean) * torch.rsqrt(var + self.eps) + self.beta


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """True where a query is allowed to attend to a key."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute softmax(QK^T/sqrt(d_k))V without fused attention APIs."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=training)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 蹇呴』鑳借 num_heads 鏁撮櫎")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)
        self.out_proj = Linear(d_model, d_model)
        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        attended, _ = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout, training=self.training
        )
        batch, _, seq_len, _ = attended.shape
        merged = attended.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        return self.out_proj(merged)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = Linear(d_model, hidden_dim)
        self.fc2 = Linear(hidden_dim, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        return self.fc2(F.dropout(x, p=self.dropout, training=self.training))


class TransformerBlock(nn.Module):
    """Pre-norm decoder block; additions are the residual connections."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + F.dropout(self.attn(self.norm1(x), mask), p=self.dropout, training=self.training)
        x = x + F.dropout(self.ffn(self.norm2(x)), p=self.dropout, training=self.training)
        return x


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1


class MiniTransformer(nn.Module):
    """Decoder-only character language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.position = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.final_norm = LayerNorm(config.d_model)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.size(1) > self.config.max_seq_len:
            raise ValueError("杈撳叆闀垮害瓒呰繃 max_seq_len")
        x = self.position(self.embedding(token_ids))
        mask = causal_mask(token_ids.size(1), token_ids.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        # Tied output projection: reuse embedding weights, [B,T,C] @ [C,V].
        logits = x @ self.embedding.weight.transpose(0, 1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature 蹇呴』澶т簬 0")
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None and 0 < top_k < next_logits.size(-1):
                threshold = torch.topk(next_logits, top_k).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat((token_ids, next_id), dim=1)
        return token_ids

    def config_dict(self) -> dict:
        return asdict(self.config)


