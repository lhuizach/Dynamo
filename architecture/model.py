from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4
    ffn_dim: int = 5632
    max_seq_len: int = 4096
    vocab_size: int = 16384
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squaring activations in fp16 can overflow under autocast — compute the norm in fp32
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)  # (end, dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # cos, sin: (T, head_dim/2) — real arithmetic, fully compilable
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim/2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    xq1, xq2 = xq[..., ::2], xq[..., 1::2]
    xk1, xk2 = xk[..., ::2], xk[..., 1::2]
    xq_out = torch.stack([xq1 * cos - xq2 * sin, xq1 * sin + xq2 * cos], dim=-1).flatten(3)
    xk_out = torch.stack([xk1 * cos - xk2 * sin, xk1 * sin + xk2 * cos], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        xk = xk.repeat_interleave(self.n_rep, dim=2)
        xv = xv.repeat_interleave(self.n_rep, dim=2)

        out = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            is_causal=True,
        )
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = FeedForward(config)
        self.use_checkpoint = False

    def _inner(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._inner, x, cos, sin, use_reentrant=False
            )
        return self._inner(x, cos, sin)


class Dynamo(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # weight tying

        cos, sin = precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len * 2)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = tokens.shape
        x = self.embed_tokens(tokens)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_id: Optional[int] = None,
        repetition_penalty: float = 1.3,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # penalise tokens proportional to how recently they appeared
            if repetition_penalty != 1.0:
                recent = idx[0, -64:].tolist()  # last 64 tokens
                for token_id in set(recent):
                    count = recent.count(token_id)
                    logits[0, token_id] /= (repetition_penalty ** count)

            logits = logits / temperature

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            to_remove = (cumulative_probs - sorted_probs) > top_p
            sorted_logits[to_remove] = float("-inf")
            logits.scatter_(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        return idx


if __name__ == "__main__":
    config = ModelConfig()
    model = Dynamo(config)
    total = sum(p.numel() for p in model.parameters())
    unique = sum(p.numel() for p in set(model.parameters()))
    print(f"Total parameters (with tying counted once): {unique / 1e6:.1f}M")
    print(f"Total parameter slots: {total / 1e6:.1f}M")
    x = torch.randint(0, config.vocab_size, (1, 16))
    logits, _ = model(x)
    print(f"Output shape: {logits.shape}")
