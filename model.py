"""
Adapted from nanoGPT/model.py

Changes:
  * vocab_size defaults to 8 (A C G T N + 3 pad slots for tensor alignment)
  * pos_emb="rope" available to simplify parameter counting (but default "learned" still an option)
  * biases default off
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flops import flops_per_token, param_counts

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 8
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    ffn_mult: int = 4
    dropout: float = 0.0 # 0.0 for single-epoch scaling runs
    bias: bool = False
    pos_emb: str = "rope" # "rope" | "learned"
    tie_embeddings: bool = True


## RoPE ##################################################################################

def _rope_cache(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, nh, T, hd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

## Blocks ############################################################################

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_rope = cfg.pos_emb == "rope"
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x, rope=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_rope and rope is not None:
            cos, sin = rope
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None):
        x = x + self.attn(self.ln_1(x), rope=rope)
        return x + self.mlp(self.ln_2(x))


## GPT ##################################################################################

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = (
            nn.Embedding(cfg.block_size, cfg.n_embd)
            if cfg.pos_emb == "learned" else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        self._rope = None
        self.apply(self._init_weights)
        # GPT-2 scaled init on residual projections
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def param_report(self) -> dict:
        """Analytic counts (flops.py) plus the true counts from the module"""
        c = self.cfg
        rep = param_counts(
            c.n_layer, c.n_embd, c.vocab_size, c.block_size,
            ffn_mult=c.ffn_mult, pos_emb=c.pos_emb,
            tie_embeddings=c.tie_embeddings, bias=c.bias,
        )
        actual = sum(p.numel() for p in self.parameters())
        emb = self.wte.weight.numel() + (self.wpe.weight.numel() if self.wpe else 0)
        if not c.tie_embeddings:
            emb += self.lm_head.weight.numel()
        rep["actual_total"] = actual
        rep["actual_non_embedding"] = actual - emb
        return rep

    def flops_per_token(self) -> dict:
        c = self.cfg
        return flops_per_token(
            c.n_layer, c.n_head, c.n_embd, c.block_size, c.vocab_size,
            ffn_mult=c.ffn_mult,
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq len {T} > block_size"
        x = self.wte(idx)
        rope = None
        if self.cfg.pos_emb == "learned":
            pos = torch.arange(T, device=idx.device)
            x = x + self.wpe(pos)
        else:
            if self._rope is None or self._rope[0].shape[0] < T:
                self._rope = _rope_cache(
                    self.cfg.block_size, self.cfg.n_embd // self.cfg.n_head,
                    idx.device, x.dtype,
                )
            rope = self._rope
        x = self.drop(x)
        for blk in self.h:
            x = blk(x, rope=rope)
        x = self.ln_f(x)

        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        AdamW with decay on 2D params only
        """
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(
            groups, lr=learning_rate, betas=betas, fused=fused
        )

    def config_dict(self) -> dict:
        return asdict(self.cfg)
