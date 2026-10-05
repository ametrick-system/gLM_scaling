"""Exact parameter and FLOP accounting"""

from __future__ import annotations
from dataclasses import dataclass

def param_counts(
    n_layer: int,
    n_embd: int,
    vocab_size: int,
    block_size: int,
    *,
    ffn_mult: int = 4,
    pos_emb: str = "rope",
    tie_embeddings: bool = True,
    bias: bool = False,
) -> dict[str, int]:
    """Parameter counts, split into embedding and non-embedding

    Non-embedding: 
    12 * n_layer * n_embd**2 for a pre-LN transformer with 4x MLP expansion (4 * d^2 attention + 8 * d^2 MLP),
    plus LayerNorm gains (and biases, and projection biases, if bias=True)

    Exact agreement with the live module is checked by tests/test_accounting.py
    """
    d = n_embd
    nb = 1 if bias else 0
    attn = 4 * d * d + nb * (3 * d + d)   # q,k,v (3d) + out proj (d) biases
    mlp = 2 * ffn_mult * d * d + nb * (ffn_mult * d + d)
    ln = 2 * (1 + nb) * d                 # two LayerNorms per block
    per_layer = attn + mlp + ln
    non_embedding = n_layer * per_layer + (1 + nb) * d   # + final LayerNorm

    tok_emb = vocab_size * d
    pos = block_size * d if pos_emb == "learned" else 0
    # untied output head is a separate matrix
    head = 0 if tie_embeddings else vocab_size * d
    embedding = tok_emb + pos + head

    return {
        "n_non_embedding": non_embedding,
        "n_embedding": embedding,
        "n_total": non_embedding + embedding,
        "n_per_layer": per_layer,
    }

def flops_per_token(
    n_layer: int,
    n_head: int,
    n_embd: int,
    block_size: int,
    vocab_size: int,
    *,
    ffn_mult: int = 4,
    include_backward: bool = True,
) -> dict[str, float]:
    """Follows Chinchilla paper Appendix F calculations"""
    d, L, T, V = n_embd, n_layer, block_size, vocab_size

    qkv = 3 * 2 * d * d
    attn_logits = 2 * T * d
    attn_softmax = 3 * T * n_head
    attn_values = 2 * T * d
    out_proj = 2 * d * d
    mlp = 2 * 2 * ffn_mult * d * d

    per_layer = qkv + attn_logits + attn_softmax + attn_values + out_proj + mlp
    logits = 2 * d * V

    fwd = L * per_layer + logits
    total = 3.0 * fwd if include_backward else float(fwd)

    dense_only = L * (qkv + out_proj + mlp) + logits
    attn_only = L * (attn_logits + attn_softmax + attn_values)

    return {
        "forward": float(fwd),
        "total": total,
        "frac_attention": attn_only / (dense_only + attn_only),
    }

def flops_6nd(n_params: float, n_tokens: float) -> float:
    """The 6ND approximation [FOR REFERENCE ONLY]"""
    return 6.0 * n_params * n_tokens

@dataclass(frozen=True)
class Shape:
    n_layer: int
    n_head: int
    n_embd: int

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def __str__(self) -> str:
        return f"L{self.n_layer}-H{self.n_head}-d{self.n_embd}"

def shape_for_target_params(
    n_target: float,
    *,
    aspect_ratio: int = 64,
    head_dim: int = 64,
    d_multiple: int = 64,
    min_layers: int = 2,
) -> Shape:
    """Pick (n_layer, n_head, n_embd) hitting ~n_target non-embedding params

    Holds width/depth ratio fixed at aspect_ratio (d_model / n_layer),
    so model shape is not a free variable confounded with size
    """
    d = (n_target * aspect_ratio / 12.0) ** (1.0 / 3.0)
    d = max(d_multiple, int(round(d / d_multiple)) * d_multiple)
    n_layer = max(min_layers, int(round(d / aspect_ratio)))
    n_head = max(1, d // head_dim)
    return Shape(n_layer=n_layer, n_head=n_head, n_embd=d)
