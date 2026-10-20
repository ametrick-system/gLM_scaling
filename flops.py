"""Exact parameter and FLOP accounting"""

from __future__ import annotations
from dataclasses import dataclass

def param_counts(
    n_layers: int,
    d_model: int,
    vocab_size: int,
    block_size: int,
    *,
    ffn_mult: int = 4,
    pos_emb: str = "rope",
    tie_embeddings: bool = True,
    bias: bool = False,
) -> dict[str, int]:
    """
    Parameter counts, split into embedding and non-embedding
    Exact agreement with the live module is checked by tests/test_accounting.py
    """
    d = d_model
    d_ff = ffn_mult * d

    bias_param = 0
    if bias:
        attention_bias = 4 * d # d for each of q, k, v, out projection
        ff_bias = d_ff + d # one bias param for each output and input to ff
        ln_bias = 2 * d # 2 LayerNorms per block
        bias_param += n_layers * (attention_bias + ff_bias + ln_bias) + d # +d for final LayerNorm 

    # non-embedding parameters
    attention_param = 4 * d * d # q, k, v, out
    ff_param = 2 * d_ff * d # d -> d_ff then d_ff -> d
    ln_param = 2 * d

    per_layer = attention_param + ff_param + ln_param

    non_embedding_param = n_layers * per_layer + d + bias_param # +d for final LayerNorm

    # embedding parameters
    token_embedding_param = vocab_size * d 
    position_embedding_param = block_size * d if pos_emb == "learned" else 0 # 0 if RoPE (default)
    head_param = 0 if tie_embeddings else vocab_size * d # extra matrix if embedding weights not shared with head
    
    embedding_param = token_embedding_param + position_embedding_param + head_param

    return {
        "n_non_embedding": non_embedding_param,
        "n_embedding": embedding_param,
        "n_total": non_embedding_param + embedding_param,
        "n_per_layer": per_layer, # without bias
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
