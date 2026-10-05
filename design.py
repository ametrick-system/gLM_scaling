"""IsoFLOP experiment designer"""

from __future__ import annotations

import itertools
import json
import os
import sys
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flops import flops_per_token, param_counts, shape_for_target_params

HUMAN_GENOME_TOKENS = 3.1e9 # hg38 non-N, single strand
MAMMAL_GENOME_TOKENS = 2.7e9 # rough Zoonomia average

@dataclass
class Run:
    budget_flops: float
    size_index: int
    n_layer: int
    n_head: int
    n_embd: int
    n_non_embedding: int
    n_total: int
    block_size: int
    tokens: int
    flops_per_token: float
    tokens_per_param: float
    frac_attention_flops: float
    spread: str
    seed: int

    @property
    def run_id(self) -> str:
        return (
            f"C{self.budget_flops:.0e}".replace("+", "")
            + f"_N{self.n_non_embedding/1e6:.1f}M"
            + f"_{self.spread}_s{self.seed}"
        )


def chinchilla_optimal_n(budget: float, tokens_per_param: float = 20.0) -> float:
    """N such that C = 6*N*D with D = tokens_per_param * N."""
    return (budget / (6.0 * tokens_per_param)) ** 0.5


def build_grid(
    budgets: list[float],
    *,
    block_size: int = 1024,
    vocab_size: int = 8,
    n_sizes: int = 5,
    size_span: float = 4.0, # ratio of largest to Chinchilla-opt
    spread_levels: tuple[str, ...] = ("narrow", "medium", "broad"),
    seeds: tuple[int, ...] = (0,), # 1 seed per point
    tokens_per_param: float = 20.0,
    aspect_ratio: int = 64,
) -> list[Run]:
    runs: list[Run] = []
    lo = 1.0 / size_span
    mults = [lo * (size_span**2) ** (i / (n_sizes - 1)) for i in range(n_sizes)]

    for budget in budgets:
        n_opt = chinchilla_optimal_n(budget, tokens_per_param)
        for si, m in enumerate(mults):
            shape = shape_for_target_params(
                n_opt * m, aspect_ratio=aspect_ratio
            )
            pc = param_counts(
                shape.n_layer, shape.n_embd, vocab_size, block_size
            )
            fpt = flops_per_token(
                shape.n_layer, shape.n_head, shape.n_embd,
                block_size, vocab_size,
            )
            tokens = int(budget / fpt["total"])
            for spread, seed in itertools.product(spread_levels, seeds):
                runs.append(
                    Run(
                        budget_flops=budget,
                        size_index=si,
                        n_layer=shape.n_layer,
                        n_head=shape.n_head,
                        n_embd=shape.n_embd,
                        n_non_embedding=pc["n_non_embedding"],
                        n_total=pc["n_total"],
                        block_size=block_size,
                        tokens=tokens,
                        flops_per_token=fpt["total"],
                        tokens_per_param=tokens / pc["n_non_embedding"],
                        frac_attention_flops=fpt["frac_attention"],
                        spread=spread,
                        seed=seed,
                    )
                )
    return runs


def noise_floor_design(
    budgets: list[float],
    *,
    n_seeds_at_min: int = 3,
    **kw,
) -> list[Run]:
    cheapest = min(budgets)
    runs: list[Run] = []
    for b in budgets:
        n = n_seeds_at_min if b == cheapest else 1
        runs += build_grid([b], seeds=tuple(range(n)), **kw)
    return runs

GPUS: dict[str, tuple[float, float]] = {
    "h100-sxm":            (989e12, 80.0),
    "a100-sxm":            (312e12, 80.0),
    "rtx-pro-6000-bw":     (500e12, 96.0),   # Blackwell, GDDR7, bf16+TF32+FP8
    "rtx-6000-ada":        (364e12, 48.0),
    "rtx-a6000":           (155e12, 48.0),
    "quadro-rtx-6000":     (65e12,  24.0),   # Turing: no bf16, no TF32
    # Bouchet's other cards
    "h200":                (989e12, 141.0),  # same compute as H100, more HBM
    "b200":                (2250e12, 180.0),
    "l40s":                (362e12, 48.0),
    "a40":                 (149.7e12, 48.0),
    "rtx-5000-ada":        (261e12, 32.0),
    "a5000":               (111.6e12, 24.0),
}

_DEVICE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("rtx pro 6000",   "rtx-pro-6000-bw"),
    ("rtx 6000 ada",   "rtx-6000-ada"),
    ("rtx 5000 ada",   "rtx-5000-ada"),
    ("rtx a6000",      "rtx-a6000"),
    ("rtx a5000",      "a5000"),
    ("a100",           "a100-sxm"),
    ("h100",           "h100-sxm"),
    ("h200",           "h200"),
    ("b200",           "b200"),
    ("l40s",           "l40s"),
    ("a40",            "a40"),
    ("quadro rtx 6000", "quadro-rtx-6000"),
)

def peak_flops_for_device(device_name: str) -> tuple[str, float] | None:
    low = device_name.lower()
    for needle, key in _DEVICE_PATTERNS:
        if needle in low:
            return key, GPUS[key][0]
    return None


def cost_report(
    runs: list[Run],
    *,
    gpu: str | None = None,
    gpu_peak_flops: float = 989e12,   # H100 SXM, bf16 dense
    mfu: float = 0.35,
) -> dict:
    if gpu is not None:
        gpu_peak_flops = GPUS[gpu][0]
    total = sum(r.budget_flops for r in runs)
    eff = gpu_peak_flops * mfu
    return {
        "n_runs": len(runs),
        "total_flops": total,
        "assumed_gpu_peak_flops": gpu_peak_flops,
        "assumed_mfu": mfu,
        "gpu_hours": total / eff / 3600.0,
        "gpu_days": total / eff / 86400.0,
        "largest_run_gpu_hours": max(r.budget_flops for r in runs) / eff / 3600.0,
        "max_tokens_single_run": max(r.tokens for r in runs),
        "max_unique_genomes_needed": max(r.tokens for r in runs)
        / MAMMAL_GENOME_TOKENS,
    }


def data_constraint_crossover(
    block_size: int = 1024,
    vocab_size: int = 8,
    tokens_per_param: float = 20.0,
    unique_tokens: float = HUMAN_GENOME_TOKENS,
    aspect_ratio: int = 64,
) -> dict:
    """Compute budget at which Chinchilla-optimal D exceeds one genome"""
    n_star = unique_tokens / tokens_per_param
    shape = shape_for_target_params(n_star, aspect_ratio=aspect_ratio)
    fpt = flops_per_token(
        shape.n_layer, shape.n_head, shape.n_embd, block_size, vocab_size
    )["total"]
    return {
        "unique_tokens": unique_tokens,
        "n_at_crossover": n_star,
        "shape_at_crossover": str(shape),
        "crossover_flops": fpt * unique_tokens,
    }

def write_grid(runs: list[Run], path: str) -> None:
    with open(path, "w") as fh:
        for r in runs:
            fh.write(json.dumps({"run_id": r.run_id, **asdict(r)}) + "\n")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--budgets", nargs="*", type=float,
                   default=[1e16, 1e17, 1e18, 1e19])
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--seeds", nargs="*", type=int, default=[0])
    p.add_argument("--noise-floor", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="replicate the cheapest budget's whole ladder 3x "
                        "(+0.2%% compute) to estimate run-to-run variance "
                        "(default: on)")
    p.add_argument("--out", default="runs/grid.jsonl")
    a = p.parse_args()
    if a.noise_floor:
        runs = noise_floor_design(a.budgets, block_size=a.block_size)
    else:
        runs = build_grid(a.budgets, block_size=a.block_size,
                          seeds=tuple(a.seeds))
    write_grid(runs, a.out)
    rep = cost_report(runs)
    rep["budgets"] = a.budgets
    rep["seeds"] = "noise_floor" if a.noise_floor else a.seeds
    print(json.dumps(rep, indent=2))
    print(json.dumps(data_constraint_crossover(a.block_size), indent=2))
