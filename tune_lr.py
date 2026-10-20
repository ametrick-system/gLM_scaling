"""
Tune the peak LR(N)

LR is fit as a power law log(LR_opt) = a + b*log(N), tuned once on the human-only
corpus and then frozen and applied identically to every (C, N, spread) run

LR_opt is tuned at a fixed moderate batch and applied across D on purpose: below
the critical batch size LR_opt is batch-insensitive
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from design import Run
from flops import flops_per_token, param_counts, shape_for_target_params

# fixed tuning constants
BLOCK = 1024
VOCAB = 8
ASPECT = 64
TUNE_TOKENS = 300_000_000 
TUNE_BATCH = 262_144 # fixed tokens/step for the LR sweep
# Small N anchors along the real 1e16/1e17 ladder (non-embedding params)
LR_ANCHOR_N = [3.1e6, 6.1e6, 10.6e6, 25.2e6]
LR_GRID = [3.0e-4, 5.0e-4, 8.0e-4, 1.3e-3, 2.0e-3, 3.0e-3]
# Held-out N for the extrapolation check
VALIDATE_N = 100.0e6
# Batch sweep: one anchor, vary the effective optimiser batch (tokens/step)
BATCH_ANCHOR_N = 25.2e6
BATCH_TOKENS = 600_000_000
BATCH_GRID = [65_536, 131_072, 262_144, 524_288, 1_048_576]


def _run_for(n_target: float, tokens: int, tag: str) -> Run:
    """A Run spec at the given non-embedding N and token budget (seed 0)."""
    s = shape_for_target_params(n_target, aspect_ratio=ASPECT)
    pc = param_counts(s.n_layer, s.n_embd, VOCAB, BLOCK)
    fpt = flops_per_token(s.n_layer, s.n_head, s.n_embd, BLOCK, VOCAB)
    return Run(
        budget_flops=tokens * fpt["total"], # not an isoFLOP point; informational
        size_index=0,
        n_layer=s.n_layer, n_head=s.n_head, n_embd=s.n_embd,
        n_non_embedding=pc["n_non_embedding"], n_total=pc["n_total"],
        block_size=BLOCK, tokens=tokens, flops_per_token=fpt["total"],
        tokens_per_param=tokens / pc["n_non_embedding"],
        frac_attention_flops=fpt["frac_attention"],
        spread=tag, seed=0,
    )


def emit(a: argparse.Namespace) -> None:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid_path = out / f"{a.mode}_grid.jsonl"
    jobs_path = out / f"{a.mode}_jobs.tsv"

    specs: list[Run] = []
    jobs: list[dict] = []  # one per training run

    if a.mode == "lr":
        for n in LR_ANCHOR_N:
            line = len(specs)
            specs.append(_run_for(n, TUNE_TOKENS, tag=f"lrtune_N{n/1e6:.1f}M"))
            for lr in LR_GRID:
                jobs.append(dict(line=line, lr=lr, tokens_per_step=TUNE_BATCH,
                                 tag=f"N{n/1e6:.1f}M_lr{lr:.1e}"))
        # the held-out validation point, emitted so it can be run after the fit
        vline = len(specs)
        specs.append(_run_for(VALIDATE_N, TUNE_TOKENS, tag=f"validate_N{VALIDATE_N/1e6:.0f}M"))
        for lr in ("PREDICTED", "PREDICTED/1.6", "PREDICTED*1.6"):
            jobs.append(dict(line=vline, lr=lr, tokens_per_step=TUNE_BATCH,
                             tag=f"validate_N{VALIDATE_N/1e6:.0f}M_{lr}"))
    elif a.mode == "batch":
        if a.lr is None:
            raise SystemExit("emit --mode batch needs --lr (use the fitted LR_opt "
                             f"for N={BATCH_ANCHOR_N/1e6:.0f}M)")
        specs.append(_run_for(BATCH_ANCHOR_N, BATCH_TOKENS, tag="batchtune"))
        for b in BATCH_GRID:
            jobs.append(dict(line=0, lr=a.lr, tokens_per_step=b,
                             tag=f"batch{b//1024}k"))
    else:
        raise SystemExit(f"unknown mode {a.mode}")

    with grid_path.open("w") as fh:
        for r in specs:
            fh.write(json.dumps({"run_id": r.run_id, **asdict(r)}) + "\n")
    with jobs_path.open("w") as fh:
        fh.write("line\tlr\ttokens_per_step\ttag\n")
        for j in jobs:
            fh.write(f"{j['line']}\t{j['lr']}\t{j['tokens_per_step']}\t{j['tag']}\n")

    print(f"wrote {grid_path} ({len(specs)} specs)")
    print(f"wrote {jobs_path} ({len(jobs)} runs)")
    if a.mode == "lr":
        print("PREDICTED rows are placeholders: run the fit first, substitute the "
              "predicted LR, then run those 3 validation rows")


def _collect(runs_dir: Path) -> list[dict]:
    """Read every final.json under runs_dir; skip truncated/timing runs"""
    rows = []
    for fj in sorted(Path(runs_dir).rglob("final.json")):
        d = json.loads(fj.read_text())
        meta = json.loads((fj.parent / "meta.json").read_text())
        rows.append(dict(
            N=d["n_non_embedding"],
            lr=meta["optim"]["lr"],
            tokens_per_step=meta["batching"]["tokens_per_step"],
            val_bits=d["val_bits_per_nt"],
            run_id=d["run_id"],
        ))
    return rows

def fit(a: argparse.Namespace) -> None:
    import numpy as np

    rows = _collect(a.runs)
    if not rows:
        raise SystemExit(f"no final.json under {a.runs} (only truncated runs?)")

    if a.mode == "lr":
        byN: dict[float, list] = {}
        for r in rows:
            byN.setdefault(r["N"], []).append(r)
        anchors = []
        edge_sizes = []
        print(f"{'N':>10} {'LR_opt':>10} {'(argmin)':>10}   sweep (lr: bits)")
        for N in sorted(byN):
            pts = sorted(byN[N], key=lambda r: r["lr"])
            amin = min(pts, key=lambda r: r["val_bits"])
            # continuous optimum = vertex of a parabola fit to val-loss vs log(LR)
            x = np.log([p["lr"] for p in pts])
            y = np.array([p["val_bits"] for p in pts])
            c2, c1, _ = np.polyfit(x, y, 2)
            if c2 > 0:                                   # opens upward -> real min
                lr_opt = float(np.exp(-c1 / (2 * c2)))
                lr_opt = min(max(lr_opt, pts[0]["lr"]), pts[-1]["lr"])  # clamp to range
            else:                                        # not convex! -> fall back
                lr_opt = amin["lr"]
            at_edge = amin["lr"] in (pts[0]["lr"], pts[-1]["lr"])
            if at_edge:
                edge_sizes.append(N / 1e6)
            anchors.append((N, lr_opt))
            sweep = "  ".join(f"{p['lr']:.1e}:{p['val_bits']:.3f}" for p in pts)
            flag = "  <-argmin@edge" if at_edge else ""
            print(f"{N/1e6:9.1f}M {lr_opt:10.2e} {amin['lr']:10.1e}   {sweep}{flag}")

        if len(anchors) < 2:
            raise SystemExit("need >= 2 anchor sizes to fit a power law")
        logN = np.log([n for n, _ in anchors])
        logL = np.log([lr for _, lr in anchors])
        b, a0 = np.polyfit(logN, logL, 1)          # logL = b*logN + a0
        resid = logL - (b * logN + a0)
        print(f"\n  fit: LR_opt = {math.exp(a0):.3e} * N^{b:.3f}")
        print(f"  (log-space residuals: {np.round(resid, 3).tolist()}, "
              f"max |resid| {np.max(np.abs(resid)):.3f})")
        pred = math.exp(a0) * VALIDATE_N ** b
        print(f"\n  PREDICTION for the held-out N={VALIDATE_N/1e6:.0f}M: LR_opt = {pred:.3e}")
        print(f"  Now run the 3 validation rows around this value and re-fit.")
        if edge_sizes:
            print(f"\n  WARNING: argmin at grid edge for N(M)={edge_sizes} -- the "
                  f"optimum is not bracketed; widen LR_GRID and re-run")
    elif a.mode == "batch":
        pts = sorted(rows, key=lambda r: r["tokens_per_step"])
        print(f"batch sweep at N={pts[0]['N']/1e6:.0f}M, LR={pts[0]['lr']:.2e}:")
        print(f"  {'batch (tok)':>12} {'val bits/nt':>12}")
        best = min(pts, key=lambda r: r["val_bits"])
        for p in pts:
            flag = "  <- best" if p is best else ""
            print(f"  {p['tokens_per_step']:12d} {p['val_bits']:12.4f}{flag}")
        print(f"\n  Critical batch is the largest batch whose val loss is within "
              f"noise of the best. If loss RISES past some batch, the D/4000 rule "
              f"must cap below it; if flat to 1M, the rule's 1M ceiling is safe.")

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit", help="write tuning grid + job table")
    e.add_argument("--mode", choices=["lr", "batch"], required=True)
    e.add_argument("--out-dir", default="tune")
    e.add_argument("--lr", type=float, default=None, help="fixed LR for batch mode")
    e.set_defaults(func=emit)

    f = sub.add_parser("fit", help="fit power law / read batch sweep")
    f.add_argument("--runs", required=True, help="dir of completed run subdirs")
    f.add_argument("--mode", choices=["lr", "batch"], required=True)
    f.set_defaults(func=fit)

    a = p.parse_args()
    a.func(a)

if __name__ == "__main__":
    main()
