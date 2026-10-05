"""Training loop for one isoFLOP point"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from design import peak_flops_for_device
from model import GPT, GPTConfig

LN2 = math.log(2.0)

class MemmapTokens:
    """Sampling from flat uint8 token file (uniformly at random)"""

    def __init__(self, path: str, block_size: int, seed: int = 0):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        assert self.data.size > block_size + 1, f"{path} too small"

    @property
    def n_unique_tokens(self) -> int:
        return int(self.data.size)

    def batch(self, batch_size: int, device: str):
        ix = self.rng.integers(
            0, self.data.size - self.block_size - 1, size=batch_size
        )
        x = np.stack([self.data[i : i + self.block_size] for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix])
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        if device.startswith("cuda"):
            return (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        return x.to(device), y.to(device)


class SequentialTokens:
    """Deterministic non-overlapping crops -- used for validation"""

    def __init__(self, path: str, block_size: int):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.block_size = block_size

    def batches(self, batch_size: int, device: str, max_batches: int | None = None):
        T = self.block_size
        n = (self.data.size - 1) // T
        if max_batches is not None:
            n = min(n, max_batches * batch_size)
        for s in range(0, n, batch_size):
            idx = range(s, min(s + batch_size, n))
            x = np.stack([self.data[i * T : i * T + T] for i in idx])
            y = np.stack([self.data[i * T + 1 : i * T + T + 1] for i in idx])
            yield (
                torch.from_numpy(x.astype(np.int64)).to(device),
                torch.from_numpy(y.astype(np.int64)).to(device),
            )

def lr_at(step: int, total_steps: int, lr: float, min_lr: float,
          warmup_frac: float = 0.01) -> float:
    warmup = max(1, int(warmup_frac * total_steps))
    if step < warmup:
        return lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total_steps - warmup)
    prog = min(1.0, prog)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, val: SequentialTokens, batch_size: int, device: str,
             ctx, max_batches: int = 50) -> dict:
    model.eval()
    tot, n = 0.0, 0
    for x, y in val.batches(batch_size, device, max_batches=max_batches):
        with ctx:
            _, loss = model(x, y)
        tot += loss.item() * x.numel()
        n += x.numel()
    model.train()
    nats = tot / max(1, n)
    return {"val_loss_nats": nats, "val_bits_per_nt": nats / LN2,
            "val_tokens_scored": n}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="grid.jsonl")
    p.add_argument("--line", type=int, required=True, help="0-based row")
    p.add_argument("--train-bin", required=True)
    p.add_argument("--val-bin", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--tokens-per-step", type=int, default=524_288,
                   help="global batch in tokens; held fixed across the sweep")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--save-ckpt", action="store_true",
                   help="write ckpt.pt (default: only final.json and log.jsonl)")
    p.add_argument("--gpu-peak-flops", type=float, default=None,
                   help="dense bf16 peak FLOP/s; enables measured-MFU logging")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop early (timing/debug); result is NOT a valid "
                        "isoFLOP point and is written to final_TRUNCATED.json")
    a = p.parse_args()

    spec = json.loads(a.config.read_text().splitlines()[a.line])
    run_id = spec["run_id"]
    out = a.out_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dev_type = "cuda" if a.device.startswith("cuda") else "cpu"
    ctx = (
        nullcontext() if dev_type == "cpu"
        else torch.amp.autocast(device_type="cuda",
                                dtype=getattr(torch, a.dtype))
    )

    cfg = GPTConfig(
        block_size=spec["block_size"],
        vocab_size=8,
        n_layer=spec["n_layer"],
        n_head=spec["n_head"],
        n_embd=spec["n_embd"],
        dropout=0.0,
    )
    model = GPT(cfg).to(a.device)
    if a.compile:
        model = torch.compile(model)

    opt = model.configure_optimizers(
        a.weight_decay, a.lr, (a.beta1, a.beta2), dev_type
    )

    train = MemmapTokens(a.train_bin, cfg.block_size, seed=spec["seed"] + 1000)
    val = SequentialTokens(a.val_bin, cfg.block_size)

    tokens_per_micro = a.micro_batch * cfg.block_size
    accum = max(1, a.tokens_per_step // tokens_per_micro)
    tokens_per_step = accum * tokens_per_micro
    total_steps = max(1, spec["tokens"] // tokens_per_step)
    min_lr = a.lr * a.min_lr_frac

    # -- repetition bookkeeping for the D' fit -------------------------
    u_d = train.n_unique_tokens
    r_d = spec["tokens"] / u_d - 1.0

    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    peak_src, peak = "none", None
    resolved = peak_flops_for_device(dev_name) if dev_name else None
    if resolved is not None:
        peak_key, peak = resolved
        peak_src = f"detected:{peak_key}"
        if a.gpu_peak_flops and abs(a.gpu_peak_flops - peak) / peak > 0.05:
            print(
                f"!! --gpu-peak-flops {a.gpu_peak_flops:.3g} disagrees with "
                f"{dev_name} ({peak:.3g} for {peak_key}); IGNORING the flag "
                f"and using the detected value. MFU would have been off by "
                f"{peak / a.gpu_peak_flops:.2f}x.",
                flush=True,
            )
            peak_src = f"detected:{peak_key}(flag_{a.gpu_peak_flops:.3g}_ignored)"
    elif a.gpu_peak_flops:
        peak, peak_src = a.gpu_peak_flops, "flag:unrecognised_device"
        print(f"   device {dev_name!r} not in the GPUS table; "
              f"using --gpu-peak-flops {peak:.3g} for MFU", flush=True)
    a.gpu_peak_flops = peak

    meta = {
        "run_id": run_id,
        "spec": spec,
        "model_config": model.config_dict() if not a.compile else cfg.__dict__,
        "param_report": (model if not a.compile else GPT(cfg)).param_report(),
        "optim": {
            "lr": a.lr, "min_lr": min_lr, "weight_decay": a.weight_decay,
            "betas": [a.beta1, a.beta2], "grad_clip": a.grad_clip,
            "warmup_frac": a.warmup_frac, "schedule": "cosine_to_token_budget",
        },
        "batching": {
            "micro_batch": a.micro_batch, "grad_accum": accum,
            "tokens_per_step": tokens_per_step, "total_steps": total_steps,
        },
        "data": {
            "train_bin": a.train_bin, "val_bin": a.val_bin,
            "unique_tokens_U_D": u_d,
            "repetitions_R_D": r_d,
            "epochs_equivalent": spec["tokens"] / u_d,
        },
        "truncated": a.max_steps is not None,
        "max_steps": a.max_steps,
        "device": a.device, "dtype": a.dtype, "torch": torch.__version__,
        "hardware": {
            "node": os.environ.get("GDNA_NODE"),
            "gpu": os.environ.get("GDNA_GPU"),
            "slurm_job": os.environ.get("GDNA_SLURM_JOB"),
            "torch_cuda_build": torch.version.cuda,
            "mfu_peak_flops": peak,
            "mfu_peak_source": peak_src,
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None
            ),
            "capability": (
                "sm_%d%d" % torch.cuda.get_device_capability(0)
                if torch.cuda.is_available() else None
            ),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    seen = 0
    win_t, win_tok = t0, 0

    stop_at = total_steps if a.max_steps is None else min(a.max_steps, total_steps)
    for step in range(stop_at):
        lr = lr_at(step, total_steps, a.lr, min_lr, a.warmup_frac)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(accum):
            x, y = train.batch(a.micro_batch, a.device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / accum
            loss.backward()
            acc_loss += loss.item()
            seen += x.numel()
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()

        if step % a.eval_every == 0 or step == stop_at - 1:
            now = time.time()
            win_dt = now - win_t
            win_tokens = seen - win_tok
            tps = win_tokens / win_dt if win_dt > 0 else float("nan")
            mfu = (
                tps * spec["flops_per_token"] / a.gpu_peak_flops
                if a.gpu_peak_flops else None
            )
            ev = evaluate(model, val, a.micro_batch, a.device, ctx)
            rec = {
                "step": step, "tokens_seen": seen, "lr": lr,
                "train_loss_nats": acc_loss,
                "train_bits_per_nt": acc_loss / LN2,
                **ev,
                "flops_so_far": seen * spec["flops_per_token"],
                "wall_s": now - t0,
                "tokens_per_s": tps,
                "mfu": mfu,
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(
                f"{run_id} step {step}/{total_steps} "
                f"train {acc_loss/LN2:.4f} val {ev['val_bits_per_nt']:.4f} bits/nt"
                + (f"  {tps/1e3:.0f}k tok/s" if win_dt > 0 else "")
                + (f"  mfu {mfu:.1%}" if mfu else ""),
                flush=True,
            )
            win_t, win_tok = time.time(), seen

    final = evaluate(model, val, a.micro_batch, a.device, ctx, max_batches=500)
    final_name = "final.json" if a.max_steps is None else "final_TRUNCATED.json"
    (out / final_name).write_text(json.dumps({
        "truncated": a.max_steps is not None,
        "run_id": run_id, "tokens_seen": seen,
        "flops_spent": seen * spec["flops_per_token"],
        "budget_flops": spec["budget_flops"],
        "n_non_embedding": spec["n_non_embedding"],
        "spread": spec["spread"], "seed": spec["seed"],
        "unique_tokens_U_D": u_d, "repetitions_R_D": r_d,
        **final,
    }, indent=2))
    if a.save_ckpt:
        torch.save(
            {"model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
             "config": cfg.__dict__},
            out / "ckpt.pt",
        )
    log.close()
    print(f"done {run_id}: {final['val_bits_per_nt']:.4f} bits/nt")


if __name__ == "__main__":
    main()
