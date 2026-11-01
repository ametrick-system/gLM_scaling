"""Preflight check. Run this as your first GPU job, before anything else.

    srun --gpus=1 --time=10 python scripts/check_env.py

Checks, in the order they can bite you:

1. Does a CUDA op actually run? On Blackwell (sm_120/sm_122) a PyTorch built
   for CUDA <= 12.6 imports fine, sees the GPU, and then fails with either
   "sm_120 is not compatible with the current PyTorch installation" or
   "no kernel image is available for execution on the device". `nvidia-smi`
   working proves nothing -- it talks to the driver, not to PyTorch's kernels.
2. Does bf16 work? Blackwell has bf16, TF32 and FP8 tensor cores, so if bf16
   fails something is wrong with the build, not the hardware.
3. Which SDPA backend is live, and what does attention cost in memory?
   This one is load-bearing for the experiment design. The activation-memory
   estimates that set the micro-batch per model size assume a flash-style
   kernel that never materialises the T x T score matrix. If SDPA silently
   falls back to the MATH backend, attention allocates B*H*T*T per layer --
   at T=1024 that is an order of magnitude more memory, and the micro-batch
   sizes in the design are wrong. Note that PyTorch's built-in FLASH_ATTENTION
   backend is NOT the Dao-AILab `flash-attn` package; do not install that one,
   it does not support sm_120 and nanoGPT does not use it.
4. Does torch.compile work? It is where cluster toolchain problems surface.

Prints a JSON block to paste into your run records. Every number in the
scaling-law fit is conditional on this stack, so it belongs in meta.json.
"""

from __future__ import annotations

import json
import platform
import sys

import torch


def _hdr(s: str) -> None:
    print(f"\n{'=' * 4} {s} {'=' * (58 - len(s))}")


def main() -> int:
    fail: list[str] = []
    env: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }

    _hdr("1. device and kernel availability")
    if not torch.cuda.is_available():
        print("FAIL torch.cuda.is_available() is False -- no GPU visible")
        return 1
    cap = torch.cuda.get_device_capability()
    env["device"] = torch.cuda.get_device_name(0)
    env["capability"] = f"sm_{cap[0]}{cap[1]}"
    env["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    env["n_visible_gpus"] = torch.cuda.device_count()
    print(f"  device      {env['device']}")
    print(f"  capability  {env['capability']}")
    print(f"  vram        {env['vram_gb']} GB   visible GPUs {env['n_visible_gpus']}")
    print(f"  torch       {env['torch']} (built for CUDA {env['torch_cuda_build']})")
    try:
        # The real test: a kernel launch, not just a device query.
        x = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        assert float((x @ x).sum()) == float((x @ x).sum())
        print("  PASS fp32 matmul executes")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL CUDA kernel launch: {type(e).__name__}: {e}")
        print("       -> your torch wheel has no kernels for this architecture.")
        print("       -> reinstall from a CUDA 12.8+ index, e.g.")
        print("          pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128")
        return 1

    _hdr("2. bf16")
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prod = x @ x
        torch.cuda.synchronize()
        env["bf16"] = True
        env["bf16_autocast_dtype"] = str(prod.dtype)
        print(f"  PASS bf16 autocast matmul executes, output dtype {prod.dtype}")
        del prod
    except Exception as e:  # noqa: BLE001
        env["bf16"] = False
        fail.append("bf16")
        print(f"  FAIL bf16: {type(e).__name__}: {e}")
    del x
    torch.cuda.empty_cache()

    _hdr("3. SDPA backends and attention memory")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, H, T, D = 4, 12, 1024, 64
    q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3))
    backends = {
        "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
        "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
        "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
        "MATH": SDPBackend.MATH,
    }
    results: dict[str, object] = {}
    for name, be in backends.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with sdpa_kernel(be):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True)
                out.sum().backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e6
            results[name] = round(peak, 1)
            print(f"  ok        {name:<20} peak {peak:8.1f} MB")
        except Exception as e:  # noqa: BLE001
            results[name] = None
            print(f"  unusable  {name:<20} {type(e).__name__}")
        for t in (q, k, v):
            t.grad = None
    env["sdpa_peak_mb"] = results

    # The T x T score matrix is B*H*T*T elements; a flash-style kernel never
    # materialises it, so its peak should be far below MATH's.
    math_mb = results.get("MATH")
    fast = [n for n in ("CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION")
            if results.get(n) is not None]
    if not fast:
        fail.append("sdpa")
        print("\n  FAIL no memory-efficient SDPA backend is usable on this GPU.")
        print("       -> attention will materialise the T x T score matrix and")
        print("          the micro-batch sizes in the design are too large.")
        print("          Re-derive them, or reduce block_size.")
    else:
        chosen = min(fast, key=lambda n: results[n])
        env["sdpa_backend"] = chosen
        if math_mb:
            print(f"\n  PASS {chosen} is {math_mb / results[chosen]:.1f}x leaner "
                  f"than MATH -- flash-style kernel confirmed")
        else:
            print(f"\n  PASS {chosen} usable (MATH unavailable for comparison)")
    del q, k, v
    torch.cuda.empty_cache()

    _hdr("4. torch.compile")
    try:
        m = torch.nn.Sequential(
            torch.nn.Linear(256, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 256)
        ).cuda()
        cm = torch.compile(m)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cm(torch.randn(8, 256, device="cuda")).sum().backward()
        torch.cuda.synchronize()
        env["torch_compile"] = True
        print("  PASS torch.compile forward+backward executes")
    except Exception as e:  # noqa: BLE001
        env["torch_compile"] = False
        print(f"  WARN torch.compile failed: {type(e).__name__}: {e}")
        print("       -> not fatal; run with --compile off, but you lose ~1.3-2x")

    _hdr("record this in every run's meta.json")
    print(json.dumps(env, indent=2))

    if fail:
        print(f"\n{len(fail)} blocking problem(s): {', '.join(fail)}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
