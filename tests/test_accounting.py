"""The analytic parameter count must equal the live module, exactly.

If flops.param_counts and model.GPT ever disagree, N is mis-measured and every
fitted scaling exponent inherits the error. This is the one test that has to
pass before any GPU time is spent.

    python tests/test_accounting.py
"""

from __future__ import annotations

import itertools
import os
import sys

def _repo_root() -> str:
    """Find the directory holding flops.py, whatever the layout.

    This file works both at tests/test_accounting.py in a structured repo and
    sitting flat beside the modules it tests, which is what a provisional
    working directory looks like before the real repo exists. Hardcoding
    "one level up" adds the wrong directory to sys.path in the flat case and
    fails to import a module that is right next door.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.dirname(here), os.getcwd()):
        if os.path.isfile(os.path.join(cand, "flops.py")):
            return cand
    raise SystemExit(
        "cannot locate flops.py from " + here + ", its parent, or the cwd.\n"
        "Run this from the directory holding flops.py and model.py."
    )


ROOT = _repo_root()
sys.path.insert(0, ROOT)

import torch

from flops import flops_6nd, param_counts, shape_for_target_params
from model import GPT, GPTConfig


def test_param_counts_match_module():
    for pos_emb, bias, tie in itertools.product(
        ["rope", "learned"], [False, True], [True, False]
    ):
        for (L, H, d) in [(2, 1, 64), (5, 5, 320), (12, 12, 768)]:
            cfg = GPTConfig(
                n_layer=L, n_head=H, n_embd=d, block_size=256, vocab_size=8,
                pos_emb=pos_emb, bias=bias, tie_embeddings=tie,
            )
            rep = GPT(cfg).param_report()
            assert rep["n_total"] == rep["actual_total"], (pos_emb, bias, tie, L, d, rep)
            assert rep["n_non_embedding"] == rep["actual_non_embedding"], (
                pos_emb, bias, tie, L, d, rep)


def test_flops_vs_6nd_gap_is_material():
    """Document, don't assume: 6ND under-counts by tens of percent here."""
    cfg = GPTConfig(n_layer=5, n_head=5, n_embd=320, block_size=1024)
    m = GPT(cfg)
    n = m.param_report()["n_non_embedding"]
    exact = m.flops_per_token()["total"]
    assert exact > flops_6nd(n, 1.0)
    assert m.flops_per_token()["frac_attention"] > 0.2


def test_forward_backward_runs_and_shapes():
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=128, vocab_size=8)
    m = GPT(cfg)
    x = torch.randint(0, 8, (3, 128))
    logits, loss = m(x, x)
    assert logits.shape == (3, 128, 8)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_shape_ladder_is_monotone():
    ns = []
    for t in [1e6, 1e7, 1e8, 1e9]:
        s = shape_for_target_params(t)
        ns.append(param_counts(s.n_layer, s.n_embd, 8, 1024)["n_non_embedding"])
        assert s.n_embd % 64 == 0 and s.n_embd % s.n_head == 0
    assert ns == sorted(ns)


def test_tokenizer_roundtrip_and_revcomp():
    from prepare_data import decode, encode, revcomp_ids

    ids, mask = encode("ACGTacgtNn")
    assert list(ids) == [0, 1, 2, 3, 0, 1, 2, 3, 4, 4]
    assert list(mask) == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    assert decode(ids[:4]) == "ACGT"
    assert decode(revcomp_ids(encode("ACGT")[0])) == "ACGT"   # palindrome
    assert decode(revcomp_ids(encode("AAAC")[0])) == "GTTT"


def test_slurm_embedded_python_compiles():
    """Every python heredoc inside a SLURM script must parse.

    A syntax error in an embedded fragment is invisible until the job reaches
    a GPU node, which can be hours of queue away, and it fails AFTER the
    expensive part has already run. This has happened twice: once from a
    backslash escape being interpreted by the layer that wrote the file,
    turning print(f"\\n ...") into an unterminated f-string.
    """
    import glob
    import re

    scripts = sorted(glob.glob(os.path.join(ROOT, "scripts", "*.slurm")))
    if not scripts:
        print(f"    (no scripts/*.slurm under {ROOT} -- nothing to check)")
        return

    checked = 0
    for path in scripts:
        src = open(path).read()
        for m in re.finditer(r"<<'?(\w+)'?\s*$(.*?)^\1$", src, re.M | re.S):
            line = src[: m.start()].count("\n") + 1
            try:
                compile(m.group(2), f"{os.path.basename(path)}:{line}", "exec")
            except SyntaxError as e:
                raise AssertionError(
                    f"{os.path.basename(path)} heredoc at line {line}: "
                    f"{e.msg} (fragment line {e.lineno})"
                ) from None
            checked += 1
    if not checked:
        print("    (scripts present but contain no python heredocs)")


def test_slurm_scripts_do_not_use_srun():
    """srun inside these single-task jobs is a liability, not a help.

    With --mem set at job level and a cluster default of --mem-per-cpu or
    --mem-per-gpu, srun inherits both and dies with 'SLURM_MEM_PER_CPU,
    SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive'. One
    GPU and one task need no job step, so the fix is to not create one. This
    guards sweep.slurm in particular, where it would kill all 90 array tasks.
    """
    import glob

    scripts = sorted(glob.glob(os.path.join(ROOT, "scripts", "*.slurm")))
    if not scripts:
        print(f"    (no scripts/*.slurm under {ROOT} -- nothing to check)")
        return
    for path in scripts:
        for i, line in enumerate(open(path), 1):
            bare = line.strip()
            if bare.startswith("#"):
                continue
            assert "srun " not in bare, (
                f"{os.path.basename(path)}:{i} uses srun -- see the docstring"
            )


def test_no_script_pins_a_card_specific_peak_flops():
    """No script may hard-set GDNA_GPU_PEAK_FLOPS or pass --gpu-peak-flops.

    env.sh once did `: "${GDNA_GPU_PEAK_FLOPS:=500e12}"`. Because `:=`
    ASSIGNS, that value then won over every ${GDNA_GPU_PEAK_FLOPS:-989e12}
    default downstream, so a run on an H100 computed MFU against the RTX PRO
    6000 Blackwell peak and reported 99.2% instead of its true 50.1% -- a
    silent factor of 989/500. train.py now resolves the peak from the live
    device via design.peak_flops_for_device(), which cannot disagree with the
    card that actually ran.
    """
    import glob

    bad = []
    for path in sorted(glob.glob(os.path.join(ROOT, "scripts", "*"))):
        if os.path.isdir(path):
            continue
        for i, line in enumerate(open(path, errors="replace"), 1):
            if line.lstrip().startswith("#"):
                continue
            if "GDNA_GPU_PEAK_FLOPS:=" in line or "--gpu-peak-flops" in line:
                bad.append(f"{os.path.basename(path)}:{i}: {line.strip()[:70]}")
    assert not bad, "card-specific peak FLOPS pinned in:\n  " + "\n  ".join(bad)


def test_device_peak_lookup_covers_bouchet_cards():
    """Every GPU type Bouchet exposes must resolve to a peak."""
    from design import peak_flops_for_device

    expected = {
        "NVIDIA H100 80GB HBM3": 989e12,
        "NVIDIA H200": 989e12,
        "NVIDIA B200": 2250e12,
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": 500e12,
        "NVIDIA L40S": 362e12,
        "NVIDIA A40": 149.7e12,
        "NVIDIA RTX 5000 Ada Generation": 261e12,
        "NVIDIA RTX A5000": 111.6e12,
    }
    for name, peak in expected.items():
        got = peak_flops_for_device(name)
        assert got is not None, f"{name} does not resolve"
        assert got[1] == peak, f"{name}: got {got[1]:.3g}, expected {peak:.3g}"
    assert peak_flops_for_device("NVIDIA GeForce RTX 4090") is None


if __name__ == "__main__":
    print(f"repo root: {ROOT}")
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
