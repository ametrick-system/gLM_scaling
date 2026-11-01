"""
(Loosly) adapted from nanoGPT/data/shakespeare_char/prepare.py

FASTA -> flat uint8 token memmap, flagging repeat elements
"""

from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np

# ------------- Character tokenization (not learned for DNA) -------------------

BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
PAD_VOCAB  = 8 # multiple of 8 for tensor-core matrix multiplication
ID_TO_BASE = {v: k for k, v in BASE_TO_ID.items()}

LOOKUP_TABLE = np.full(256, 4, dtype=np.uint8) # default to ambiguous 'N' = 4

# retain the repeat element information encoded in lowercase bases in some assemblies
REPEAT_LOOKUP_TABLE = np.zeros(256, dtype=np.uint8)

for b, i in BASE_TO_ID.items():
    LOOKUP_TABLE[ord(b)] = i # ASCII of character -> our defined base ID
    LOOKUP_TABLE[ord(b.lower())] = i # remove casing difference when converting ASCII -> ID
    if b != "N":
        REPEAT_LOOKUP_TABLE[ord(b.lower())] = 1 # retain lowercase flag in repeat lookup table

# ------------------------- Encoding and decoding ------------------------------

def encode(seq: str) -> tuple[np.ndarray, np.ndarray]:
    """ASCII DNA -> (token ids, repeat flags)"""
    a = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8) # vectorize for efficiency
    return LOOKUP_TABLE[a], REPEAT_LOOKUP_TABLE[a]

def decode(ids: np.ndarray) -> str:
    """token ids -> ASCII DNA"""
    return "".join(ID_TO_BASE.get(int(i), "N") for i in ids)

def reverse_comp_ids(ids: np.ndarray) -> np.ndarray:
    """Reverse complement of DNA sequence in token space: A <-> T (0 <-> 3), C <-> G (1 <-> 2), N <-> (4 <-> 4), pad (5-7) to themselves"""
    comp = np.array([3, 2, 1, 0, 4, 5, 6, 7], dtype=np.uint8) # base pairs/complement
    return comp[ids[::-1]] # REVERSE

# -------------------------------- IO handling ------------------------------------

# Reconstruct chromosome sequences from FASTA input
def iter_fasta(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    name, chunks = None, []
    with opener(path, "rt") as fh:
        # state machine to reconstruct (chromosome, sequence) pairs
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].strip().split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)

def prepare_out(fasta, out_prefix, *, include=None, exclude=None, min_contig=10_000, drop_ambiguous_runs=100) -> dict:
    """Write <out_prefix>.bin (uint8 tokens), .repeat_mask.bin, and .meta.json."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tok_path  = out_prefix.with_suffix(".bin")
    repeat_mask_path = Path(str(out_prefix) + ".repeat_mask.bin")

    n_written = n_ambig = n_soft = 0
    contigs: list[dict] = []
    base_hist = np.zeros(PAD_VOCAB, dtype=np.int64)

    with open(tok_path, "wb") as ftok, open(repeat_mask_path, "wb") as fmask:
        for name, seq in iter_fasta(fasta):
            if include and name not in include:
                continue
            if exclude and name in exclude:
                continue
            if len(seq) < min_contig:
                continue
            ids, msk = encode(seq)

            if drop_ambiguous_runs > 0:
                # Drop long N-runs (e.g. assembly gaps, centromeres)
                keep = _drop_long_runs(ids == 4, drop_ambiguous_runs)
                ids, msk = ids[keep], msk[keep]

            ftok.write(ids.tobytes())
            fmask.write(msk.tobytes())

            base_hist += np.bincount(ids, minlength=PAD_VOCAB)
            n_ambig += int((ids == 4).sum())
            n_soft  += int(msk.sum()) # soft = lowercase = repeat
            contigs.append({"name": name, "n_tokens": int(ids.size)})
            n_written += int(ids.size)

    meta = {
        "source_fasta": str(fasta),
        "vocab_size": PAD_VOCAB,
        "n_tokens": n_written,
        "n_contigs": len(contigs),
        "frac_ambiguous": n_ambig / max(1, n_written),
        "frac_repeat_masked": n_soft / max(1, n_written),
        "base_composition": {ID_TO_BASE.get(i, f"id{i}"): int(c) for i, c in enumerate(base_hist)},
        "gc_content": float((base_hist[1] + base_hist[2]) / max(1, base_hist[:4].sum())),
        "contigs": contigs,
        "dtype": "uint8",
        "min_contig": min_contig,
        "drop_ambiguous_runs": drop_ambiguous_runs,
    }
    with open(str(out_prefix) + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta

def _drop_long_runs(is_bad: np.ndarray, min_run: int) -> np.ndarray:
    """Boolean keep-mask removing runs of True in `is_bad` of length >= min_run"""
    if not is_bad.any():
        return np.ones(is_bad.size, dtype=bool)
    d = np.diff(np.concatenate(([0], is_bad.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends   = np.flatnonzero(d == -1)
    keep   = np.ones(is_bad.size, dtype=bool)
    long_runs = (ends - starts) >= min_run
    for s, e in zip(starts[long_runs], ends[long_runs]):
        keep[s:e] = False
    return keep

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="prefix, no extension")
    p.add_argument("--include-chrom", nargs="*", default=None)
    p.add_argument("--exclude-chrom", nargs="*", default=None)
    p.add_argument("--min-contig", type=int, default=10_000)
    p.add_argument("--drop-ambiguous-runs", type=int, default=100)
    a = p.parse_args()

    meta = prepare_out(a.fasta, 
                        a.out, include=set(a.include_chrom) if a.include_chrom else None,
                        exclude=set(a.exclude_chrom) if a.exclude_chrom else None,
                        min_contig=a.min_contig,
                        drop_ambiguous_runs=a.drop_ambiguous_runs)

    print(json.dumps({k: v for k, v in meta.items() if k != "contigs"}, indent=2))

if __name__ == "__main__":
    main()