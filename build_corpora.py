#!/usr/bin/env python
"""
Build phylogenetic-redundancy training corpora from a multiz MAF

Resulting .bin is: 
        [human core] SEP [species_1 core] SEP [species_2 core] SEP ...
with SEP = --separator-id (default 5: a reserved vocab slot, identity under reverse-complement)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare_data import (
    AMBIG_ID, ID_TO_BASE, PAD_VOCAB, _LUT, _MASK_LUT, encode,
)

_DASH = ord("-")
_VALID = np.zeros(256, dtype=bool)
for _c in b"ACGTacgt":
    _VALID[_c] = True


def iter_maf_blocks(path: Path):
    """Yield {species: aligned_text} per alignment block (species = src before first '.')"""
    opener = gzip.open if str(path).endswith(".gz") else open
    block: dict[str, str] = {}
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("s "):
                parts = line.split()
                if len(parts) < 7:
                    continue
                sp = parts[1].split(".", 1)[0]
                if sp not in block: # first row per species per block
                    block[sp] = parts[6]
            elif line.startswith("a"):
                if block:
                    yield block
                    block = {}
    if block:
        yield block

def _iter_all_blocks(maf_paths):
    for mp in maf_paths:
        yield from iter_maf_blocks(mp)

def build_core(maf_paths, human: str, min_species: int, core_species, want,
               log_every: int = 200_000):
    """
    One pass over all MAF files
    Returns ({species: (ids, mask)}, n_core_cols, n_human_valid_cols)
    """
    if isinstance(maf_paths, (str, Path)):
        maf_paths = [maf_paths]
    core_species = set(core_species) - {human}  # coverage counts NON-human only
    want = set(want) | {human}
    needed = core_species | want | {human}
    buf_ids = {s: bytearray() for s in want}
    buf_msk = {s: bytearray() for s in want}
    n_core = n_human_valid = n_blocks = 0
    t0 = time.time()
    print(f"parsing {len(maf_paths)} MAF file(s) for human={human} "
          f"min_species={min_species}...", flush=True)
    for block in _iter_all_blocks(maf_paths):
        htext = block.get(human)
        if htext is None:
            continue
        hbytes = htext.encode("ascii", "ignore")
        L = len(hbytes)
        # stack all equal-length rows (human first) into one 2D uint8 array
        names = [human]
        rows = [hbytes]
        for n, t in block.items():
            if n == human or n not in needed:
                continue
            b = t.encode("ascii", "ignore")
            if len(b) == L:
                names.append(n)
                rows.append(b)
        mat = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), L)
        row_of = {n: i for i, n in enumerate(names)}

        hvalid = _VALID[mat[0]]                 # human is row 0
        n_human_valid += int(hvalid.sum())
        core_rows = [row_of[s] for s in core_species if s in row_of]
        present = ((mat[core_rows] != _DASH).sum(axis=0)
                   if core_rows else np.zeros(L, dtype=np.int64))
        core = hvalid & (present >= min_species)
        nc = int(core.sum())
        n_core += nc

        n_blocks += 1
        if log_every and n_blocks % log_every == 0:
            print(f"  ...{n_blocks:,} blocks | {n_human_valid/1e6:.1f}M human cols "
                  f"| {n_core/1e6:.2f}M core | {time.time() - t0:.0f}s", flush=True)

        if nc == 0:
            continue
        core_idx = np.flatnonzero(core)
        for sp in want:
            i = row_of.get(sp)
            if i is None:
                continue
            cols = mat[i, core_idx]
            cols = cols[cols != _DASH]          # de-gap THIS species
            if cols.size == 0:
                continue
            buf_ids[sp].extend(_LUT[cols])
            buf_msk[sp].extend(_MASK_LUT[cols])

    print(f"parse done: {n_blocks:,} blocks, {time.time() - t0:.0f}s", flush=True)
    streams = {}
    for sp in want:
        streams[sp] = (np.frombuffer(buf_ids[sp], np.uint8),
                       np.frombuffer(buf_msk[sp], np.uint8))
    return streams, n_core, n_human_valid


def write_condition(out_prefix, human, species, streams, sep_id, maf_path,
                    min_species, f_core, core_species=None):
    hid, hmk = streams[human]
    ids_parts, msk_parts = [hid], [hmk]
    segments = [{"species": human, "n_tokens": int(hid.size)}]
    sep_i = np.array([sep_id], np.uint8)
    sep_m = np.array([0], np.uint8)
    for sp in species:
        sid, smk = streams[sp]
        ids_parts += [sep_i, sid]
        msk_parts += [sep_m, smk]
        segments.append({"species": sp, "n_tokens": int(sid.size)})
    ids = np.concatenate(ids_parts)
    msk = np.concatenate(msk_parts)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(str(out_prefix) + ".bin")
    msk.tofile(str(out_prefix) + ".mask.bin")

    hist = np.bincount(ids, minlength=PAD_VOCAB)
    meta = {
        "source_maf": str(maf_path),
        "condition": out_prefix.name,
        "human": human,
        "species": list(species),
        "k": len(species),
        "separator_id": sep_id,
        "min_species": min_species,
        "core_species": list(core_species) if core_species is not None else None,
        "f_core": f_core,
        "vocab_size": PAD_VOCAB,
        "n_tokens": int(ids.size),
        "n_separators": int((ids == sep_id).sum()),
        "frac_ambiguous": float((ids == AMBIG_ID).sum()) / max(1, ids.size),
        "frac_soft_masked": float(msk.sum()) / max(1, ids.size),
        "base_composition": {
            ID_TO_BASE.get(i, f"id{i}"): int(c) for i, c in enumerate(hist)
        },
        "gc_content": float((hist[1] + hist[2]) / max(1, hist[:4].sum())),
        "segments": segments,
        "dtype": "uint8",
    }
    with open(str(out_prefix) + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maf", type=Path, nargs="+", required=True,
                   help="one or more MAF files (.maf/.maf.gz), e.g. chr1 chr2 chr3")
    p.add_argument("--manifest", type=Path, required=True, help="conditions JSON")
    p.add_argument("--out-dir", type=Path, default=Path("corpora"))
    p.add_argument("--separator-id", type=int, default=5)
    args = p.parse_args()

    man = json.loads(args.manifest.read_text())
    human = man.get("human", "hg38")
    min_species = int(man["min_species"])
    conditions = man["conditions"]
    if "core_species" not in man:
        sys.exit("ABORT: manifest needs 'core_species' -- the FIXED ladder the core "
                 "is defined over (core = human-valid column with >= min_species of "
                 "these present). Omitting it and counting all MAF species inflates "
                 "f_core ~3x (0.64 vs the measured 0.23)!!")
    core_species = man["core_species"]

    want = {sp for spp in conditions.values() for sp in spp}
    streams, n_core, n_hvalid = build_core(args.maf, human, min_species,
                                           core_species, want)
    maf_src = ",".join(str(m) for m in args.maf)
    f_core = n_core / max(1, n_hvalid)
    print(f"f_core = {f_core:.4f}  ({n_core:,} core / {n_hvalid:,} human-valid columns, "
          f">= {min_species} of {len(core_species)} core species)")

    for name, species in conditions.items():
        missing = [s for s in species if streams[s][0].size == 0]
        meta = write_condition(args.out_dir / name, human, species, streams,
                               args.separator_id, maf_src, min_species, f_core,
                               core_species)
        warn = f"  !! absent in MAF: {missing}" if missing else ""
        print(f"  {name:12s} k={meta['k']} tokens={meta['n_tokens']:,} "
              f"GC={meta['gc_content']:.3f} seps={meta['n_separators']}{warn}")

if __name__ == "__main__":
    main()
