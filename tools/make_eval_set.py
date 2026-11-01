#!/usr/bin/env python3
"""Freeze a stratified evaluation sample from the sharded corpus.

The uniform seed-42 sample used by tools/evaluate_at_scale.py gives the tiny
categories (invalid: 9 cases, zero: 6 in a 30M corpus) essentially no
coverage, and per-category confidence intervals are useless without cases.
This draws a *stratified* sample instead:

  1. every category gets min(its size, --floor) cases;
  2. the rest of the budget is split across categories in proportion to
     corpus size (largest-remainder), capped at each category's capacity.

Within a category the draw is uniform without replacement (one seeded RNG
per category), in a single streaming pass: memory is O(total). Because the
sample over-represents small categories on purpose, overall accuracy must be
reweighted by corpus share; evaluation.summarize() does that.

Output (both committed, both small):
    data/eval_sets/<name>.jsonl       one case per line, with a stable id
    data/eval_sets/<name>.meta.json   seed, allocation, corpus counts, sha256

Usage:
    python tools/make_eval_set.py --name ambigous_spec_strat2000_seed42 --size 2000 --floor 100 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from test_case_store import load_manifest  # noqa: E402

DEFAULT_CORPUS = ROOT / "data" / "test_cases_bulk"
OUT_DIR = ROOT / "data" / "eval_sets"


def allocate(counts: dict, size: int, floor: int) -> dict:
    """Split `size` cases across categories: floor first, then proportional."""
    if size > sum(counts.values()):
        raise ValueError("size exceeds corpus")
    alloc = {c: min(n, floor) for c, n in counts.items()}
    if sum(alloc.values()) > size:
        raise ValueError(f"floor {floor} x {len(counts)} categories exceeds size {size}")
    remaining = size - sum(alloc.values())
    while remaining > 0:
        room = {c: counts[c] - alloc[c] for c in counts if counts[c] - alloc[c] > 0}
        if not room:
            break
        total_room_weight = sum(counts[c] for c in room)
        quota = {c: remaining * counts[c] / total_room_weight for c in room}
        give = {c: min(int(quota[c]), room[c]) for c in room}
        leftover = remaining - sum(give.values())
        by_fraction = sorted(room, key=lambda c: (quota[c] - int(quota[c]), counts[c]), reverse=True)
        for c in by_fraction:
            if leftover == 0:
                break
            if give[c] < room[c]:
                give[c] += 1
                leftover -= 1
        for c, g in give.items():
            alloc[c] += g
        remaining = size - sum(alloc.values())
    assert sum(alloc.values()) == size
    return dict(sorted(alloc.items()))


def draw(corpus_dir: Path, alloc: dict, counts: dict, seed: int) -> list[dict]:
    """Uniform sample without replacement within each category, in one pass.

    The per-category sizes are known exactly (manifest, verified against the
    shards by tools/build_ledger.py), so the k positions to keep in each
    category are chosen up front with random.sample over range(size); a
    streaming pass then keeps the lines at those positions and only parses
    JSON for the ones it keeps. Same distribution as reservoir sampling,
    far fewer operations per line. The pass counts every category and fails
    loudly if the shards disagree with the manifest.
    """
    wanted = {c: set(random.Random(f"{seed}:{c}").sample(range(counts[c]), k)) for c, k in alloc.items()}
    seen = dict.fromkeys(counts, 0)
    kept: dict = {c: [] for c in alloc}
    mark = '"category": "'
    for shard in sorted(Path(corpus_dir).glob("shard_*.jsonl")):
        with shard.open() as f:
            for line in f:
                if not line.strip():
                    continue
                i = line.rfind(mark)
                c = line[i + len(mark): line.find('"', i + len(mark))]
                idx = seen[c]
                seen[c] = idx + 1
                if idx in wanted.get(c, ()):
                    kept[c].append(json.loads(line))
    if seen != counts:
        raise ValueError(f"shards disagree with manifest: {seen} vs {counts}")
    out = [case for c in sorted(kept) for case in kept[c]]
    return [{"id": i, **case} for i, case in enumerate(out)]


def build(corpus_dir: Path, size: int, floor: int, seed: int) -> tuple[list[dict], dict]:
    manifest = load_manifest(corpus_dir)
    counts = manifest["by_category"]
    alloc = allocate(counts, size, floor)
    cases = draw(corpus_dir, alloc, counts, seed)
    body = "".join(json.dumps(c, sort_keys=True) + "\n" for c in cases)
    meta = {
        "method": "stratified: floor per category, remainder proportional to corpus size; "
                  "uniform draw without replacement within category, one RNG per category",
        "size": size, "floor": floor, "seed": seed,
        "allocation": alloc, "corpus_counts": counts, "corpus_total": manifest["total_cases"],
        "cases_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    return cases, meta


def write(cases: list[dict], meta: dict, name: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    path.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cases))
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--name", default="ambigous_spec_strat2000_seed42")
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--floor", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cases, meta = build(args.corpus, args.size, args.floor, args.seed)
    path = write(cases, meta, args.name)
    print(f"wrote {len(cases)} cases to {path.relative_to(ROOT)}")
    for c, n in meta["allocation"].items():
        print(f"  {c:16s} {n:5d}  (corpus {meta['corpus_counts'][c]:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
