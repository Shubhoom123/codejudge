"""Storage layer for large, programmatically-generated test-case corpora.

A single JSON array doesn't hold up once you're generating millions of
cases — it has to be loaded whole into memory to read even one record,
and a multi-GB file doesn't belong in git. Cases are written as
newline-delimited JSON (JSONL) shards of SHARD_SIZE lines each, in their
own directory, alongside a manifest.json describing what's in there.

Reading supports two modes:
  - iter_shards(): stream every case, one at a time (for anything that
    needs to look at all of them, e.g. building the corpus itself)
  - reservoir_sample(): pick k cases uniformly at random from a corpus of
    arbitrary size *without* loading it all into memory — this is what
    evaluate_at_scale.py uses, since a model only needs to be checked
    against a sampled subset, not the whole multi-million-case pool.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SHARD_SIZE = 100_000


class ShardWriter:
    """Streams cases to disk in fixed-size JSONL shards.

    Safe to construct one per worker process — each writer owns its own
    shard filenames (via `shard_prefix`) so parallel workers never
    collide on the same file.
    """

    def __init__(self, out_dir: Path, shard_prefix: str, shard_size: int = SHARD_SIZE):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_prefix = shard_prefix
        self.shard_size = shard_size
        self._shard_index = 0
        self._buffer = []
        self.total_written = 0

    def write(self, case: dict) -> None:
        self._buffer.append(case)
        self.total_written += 1
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        shard_path = self.out_dir / f"shard_{self.shard_prefix}_{self._shard_index:05d}.jsonl"
        with shard_path.open("w") as f:
            for case in self._buffer:
                f.write(json.dumps(case) + "\n")
        self._shard_index += 1
        self._buffer = []

    def close(self) -> None:
        self._flush()


def iter_shards(corpus_dir: Path):
    """Stream every case in the corpus, one shard file at a time."""
    corpus_dir = Path(corpus_dir)
    for shard_path in sorted(corpus_dir.glob("shard_*.jsonl")):
        with shard_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def reservoir_sample(corpus_dir: Path, k: int, seed: int = 42) -> list[dict]:
    """Sample k cases uniformly from a sharded corpus of any size.

    Uses reservoir sampling (Algorithm R) so peak memory is O(k), not
    O(corpus size) — needed once the corpus is millions of cases.
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []
    for i, case in enumerate(iter_shards(corpus_dir)):
        if i < k:
            reservoir.append(case)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = case
    return reservoir


def write_manifest(corpus_dir: Path, manifest: dict) -> None:
    (Path(corpus_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_manifest(corpus_dir: Path) -> dict:
    manifest_path = Path(corpus_dir) / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def count_cases(corpus_dir: Path) -> int:
    manifest = load_manifest(corpus_dir)
    if "total_cases" in manifest:
        return manifest["total_cases"]
    return sum(1 for _ in iter_shards(corpus_dir))
