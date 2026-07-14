"""
One-off utility: trim a bulk test-case corpus down to an exact case count.

tools/generate_test_cases_bulk.py's --target controls how many base
integers get swept, not the final case count directly (each integer
contributes a written-number phrase plus several gated format variants,
so the total is a derived, not a set, number). To land on an exact,
citable corpus size (e.g. "30,000,000 test cases"), generate MORE than
you need with --target, then trim down to the exact count with this
script.

Usage:
    python tools/trim_corpus.py --corpus data/test_cases_bulk --target 30000000
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from test_case_store import iter_shards, load_manifest, write_manifest

SHARD_SIZE = 100_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/test_cases_bulk",
                         help="Corpus directory to trim (must already be fully generated)")
    parser.add_argument("--target", type=int, required=True,
                         help="Exact number of cases to keep")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    if not (corpus_dir / "manifest.json").exists():
        print(f"No manifest.json in {corpus_dir} — generation didn't finish. "
              "Re-run tools/generate_test_cases_bulk.py first.")
        sys.exit(1)

    old_manifest = load_manifest(corpus_dir)
    available = old_manifest.get("total_cases", "unknown")
    print(f"Corpus has {available:,} cases" if isinstance(available, int) else "Corpus has an unknown case count")
    if isinstance(available, int) and available < args.target:
        print(f"Corpus only has {available:,} cases, fewer than the requested {args.target:,}. "
              "Re-run the generator with a larger --target first.")
        sys.exit(1)

    tmp_dir = corpus_dir.parent / (corpus_dir.name + "_trimmed_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    by_category: dict[str, int] = {}
    buffer: list[dict] = []
    shard_idx = 0
    total = 0

    def flush():
        nonlocal buffer, shard_idx
        if not buffer:
            return
        path = tmp_dir / f"shard_final_{shard_idx:05d}.jsonl"
        with path.open("w") as f:
            for case in buffer:
                f.write(json.dumps(case) + "\n")
        shard_idx += 1
        buffer = []

    for case in iter_shards(corpus_dir):
        if total >= args.target:
            break
        buffer.append(case)
        total += 1
        by_category[case.get("category", "unknown")] = by_category.get(case.get("category", "unknown"), 0) + 1
        if len(buffer) >= SHARD_SIZE:
            flush()
    flush()

    if total < args.target:
        shutil.rmtree(tmp_dir)
        print(f"Only found {total:,} cases while streaming — corpus is smaller than its manifest claims.")
        sys.exit(1)

    shutil.rmtree(corpus_dir)
    tmp_dir.rename(corpus_dir)

    manifest = {
        "total_cases": total,
        "workers": old_manifest.get("workers"),
        "target_integers": old_manifest.get("target_integers"),
        "generation_seconds": old_manifest.get("generation_seconds"),
        "cases_per_second": old_manifest.get("cases_per_second"),
        "by_category": by_category,
        "note": (
            f"Trimmed to exactly {total:,} cases from a larger generation run "
            f"(originally {available:,}) for a fixed, citable corpus size. "
            "Trimming takes the first N cases in shard order, so the category "
            "mix is whatever that prefix happened to contain, not necessarily "
            "identical to the untrimmed corpus's proportions."
        ),
    }
    write_manifest(corpus_dir, manifest)

    print(f"Trimmed corpus to exactly {total:,} cases across {shard_idx} shards.")
    print("By category:")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,}")


if __name__ == "__main__":
    main()
