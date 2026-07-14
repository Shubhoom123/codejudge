"""
Generates a large, sharded test-case corpus for parse_amount using
multiprocessing, streamed straight to disk instead of held in memory.

The scaling lever is combinatorial, not arbitrary: every positive integer
in a chosen range has exactly one canonical "written-number" phrase (e.g.
482613 -> "four hundred eighty two thousand six hundred thirteen"), and
every phrase is validated by round-tripping it through the reference
parse_amount() before being kept — if the reference can't parse its own
generated phrase back to the same number, the case is dropped, not kept.
Layered on top are the same dollar-sign / comma / whitespace / negative /
decimal format variants from tools/generate_test_cases.py, applied across
a much larger number pool than that script uses.

This does NOT invent new formats beyond what tasks/ambigous_spec/hint.md
already declares as valid input — it's a wider combinatorial sweep of the
same declared contract, which is the whole point (catch gaps the 24
hand-written grader cases and the original 56k-case generator miss).

Output: data/test_cases_bulk/shard_*.jsonl + manifest.json
        (sharded JSONL, NOT one JSON array — see test_case_store.py)

Usage:
    python tools/generate_test_cases_bulk.py --target 2000000 --workers 4
    python tools/generate_test_cases_bulk.py --target 50000 --workers 4   # quick run
"""
import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from test_case_store import ShardWriter, write_manifest

# ─── Reference solution — same contract as tasks/ambigous_spec/grader.py ───

def parse_amount(text: str) -> float:
    text = text.strip().lower()
    if not text:
        raise ValueError("Empty input")

    text = text.replace("$", "").replace(",", "")
    if "dollars" in text:
        text = text.replace("dollars", "").strip()
    if "dollar" in text:
        text = text.replace("dollar", "").strip()
    if "usd" in text:
        text = text.replace("usd", "").strip()

    text = text.strip()

    try:
        return float(text)
    except ValueError:
        pass

    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
        "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90,
    }

    words = text.replace(" and ", " ").split()
    total = 0
    current = 0
    for word in words:
        if word in number_words:
            current += number_words[word]
        elif word == "hundred":
            current = current * 100 if current else 100
        elif word == "thousand":
            total += (current or 1) * 1000
            current = 0
        elif word == "million":
            total += (current or 1) * 1000000
            current = 0
        else:
            raise ValueError(f"Cannot parse: {word!r}")

    result = total + current
    if result == 0 and text not in ("zero", "0", "$0", "0.0"):
        raise ValueError(f"Cannot parse: {text!r}")
    return float(result)


# ─── Integer -> written-number phrase (inverse of the parser above) ────────

ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _three_digit_to_words(n: int) -> str:
    parts = []
    hundreds, rem = divmod(n, 100)
    if hundreds:
        parts.append(f"{ONES[hundreds]} hundred")
    if rem:
        if rem < 20:
            parts.append(ONES[rem])
        else:
            tens_digit, ones_digit = divmod(rem, 10)
            parts.append(f"{TENS[tens_digit]} {ONES[ones_digit]}" if ones_digit else TENS[tens_digit])
    return " ".join(parts)


def number_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    millions, rem = divmod(n, 1_000_000)
    thousands, rem = divmod(rem, 1000)
    parts = []
    if millions:
        parts.append(f"{_three_digit_to_words(millions)} million")
    if thousands:
        parts.append(f"{_three_digit_to_words(thousands)} thousand")
    if rem:
        parts.append(_three_digit_to_words(rem))
    return " ".join(parts)


# ─── Per-worker chunk generation ────────────────────────────────────────────

WHITESPACE_VARIANTS = [("  ", "  "), ("\t", "\t"), ("   ", ""), ("", "   "), ("\n", "")]


def generate_chunk(task: tuple) -> dict:
    worker_id, start, end, out_dir, shard_size = task
    writer = ShardWriter(Path(out_dir), shard_prefix=f"w{worker_id}", shard_size=shard_size)
    counts: dict[str, int] = {}
    rng = random.Random(worker_id)  # deterministic per worker

    def emit(text: str, expected: float, category: str) -> None:
        writer.write({"input": text, "expected": expected, "category": category})
        counts[category] = counts.get(category, 0) + 1

    for n in range(start, end):
        # Written numbers: the reference must round-trip its own phrase
        # back to the same value, or the case is dropped as untrustworthy.
        words = number_to_words(n)
        try:
            if parse_amount(words) == float(n):
                emit(words, float(n), "written_number")
        except Exception:
            pass

        # Modulo gates keep the format-variant categories proportionate
        # instead of letting them dominate the written-number sweep.
        if n % 7 == 0:
            for text, cat in ((f"${n}", "dollar_sign"), (f"{n} dollars", "word_dollars"), (str(n), "plain_number")):
                try:
                    emit(text, parse_amount(text), cat)
                except Exception:
                    pass

        if n % 11 == 0 and n >= 1000:
            text = f"${n:,}"
            try:
                emit(text, parse_amount(text), "comma_thousands")
            except Exception:
                pass

        if n % 13 == 0 and n > 0:
            for prefix, suffix in WHITESPACE_VARIANTS:
                text = f"{prefix}${n}{suffix}"
                try:
                    emit(text, parse_amount(text), "whitespace")
                except Exception:
                    pass
            for fmt in (f"-${n}", f"-{n}", f"- {n}"):
                try:
                    emit(fmt, parse_amount(fmt), "negative")
                except Exception:
                    pass

        if n % 17 == 0:
            cents = rng.randint(0, 99)
            text = f"${n}.{cents:02d}"
            try:
                emit(text, parse_amount(text), "decimal")
            except Exception:
                pass

    writer.close()
    return {"worker_id": worker_id, "written": writer.total_written, "by_category": counts}


# Hand-picked edge-case candidates that aren't naturally combinatorial.
# These are *candidates*, not guaranteed cases — each is only kept if it
# actually behaves as intended under the reference solution (see
# build_fixed_edge_cases below). "$$100" looked like it should raise
# ValueError but doesn't (the reference just strips both "$"s), which is
# exactly the kind of assumption this validation step catches.
INVALID_CANDIDATES = [
    "banana", "xyz", "hello world", "$$100", "€100", "£50", "NaN", "inf",
    "1e10", "   ", "hundred hundred", "million billion", "one.two", "12/50",
]
ZERO_CANDIDATES = ["0", "$0", "zero", "0.00", "$0.00", "0 dollars"]


def build_fixed_edge_cases() -> list[tuple[str, object, str]]:
    """Validate hand-picked candidates against the reference solution and
    keep only the ones that actually do what their category claims."""
    cases = []
    for text in INVALID_CANDIDATES:
        try:
            parse_amount(text)
            # Didn't raise — this candidate doesn't belong in "invalid".
        except ValueError:
            cases.append((text, "raises_ValueError", "invalid"))
    for text in ZERO_CANDIDATES:
        try:
            expected = parse_amount(text)
            if expected == 0.0:
                cases.append((text, expected, "zero"))
        except ValueError:
            pass
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2_000_000,
                        help="Number of base integers to sweep across the worker pool. "
                             "Actual case count is higher — each integer contributes a "
                             "written-number phrase plus several gated format variants.")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--out-dir", default="data/test_cases_bulk")
    parser.add_argument("--shard-size", type=int, default=100_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{out_dir} already has files in it — remove it first for a clean regeneration.")

    print(f"Sweeping {args.target:,} integers across {args.workers} worker processes...")
    print(f"Output: {out_dir}/ (sharded JSONL, {args.shard_size:,} lines/shard)\n")

    chunk_size = args.target // args.workers
    tasks = []
    for w in range(args.workers):
        start = w * chunk_size
        end = args.target if w == args.workers - 1 else (w + 1) * chunk_size
        tasks.append((w, start, end, str(out_dir), args.shard_size))

    start_time = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(generate_chunk, tasks)
    elapsed = time.time() - start_time

    fixed_edge_cases = build_fixed_edge_cases()
    edge_writer = ShardWriter(out_dir, shard_prefix="edge", shard_size=args.shard_size)
    for text, expected, category in fixed_edge_cases:
        edge_writer.write({"input": text, "expected": expected, "category": category})
    edge_writer.close()

    total = sum(r["written"] for r in results) + edge_writer.total_written
    by_category: dict[str, int] = {}
    for r in results:
        for cat, count in r["by_category"].items():
            by_category[cat] = by_category.get(cat, 0) + count
    for _, _, cat in fixed_edge_cases:
        by_category[cat] = by_category.get(cat, 0) + 1

    manifest = {
        "total_cases": total,
        "workers": args.workers,
        "target_integers": args.target,
        "generation_seconds": round(elapsed, 2),
        "cases_per_second": round(total / elapsed) if elapsed > 0 else None,
        "by_category": by_category,
        "note": (
            "Deduplication is per-worker only, not global: workers operate "
            "on disjoint integer ranges, so exact duplicates across workers "
            "are possible but rare and not corrected for. Fine for a "
            "sampled evaluation corpus; not a guarantee of a perfectly "
            "unique set."
        ),
    }
    write_manifest(out_dir, manifest)

    print(f"Done in {elapsed:.1f}s ({manifest['cases_per_second']:,} cases/sec)")
    print(f"Total cases: {total:,}")
    print("By category:")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,}")
    print(f"\nManifest: {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
