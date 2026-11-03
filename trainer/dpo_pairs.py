#!/usr/bin/env python3
"""Build DPO preference pairs from the teacher attempts, and check they are not degenerate.

    python trainer/dpo_pairs.py            # writes data/dpo/<task>_pairs_seed0.jsonl + .meta.json

Every teacher attempt answers the same prompt, so a pair is (chosen = an
attempt that passed all 24 grader tests, rejected = an attempt for the SAME
prompt that scored lower). The margin is chosen_score - rejected_score on the
24-test grader (one failed test = 0.0417, shown rounded as 0.04).

Choices worth knowing
  * Attempts whose grader crashed ("No output from grader", score 0) are NOT
    used as rejected: they say nothing about solution quality.
  * Each chosen attempt is paired with `per_chosen` rejected ones, drawn
    without replacement from three margin tiers (near / mid / far) so the
    pairs are not all easy and not all one-test-away. All draws are seeded.
  * Pair text lives in a .jsonl that is NOT committed (it is re-derivable);
    the committed .meta.json holds the sha256 of that file plus the
    diagnostics, so a run can prove which pairs it trained on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cj_config  # noqa: E402

# (name, max margin). A margin is 1 - rejected_score; scores are 2-decimal fractions of 24.
TIERS = (("near", 0.08), ("mid", 0.17), ("far", 1.01))
DEFAULT_PER_CHOSEN = {"near": 1, "mid": 2, "far": 1}
WRITTEN_TESTS = {"written_number_single", "compound_written_number", "written_thousand", "large_written_million"}


def tier_of(margin: float) -> str:
    for name, hi in TIERS:
        if margin <= hi + 1e-9:
            return name
    raise ValueError(margin)


def load_attempts(training_data: Path, task: str) -> tuple[list[dict], dict]:
    """Graded attempts with their failed tests, plus counts of what was left out."""
    raw = json.loads(Path(training_data).read_text())
    problem = cj_config.load_task_prompt(task).strip()
    out, crashed, other_task = [], 0, 0
    for idx, ex in enumerate(raw):
        if ex["task"] != task:
            other_task += 1
            continue
        if ex["prompt"].strip() != problem:
            raise ValueError(f"attempt {idx} was recorded against a different prompt than problem.md")
        tests = ex["scores"].get("tests")
        if tests is None:  # grader error / crash
            crashed += 1
            continue
        out.append({"id": idx, "solution": ex["solution"].strip(), "score": float(ex["score"]),
                    "failed": sorted(t["name"] for t in tests if not t["passed"])})
    return out, {"n_raw": len(raw), "n_grader_crash_excluded": crashed, "n_other_task": other_task}


def build_pairs(attempts: list[dict], *, seed: int, per_chosen: dict | None = None,
                min_score: float = 1.0) -> list[dict]:
    per_chosen = per_chosen or DEFAULT_PER_CHOSEN
    chosen = sorted((a for a in attempts if a["score"] >= min_score), key=lambda a: a["id"])
    rejected = [a for a in attempts if a["score"] < min_score]
    pools: dict[str, list[dict]] = {name: [] for name, _ in TIERS}
    for a in sorted(rejected, key=lambda a: a["id"]):
        pools[tier_of(round(min_score - a["score"], 4))].append(a)
    pairs = []
    for c in chosen:
        rng = random.Random(f"{seed}:{c['id']}")
        for name, _ in TIERS:
            pool = [r for r in pools[name] if r["solution"] != c["solution"]]
            for r in rng.sample(pool, min(per_chosen.get(name, 0), len(pool))):
                pairs.append({"chosen_id": c["id"], "rejected_id": r["id"], "chosen_score": c["score"],
                              "rejected_score": r["score"], "margin": round(c["score"] - r["score"], 4),
                              "tier": name, "rejected_failed_tests": r["failed"],
                              "chosen": c["solution"], "rejected": r["solution"]})
    return pairs


def diagnostics(pairs: list[dict], attempts: list[dict], excluded: dict) -> dict:
    """Facts that tell you whether the pairs can teach anything, or are degenerate."""
    if not pairs:
        raise ValueError("no pairs built")
    margins = [p["margin"] for p in pairs]
    rej_use: dict[int, int] = {}
    for p in pairs:
        rej_use[p["rejected_id"]] = rej_use.get(p["rejected_id"], 0) + 1
    chosen_len = [len(p["chosen"]) for p in pairs]
    rej_len = [len(p["rejected"]) for p in pairs]
    fail_hist: dict[str, int] = {}
    for p in pairs:
        for t in p["rejected_failed_tests"]:
            fail_hist[t] = fail_hist.get(t, 0) + 1
    return {
        **excluded,
        "n_attempts_graded": len(attempts),
        "n_pairs": len(pairs),
        "n_unique_chosen": len({p["chosen_id"] for p in pairs}),
        "n_unique_rejected": len({p["rejected_id"] for p in pairs}),
        "max_times_a_rejected_is_reused": max(rej_use.values()),
        "tier_counts": {name: sum(p["tier"] == name for p in pairs) for name, _ in TIERS},
        "margin": {"min": min(margins), "mean": round(st.mean(margins), 4), "max": max(margins),
                   "n_distinct": len(set(margins))},
        "chars": {"chosen_mean": round(st.mean(chosen_len), 1), "rejected_mean": round(st.mean(rej_len), 1),
                  "frac_pairs_rejected_longer": round(sum(r > c for c, r in zip(chosen_len, rej_len)) / len(pairs), 4)},
        "rejected_failed_test_counts": dict(sorted(fail_hist.items(), key=lambda kv: -kv[1])),
        "frac_pairs_rejected_fails_a_written_number_test": round(
            sum(any(t in WRITTEN_TESTS for t in p["rejected_failed_tests"]) for p in pairs) / len(pairs), 4),
        "checks": {
            "all_margins_positive": all(m > 0 for m in margins),
            "no_identical_pairs": all(p["chosen"] != p["rejected"] for p in pairs),
            "no_duplicate_pairs": len({(p["chosen_id"], p["rejected_id"]) for p in pairs}) == len(pairs),
        },
    }


def _rel(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(path)


def pairs_sha256(pairs: list[dict]) -> str:
    return hashlib.sha256("\n".join(json.dumps(p, sort_keys=True) for p in pairs).encode()).hexdigest()


def build(training_data: Path, task: str, *, seed: int, per_chosen: dict | None = None) -> tuple[list[dict], dict]:
    attempts, excluded = load_attempts(training_data, task)
    pairs = build_pairs(attempts, seed=seed, per_chosen=per_chosen)
    meta = {"schema": 1, "task": task, "seed": seed, "per_chosen": per_chosen or DEFAULT_PER_CHOSEN,
            "tiers_max_margin": dict(TIERS), "training_data": _rel(training_data),
            "pairs_sha256": pairs_sha256(pairs), "diagnostics": diagnostics(pairs, attempts, excluded)}
    return pairs, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="ambigous_spec")
    ap.add_argument("--training-data", type=Path,
                    default=ROOT / "data" / "training_runs" / "ambigous_spec" / "training_data.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "dpo")
    args = ap.parse_args(argv)
    pairs, meta = build(args.training_data, args.task, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_pairs_seed{args.seed}"
    (args.out_dir / f"{stem}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    (args.out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    d = meta["diagnostics"]
    print(f"{d['n_pairs']} pairs from {d['n_unique_chosen']} chosen / {d['n_unique_rejected']} rejected attempts "
          f"(sha {meta['pairs_sha256'][:12]})")
    print(f"  tiers {d['tier_counts']}  margin {d['margin']}")
    print(f"  chars chosen {d['chars']['chosen_mean']} vs rejected {d['chars']['rejected_mean']} "
          f"(rejected longer in {d['chars']['frac_pairs_rejected_longer']:.0%} of pairs)")
    print(f"  rejected fails a written-number test in {d['frac_pairs_rejected_fails_a_written_number_test']:.0%} of pairs")
    print(f"  checks {d['checks']}")
    return 0 if all(d["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
