#!/usr/bin/env python3
"""Score the saved v1/v2/v3 solutions on the frozen stratified eval set.

results/run_*.json (gitignored) holds the greedy solution each qwen2-codejudge
version produced on the hand-written grader. Re-scoring that same code on the
stratified set gives baselines that are directly comparable with the v4
runs, without needing Ollama or the model weights. Identical solution texts
are scored once.

    python tools/score_saved_solutions.py
Writes experiments/baselines/<model_tag>.json (committed; small).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import evaluation as ev  # noqa: E402
from code_cleaning import clean_llm_code  # noqa: E402

EVAL_SET = ROOT / "data" / "eval_sets" / "ambigous_spec_strat2000_seed42.jsonl"
OUT_DIR = ROOT / "experiments" / "baselines"


def latest_saved_solutions(results_dir: Path) -> dict:
    latest: dict = {}
    for f in sorted(results_dir.glob("run_*.json")):
        for t in json.loads(f.read_text()):
            if t.get("task") != "ambigous_spec":
                continue
            for tag, v in t.get("models", {}).items():
                if isinstance(v, dict) and v.get("solution") and "total" in v.get("scores", {}) \
                        and tag.startswith("qwen2-codejudge"):
                    latest[tag] = (f, v)
    return latest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--eval-set", type=Path, default=EVAL_SET)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    args = ap.parse_args()

    cases, meta = ev.load_eval_set(args.eval_set)
    prov = ev.provenance()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    for tag, (run_file, v) in sorted(latest_saved_solutions(args.results_dir).items()):
        solution = clean_llm_code(v["solution"])
        key = hashlib.sha256(solution.encode()).hexdigest()
        if key not in cache:
            print(f"scoring {tag} ({len(cases)} cases)...", flush=True)
            cache[key] = ev.summarize(ev.score_solution(solution, cases), meta["corpus_counts"],
                                      seed=args.bootstrap_seed)
        else:
            print(f"{tag}: identical solution text to an already-scored one, reusing", flush=True)
        scores = v["scores"]
        summary = {
            "schema": 1, "kind": "baseline", "arm": "baseline", "run_id": f"baseline_{tag}",
            "n_examples": None, "seed": None, "model_tag": tag,
            "source_run_file": str(run_file.relative_to(ROOT)),
            "source_run_sha256": hashlib.sha256(run_file.read_bytes()).hexdigest(),
            "solution_sha256": key, "eval_set_sha256": meta["cases_sha256"], **prov,
            "metrics": {"eval": cache[key],
                        "hand_grader": {"passed": scores["passed"], "total": scores["total"],
                                        "failed_tests": [t["name"] for t in scores.get("tests", []) if not t["passed"]]}},
            "diagnostics": {"generation": ev.diagnose_generation(v["solution"], solution)},
            "solution": solution,
        }
        out = OUT_DIR / f"{tag.replace(':', '_')}.json"
        out.write_text(json.dumps(summary, indent=2) + "\n")
        w = summary["metrics"]["eval"]["weighted"]
        print(f"  {tag}: weighted {w['accuracy'] * 100:.1f}% CI {w['ci95']}  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
