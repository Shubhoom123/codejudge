"""Scoring, bootstrap confidence intervals and generation diagnostics.

Used by trainer/sft_train.py (on Colab, right after training), by
tools/score_saved_solutions.py (baselines from saved run files) and by
tools/build_ledger.py (to turn run summaries into ledger entries).

Execution of generated code goes through tools/evaluate_at_scale.py::
run_test_case, which goes through sandbox.run_isolated(): this module adds
no second execution path.

What the confidence intervals mean (read before quoting one). A run
produces ONE greedy-decoded solution, so every case in a category passes or
fails together, for the most part. The bootstrap here resamples *test
cases*, so it measures test-set sampling noise only. It says nothing about
training noise: that is the spread across seeds, which aggregate_runs
reports separately. A tight bootstrap CI around 0.0% does not mean a model
"reliably" fails; it means this one solution fails this category's cases.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_ENTRY_POINT = "parse_amount"
WRITTEN_LOGIC_RE = re.compile(r"\b(thousand|million|hundred)\b", re.IGNORECASE)


# ---------------------------------------------------------------- eval set
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def meta_path(cases_path: Path) -> Path:
    return Path(cases_path).with_suffix(".meta.json")


def load_eval_set(cases_path: Path) -> tuple[list[dict], dict]:
    """Load a frozen eval set and verify it against its metadata hash."""
    cases_path = Path(cases_path)
    raw = cases_path.read_bytes()
    meta = json.loads(meta_path(cases_path).read_text())
    if sha256_bytes(raw) != meta["cases_sha256"]:
        raise ValueError(f"{cases_path} does not match the sha256 in its metadata; the eval set was modified")
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return cases, meta


# -------------------------------------------------------------- provenance
def provenance() -> dict:
    """Code commit and environment to stamp onto every run summary."""
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    from datetime import datetime, timezone
    return {"code_commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain")),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "python": platform.python_version(), "platform": platform.platform()}


# ----------------------------------------------------------------- scoring
def score_solution(solution: str, cases: list[dict], entry_point: str = DEFAULT_ENTRY_POINT,
                   workers: int = 4) -> list[dict]:
    from evaluate_at_scale import run_test_case  # sandboxed, one subprocess per case

    def one(case: dict) -> dict:
        passed, detail = run_test_case(solution, entry_point, case["input"], case["expected"])
        return {"id": case.get("id"), "category": case["category"], "passed": bool(passed), "detail": detail[:120]}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, cases))


# -------------------------------------------------------------- statistics
def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def summarize(results: list[dict], corpus_counts: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Per-category accuracy with bootstrap 95% CIs, plus corpus-weighted overall.

    `weighted` reweights each category by its share of the full corpus, so a
    stratified sample (which over-represents small categories on purpose)
    still estimates accuracy on the corpus. Weights are renormalised over the
    categories present in `results`. The overall CI comes from a stratified
    bootstrap: resample within each category, recombine with the weights.
    """
    by: dict = defaultdict(lambda: [0, 0])
    for r in results:
        by[r["category"]][0] += bool(r["passed"])
        by[r["category"]][1] += 1
    cats = sorted(by)
    total_w = sum(corpus_counts[c] for c in cats)
    weight = {c: corpus_counts[c] / total_w for c in cats}
    p = {c: by[c][0] / by[c][1] for c in cats}

    rng = random.Random(seed)
    cat_draws: dict = {c: [] for c in cats}
    overall_draws: list[float] = []
    for _ in range(n_boot):
        acc = 0.0
        for c in cats:
            k, n = by[c]
            if k in (0, n):  # all-same outcome: every resample is identical
                a = p[c]
            else:
                a = sum(rng.random() < p[c] for _ in range(n)) / n
            cat_draws[c].append(a)
            acc += weight[c] * a
        overall_draws.append(acc)

    def ci(vals: list[float]) -> list[float]:
        s = sorted(vals)
        return [round(_percentile(s, 0.025), 4), round(_percentile(s, 0.975), 4)]

    per_cat = {c: {"passed": by[c][0], "total": by[c][1], "accuracy": round(p[c], 4), "ci95": ci(cat_draws[c])}
               for c in cats}
    passed, total = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    return {
        "per_category": per_cat,
        "weighted": {"accuracy": round(sum(weight[c] * p[c] for c in cats), 4), "ci95": ci(overall_draws),
                     "weights": {c: round(weight[c], 6) for c in cats}},
        "unweighted": {"passed": passed, "total": total, "accuracy": round(passed / total, 4)},
        "bootstrap": {"resamples": n_boot, "seed": seed, "unit": "test cases within category"},
    }


# ------------------------------------------------------------- diagnostics
def diagnose_generation(raw: str, cleaned: str, *, n_new_tokens: int | None = None,
                        max_new_tokens: int | None = None, finish_reason: str | None = None) -> dict:
    """Answer 'did the model even try?' for one generation (Phase 1 step 4)."""
    try:
        tree = ast.parse(cleaned)
        parses = True
        defines = any(isinstance(n, ast.FunctionDef) and n.name == DEFAULT_ENTRY_POINT for n in tree.body)
    except SyntaxError:
        parses, defines = False, False
    hit_max = None if None in (n_new_tokens, max_new_tokens) else n_new_tokens >= max_new_tokens
    return {
        "n_chars_raw": len(raw), "n_chars_cleaned": len(cleaned),
        "has_markdown_fence": "```" in raw,
        "cleaning_changed_code": raw.strip() != cleaned,
        "parses": parses, "defines_parse_amount": defines,
        "mentions_written_number_words": bool(WRITTEN_LOGIC_RE.search(cleaned)),
        "n_new_tokens": n_new_tokens, "max_new_tokens": max_new_tokens,
        "hit_max_new_tokens": hit_max, "finish_reason": finish_reason,
    }


def loss_summary(log_history: list[dict]) -> dict:
    losses = [h["loss"] for h in log_history if "loss" in h]
    if not losses:
        return {"n_logged": 0, "first": None, "last": None, "min": None, "decreased": None}
    head = losses[: max(1, len(losses) // 4)]
    tail = losses[-max(1, len(losses) // 4):]
    return {"n_logged": len(losses), "first": round(losses[0], 4), "last": round(losses[-1], 4),
            "min": round(min(losses), 4),
            "decreased": (sum(tail) / len(tail)) < (sum(head) / len(head))}


# ------------------------------------------------------------- aggregation
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def aggregate_runs(summaries: list[dict]) -> dict:
    """Group run summaries by (arm, n_examples) and report cross-seed spread.

    With 3 seeds a standard deviation is a rough indication only; the full
    per-seed numbers are kept so nothing is hidden behind the mean.
    """
    groups: dict = defaultdict(list)
    for s in summaries:
        if s.get("kind") == "sft_run":
            groups[(s["arm"], s["n_examples"])].append(s)
    out = {}
    for (arm, n), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        runs = sorted(runs, key=lambda r: r["seed"])
        w = [r["metrics"]["eval"]["weighted"]["accuracy"] for r in runs]
        cats = sorted({c for r in runs for c in r["metrics"]["eval"]["per_category"]})
        per_cat = {}
        for c in cats:
            accs = [r["metrics"]["eval"]["per_category"][c]["accuracy"] for r in runs
                    if c in r["metrics"]["eval"]["per_category"]]
            per_cat[c] = {"mean": round(_mean(accs), 4), "min": min(accs), "max": max(accs), "per_seed": accs}
        out[f"{arm}:n{n}"] = {
            "arm": arm, "n_examples": n, "seeds": [r["seed"] for r in runs],
            "weighted_accuracy": {"mean": round(_mean(w), 4),
                                  "std": None if _std(w) is None else round(_std(w), 4),
                                  "per_seed": w,
                                  "per_seed_ci95": [r["metrics"]["eval"]["weighted"]["ci95"] for r in runs]},
            "hand_grader_passed": [r["metrics"]["hand_grader"]["passed"] for r in runs],
            "per_category": per_cat,
        }
    return out


def render_ablation_table(agg: dict) -> str:
    if not agg:
        return "\nNo v4 training runs are recorded in the ledger yet.\n"
    cats = sorted({c for g in agg.values() for c in g["per_category"]})
    head = ["Arm", "Train n", "Seeds", "Weighted acc. (mean, std)", "Hand grader /24"] + cats
    rows = ["", "| " + " | ".join(head) + " |", "|" + "|".join(["---"] * 5 + ["---:"] * len(cats)) + "|"]
    for g in agg.values():
        wa = g["weighted_accuracy"]
        std = "n/a" if wa["std"] is None else f"{wa['std'] * 100:.1f}"
        cells = [g["arm"], str(g["n_examples"]), str(len(g["seeds"])), f"{wa['mean'] * 100:.1f}% (std {std})",
                 "/".join(str(x) for x in g["hand_grader_passed"])]
        for c in cats:
            pc = g["per_category"].get(c)
            cells.append("" if pc is None else f"{pc['mean'] * 100:.1f}% [{pc['min'] * 100:.0f}-{pc['max'] * 100:.0f}]")
        rows.append("| " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("Per-category cells are the mean across seeds with the [min-max] seed range. "
                "Per-seed bootstrap 95% CIs are in the run summaries under `experiments/`.")
    return "\n".join(rows) + "\n"
