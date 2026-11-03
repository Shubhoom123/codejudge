#!/usr/bin/env python3
"""Build the results ledger: every headline number in the README is produced here.

Rule for this repo: no number goes in the README (or on a resume) unless it
is an entry in results_ledger.json. Each entry records what produced it:
the git commit (and whether the tree was dirty), the date, the seed and
model tag where they apply, the source artifact, and the command that
regenerates it. `docs/RESULTS.md` and the numbers inside the README's
`<!--ledger:ID-->...<!--/ledger-->` spans are *rendered from the ledger*,
never typed by hand.

Entry kinds (see KINDS):
  measured        computed by this script from code or an artifact in this run
  recorded        read from an artifact written by an earlier run that this
                  script cannot re-execute (e.g. a gitignored results/run_*.json
                  that needed Ollama and a fine-tuned model)
  manifest_claim  taken from a manifest; the underlying shards were not
                  available to re-count, so it is NOT independently verified
  unverified      a number someone has claimed that no artifact supports;
                  value is null on purpose

Usage:
    python tools/build_ledger.py                       # measure + write ledger, docs/RESULTS.md, README spans
    python tools/build_ledger.py --skip-corpus-verify  # fast dev run; corpus entries become manifest_claim
    python tools/build_ledger.py --check               # no measuring: README / RESULTS.md agree with the committed ledger?

Run it on the machine that holds the gitignored artifacts (shards, results/).
The corpus verification streams ~3 GB of shards and takes a few minutes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from test_case_store import reservoir_sample  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import evaluation as ev  # noqa: E402
import make_eval_set as mes  # noqa: E402

LEDGER_PATH = ROOT / "results_ledger.json"
RESULTS_MD = ROOT / "docs" / "RESULTS.md"
README = ROOT / "README.md"
CORPUS_DIR = ROOT / "data" / "test_cases_bulk"
BULK_EVAL = ROOT / "data" / "qwen2_codejudge_bulk_eval.json"
TRAINING_DATA = ROOT / "data" / "training_runs" / "ambigous_spec" / "training_data.json"
RESULTS_DIR = ROOT / "results"
EXPERIMENTS_DIR = ROOT / "experiments"
EVAL_SET = ROOT / "data" / "eval_sets" / "ambigous_spec_strat2000_seed42.jsonl"

SCHEMA_VERSION = 1
KINDS = {"measured", "recorded", "manifest_claim", "unverified"}
SPAN_RE = re.compile(r"<!--ledger:([A-Za-z0-9_.\-]+)-->(.*?)<!--/ledger-->", re.DOTALL)


# --------------------------------------------------------------------------
# provenance helpers
# --------------------------------------------------------------------------
def _git(*args: str) -> str:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


class Stamp:
    """Captured once at the start of a run so every entry shares one commit/date."""

    def __init__(self) -> None:
        self.commit = _git("rev-parse", "HEAD")
        self.dirty = bool(_git("status", "--porcelain"))
        self.now = datetime.now(timezone.utc)

    def entry(self, value, *, kind: str, source: str, command: str, seed=None,
              model_tag=None, notes: str | None = None, **extra) -> dict:
        assert kind in KINDS, kind
        e = {
            "value": value, "kind": kind, "source": source, "command": command,
            "commit": self.commit, "dirty": self.dirty,
            "date": self.now.strftime("%Y-%m-%d"),
            "seed": seed, "model_tag": model_tag,
        }
        if notes:
            e["notes"] = notes
        e.update(extra)
        return e


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------
def _pytest_env() -> dict:
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def count_def_tests() -> tuple[int, dict]:
    per = {}
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        per[p.name] = len(re.findall(r"^\s*def test_", p.read_text(), flags=re.M))
    return sum(per.values()), per


def collect_tests() -> tuple[int, dict]:
    """Number of items pytest would run (parametrization expanded)."""
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=",
         "-p", "no:cacheprovider"],
        cwd=ROOT, env=_pytest_env(), capture_output=True, text=True)
    ids = [ln for ln in p.stdout.splitlines() if "::" in ln]
    per = Counter(ln.split("::")[0].split("/")[-1] for ln in ids)
    return len(ids), dict(per)


def run_suite() -> dict:
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "junit.xml"
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-o", "addopts=",
             "-p", "no:cacheprovider", f"--junitxml={xml}"],
            cwd=ROOT, env=_pytest_env(), capture_output=True, text=True)
        root = ET.parse(xml).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        n, f, e, s = (int(suite.get(k)) for k in ("tests", "failures", "errors", "skipped"))
    return {"collected": n, "passed": n - f - e - s, "failed": f + e, "skipped": s}


def measure_tests(st: Stamp, suite: dict | None) -> dict:
    n, per = collect_tests()
    defs, defs_per = count_def_tests()
    cmd = "python -m pytest -o addopts= -q"
    out = {
        "tests.collected": st.entry(n, kind="measured", source="pytest --collect-only", command=cmd + " --collect-only",
                                    notes="Parametrized cases are counted individually."),
        "tests.files": st.entry(len(per), kind="measured", source="pytest --collect-only", command=cmd + " --collect-only"),
        "tests.per_file": st.entry(per, kind="measured", source="pytest --collect-only", command=cmd + " --collect-only"),
        "tests.def_test_functions": st.entry(
            defs, kind="measured", source="tests/test_*.py (regex)", command="grep -c 'def test_' tests/*.py",
            notes="Undercounts: parametrized tests expand into several cases.", per_file=defs_per),
    }
    for key in ("passed", "failed", "skipped"):
        out[f"tests.{key}"] = st.entry(
            None if suite is None else suite[key], kind="measured", source="pytest --junitxml", command=cmd,
            notes=("Platform-dependent: the RLIMIT_AS memory test is skipped on macOS, so a Mac run is "
                   "59 passed + 1 skipped. See environment in this ledger for where this was measured."))
    return out


CAT_MARK = '"category": "'


def reservoir_cases(corpus_dir: Path, k: int, seed: int) -> list[dict]:
    """test_case_store.reservoir_sample, but JSON-parsing only the k lines it keeps.

    It consumes the RNG exactly as reservoir_sample does (one randint(0, i)
    per line after the first k), so the result is identical; tests assert that.
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    i = 0
    for shard in sorted(Path(corpus_dir).glob("shard_*.jsonl")):
        with shard.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if i < k:
                    reservoir.append(line)
                else:
                    j = rng.randint(0, i)
                    if j < k:
                        reservoir[j] = line
                i += 1
    return [json.loads(x) for x in reservoir]


def _corpus_job(corpus_dir: Path):
    return stream_corpus(sorted(Path(corpus_dir).glob("shard_*.jsonl")))


def _bulk_job(corpus_dir: Path, k: int, seed: int):
    return reservoir_cases(corpus_dir, k, seed)


def _eval_set_job(corpus_dir: Path, size: int, floor: int, seed: int):
    return mes.build(corpus_dir, size, floor, seed)[1]["cases_sha256"]


def stream_corpus(shards: list[Path]) -> tuple[int, Counter]:
    cats: Counter = Counter()
    lines = 0
    for p in shards:
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                lines += 1
                i = line.rfind(CAT_MARK)
                if i < 0:
                    cats["<no category>"] += 1
                    continue
                j = line.find('"', i + len(CAT_MARK))
                cats[line[i + len(CAT_MARK):j]] += 1
    return lines, cats


def measure_corpus(st: Stamp, verify: bool, streamed=None) -> tuple[dict, bool]:
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    shards = sorted(CORPUS_DIR.glob("shard_*.jsonl"))
    verified = False
    if verify and shards:
        total, cats = streamed if streamed is not None else stream_corpus(shards)
        by_cat = dict(cats)
        mism = {k: {"manifest": manifest["by_category"].get(k), "shards": by_cat.get(k)}
                for k in set(by_cat) | set(manifest["by_category"])
                if manifest["by_category"].get(k) != by_cat.get(k)}
        verified = total == manifest["total_cases"] and not mism
        kind, cmd = "measured", "python tools/build_ledger.py   # streams every shard"
        extra = {"verified_against_manifest": verified, "manifest_total": manifest["total_cases"],
                 "mismatches": mism, "shard_files": len(shards)}
    else:
        total, by_cat = manifest["total_cases"], manifest["by_category"]
        kind, cmd = "manifest_claim", "python tools/generate_test_cases_bulk.py   # then re-run build_ledger"
        extra = {"verified_against_manifest": None, "shard_files": len(shards)}
    src = rel(CORPUS_DIR / "manifest.json")
    return {
        "corpus.total_cases": st.entry(total, kind=kind, source=src, command=cmd, **extra),
        "corpus.by_category": st.entry(by_cat, kind=kind, source=src, command=cmd),
        "corpus.written_number_pct": st.entry(
            by_cat["written_number"] / total, kind=kind, source=src, command=cmd,
            notes="written_number cases / total cases (fraction, not percent)."),
        "corpus.manifest_note": st.entry(manifest.get("note"), kind="recorded", source=src, command="cat " + src),
    }, verified


def measure_bulk_eval(st: Stamp, verify_sample: bool, sample=None) -> dict:
    d = json.loads(BULK_EVAL.read_text())
    rs = d["results"]
    by: dict = defaultdict(lambda: [0, 0])
    for r in rs:
        by[r["category"]][0] += bool(r["passed"])
        by[r["category"]][1] += 1
    passed = sum(v[0] for v in by.values())
    total = len(rs)
    cats = {c: {"passed": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 4)} for c, v in sorted(by.items())}
    consistent = (round(passed / total, 4) == d["overall_accuracy"]
                  and all(d["by_category"][c]["passed"] == cats[c]["passed"]
                          and d["by_category"][c]["total"] == cats[c]["total"] for c in cats))
    cmd = (f"python tools/evaluate_at_scale.py --model {d['model']} --task {d['task']} "
           f"--samples {d['samples']} --seed {d['seed']} --output {rel(BULK_EVAL)}")
    out = {"bulk_eval.primary": st.entry(
        {"passed": passed, "total": total, "accuracy": round(passed / total, 4), "by_category": cats,
         "stored_summary_consistent": consistent},
        kind="measured", source=rel(BULK_EVAL), command=cmd, seed=d["seed"], model_tag=d["model"],
        artifact_sha256=sha256_file(BULK_EVAL),
        notes=("Recomputed from the per-case `results` list, not from the file's summary fields. "
               "Sampling is a uniform reservoir sample (Algorithm R), NOT stratified. "
               "This is the only 2000-case bulk eval in the repo; none exists for v2 or v3."))}
    match = None
    if verify_sample and list(CORPUS_DIR.glob("shard_*.jsonl")):
        sample = sample if sample is not None else reservoir_sample(CORPUS_DIR, k=d["samples"], seed=d["seed"])
        match = ([(c["input"], c["category"]) for c in sample]
                 == [(r["input"], r["category"]) for r in rs])
    out["bulk_eval.primary.sample_matches_corpus"] = st.entry(
        match, kind="measured", source="test_case_store.reservoir_sample", command=(
            f"python -c \"from test_case_store import reservoir_sample as r; "
            f"r('data/test_cases_bulk', k={d['samples']}, seed={d['seed']})\""),
        seed=d["seed"],
        notes=("True = re-drawing the seeded sample from the current shards reproduces the eval file's inputs "
               "exactly (same order). null = shards unavailable, not checked."))
    return out


def measure_saved_grader_runs(st: Stamp) -> dict:
    latest: dict = {}
    for f in sorted(RESULTS_DIR.glob("run_*.json")):
        for t in json.loads(f.read_text()):
            if t.get("task") != "ambigous_spec":
                continue
            for tag, v in t.get("models", {}).items():
                s = v.get("scores") if isinstance(v, dict) else None
                if s and "total" in s and tag.startswith("qwen2-codejudge"):
                    latest[tag] = (f, s)
    out = {}
    for tag, (f, s) in sorted(latest.items()):
        failed = [x["name"] for x in s.get("tests", []) if not x["passed"]]
        out[f"grader.ambigous_spec.{tag}"] = st.entry(
            {"passed": s["passed"], "total": s["total"], "failed_tests": failed},
            kind="recorded", source=rel(f), command=f"python evaluator.py --model {tag} --task ambigous_spec",
            model_tag=tag, artifact_sha256=sha256_file(f),
            notes="results/ is gitignored; re-running needs Ollama and the fine-tuned model, which are not in the repo.")
    return out


def measure_training(st: Stamp) -> dict:
    d = json.loads(TRAINING_DATA.read_text())
    graded = [x for x in d if "tests" in x["scores"]]
    perfect = [x for x in d if x["score"] >= 1.0]
    fails: dict = defaultdict(int)
    tot: dict = defaultdict(int)
    for x in graded:
        for t in x["scores"]["tests"]:
            tot[t["name"]] += 1
            fails[t["name"]] += not t["passed"]
    rates = {k: {"failed": fails[k], "total": tot[k], "rate": round(fails[k] / tot[k], 6)} for k in tot}
    hist = Counter(round(x["score"], 2) for x in d)
    src, cmd = rel(TRAINING_DATA), "python -c 'import json; json.load(open(\"%s\"))'  # see tools/build_ledger.py::measure_training" % rel(TRAINING_DATA)
    note = "The file records no teacher model tag, seed or temperature, so this run cannot be reproduced from the repo."
    return {
        "training.attempts": st.entry(len(d), kind="measured", source=src, command=cmd, notes=note,
                                      artifact_sha256=sha256_file(TRAINING_DATA)),
        "training.attempts_graded": st.entry(len(graded), kind="measured", source=src, command=cmd,
                                             notes="Attempts whose grader output contains per-test results; the rest errored."),
        "training.perfect": st.entry(len(perfect), kind="measured", source=src, command=cmd),
        "training.perfect_pct": st.entry(len(perfect) / len(d), kind="measured", source=src, command=cmd,
                                         notes="perfect attempts / all attempts (fraction)."),
        "training.perfect_unique_solutions": st.entry(len({x["solution"].strip() for x in perfect}),
                                                      kind="measured", source=src, command=cmd),
        "training.score_histogram": st.entry({str(k): v for k, v in sorted(hist.items(), reverse=True)},
                                             kind="measured", source=src, command=cmd),
        "training.test_fail_rates": st.entry(rates, kind="measured", source=src, command=cmd,
                                             notes="Per grader test, over graded attempts."),
    }


def measure_dpo_pairs(st: Stamp) -> dict:
    """Preference pairs rebuilt from the teacher attempts (trainer/dpo_pairs.py), checked against the committed meta."""
    from trainer import dpo_pairs
    seed = 0
    meta_path_ = ROOT / "data" / "dpo" / f"ambigous_spec_pairs_seed{seed}.meta.json"
    if not TRAINING_DATA.exists() or not meta_path_.exists():
        return {}
    _, meta = dpo_pairs.build(TRAINING_DATA, "ambigous_spec", seed=seed)
    committed = json.loads(meta_path_.read_text())
    src = rel(TRAINING_DATA)
    cmd = "python trainer/dpo_pairs.py   # then python tools/build_ledger.py"
    d = meta["diagnostics"]
    return {
        "dpo.pairs.n_pairs": st.entry(d["n_pairs"], kind="measured", source=src, command=cmd, seed=seed,
                                      artifact_sha256=meta["pairs_sha256"]),
        "dpo.pairs.matches_committed_meta": st.entry(committed["pairs_sha256"] == meta["pairs_sha256"], kind="measured",
                                                     source=rel(meta_path_), command=cmd, seed=seed),
        "dpo.pairs.diagnostics": st.entry(d, kind="measured", source=src, command=cmd, seed=seed,
                                          notes="Grader-crash attempts (score 0, no per-test results) are never used as rejected."),
    }


def measure_eval_set(st: Stamp, verify: bool, redrawn_sha=None) -> dict:
    if not EVAL_SET.exists():
        return {}
    cases, meta = ev.load_eval_set(EVAL_SET)  # raises if the file no longer matches its sha256
    reproduces = None
    if verify and list(CORPUS_DIR.glob("shard_*.jsonl")):
        sha = redrawn_sha or _eval_set_job(CORPUS_DIR, meta["size"], meta["floor"], meta["seed"])
        reproduces = sha == meta["cases_sha256"]
    return {"eval_set.strat2000": st.entry(
        {"size": len(cases), "floor": meta["floor"], "allocation": meta["allocation"],
         "cases_sha256": meta["cases_sha256"], "reproduces_from_corpus": reproduces},
        kind="measured", source=rel(EVAL_SET), seed=meta["seed"], artifact_sha256=sha256_file(EVAL_SET),
        command=f"python tools/make_eval_set.py --size {meta['size']} --floor {meta['floor']} --seed {meta['seed']}",
        notes="Stratified: floor per category, remainder proportional to corpus size. true = re-drawing from the "
              "current shards gives byte-identical cases; null = shards unavailable, not checked.")}


def measure_experiments(st: Stamp) -> dict:
    """Run summaries written by trainer/sft_train.py (Colab) and tools/score_saved_solutions.py."""
    out: dict = {}
    summaries = []
    for f in sorted(EXPERIMENTS_DIR.glob("**/*.json")) if EXPERIMENTS_DIR.exists() else []:
        if f.name.endswith(".per_case.json"):
            continue
        d = json.loads(f.read_text())
        if d.get("kind") not in {"sft_run", "baseline"}:
            continue
        summaries.append(d)
        e = d["metrics"]["eval"]
        written = e["per_category"].get("written_number", {})
        out[f"exp.{d['run_id']}"] = st.entry(
            {"weighted_accuracy": e["weighted"]["accuracy"], "weighted_ci95": e["weighted"]["ci95"],
             "written_number_accuracy": written.get("accuracy"),
             "hand_grader_passed": d["metrics"]["hand_grader"]["passed"],
             "hand_grader_total": d["metrics"]["hand_grader"]["total"]},
            kind="recorded", source=rel(f), seed=d.get("seed"), model_tag=d.get("model_tag"),
            command=("python trainer/sft_train.py --config trainer/configs/sft_v4.json "
                     f"--n {d['n_examples']} --seed {d['seed']} --arm {d['arm']}   # needs a CUDA GPU (Colab T4)"
                     if d["kind"] == "sft_run" else "python tools/score_saved_solutions.py"),
            artifact_sha256=sha256_file(f), run_code_commit=d.get("code_commit"), run_dirty=d.get("dirty"),
            run_date=d.get("date"), run_data_sha256=d.get("data_sha256"), run_config_sha256=d.get("config_sha256"),
            run_eval_set_sha256=d.get("eval_set_sha256"), run_env=d.get("env"),
            run_solution_sha256=d.get("solution_sha256"),
            notes="Written by the run itself; this script only reads it. Not re-executable without the original "
                  "hardware/model." if d["kind"] == "sft_run" else
                  "Saved solution text from a gitignored results/ file, re-scored on the frozen eval set.")
    out["phase1.ablation"] = st.entry(
        ev.aggregate_runs(summaries), kind="measured", source="experiments/**/*.json",
        command="python tools/build_ledger.py   # evaluation.aggregate_runs over the run summaries",
        notes="Mean and std across seeds of corpus-weighted accuracy; per-category mean and min-max. "
              "Bootstrap CIs resample test cases only (see evaluation.py docstring).")
    out["phase1.runs_recorded"] = st.entry(
        sum(d["kind"] == "sft_run" for d in summaries), kind="measured", source="experiments/**/*.json",
        command="python tools/build_ledger.py")
    return out


def search_for_figure(pattern: str, git_pattern: str) -> dict:
    """Look for a claimed figure everywhere a result could plausibly live."""
    rx = re.compile(pattern)
    roots = [README, ROOT / "docs", RESULTS_DIR, ROOT / "data", ROOT / ".history"]
    hits, scanned = [], 0
    for r in roots:
        files = [r] if r.is_file() else (sorted(p for p in r.rglob("*") if p.is_file()) if r.exists() else [])
        for p in files:
            if p.suffix not in {".md", ".json", ".jsonl", ".txt", ".py"} or p.name in {"test_cases.json", "RESULTS.md"} \
                    or "test_cases_bulk" in p.parts or p.name == "results_ledger.json":
                continue
            if p.stat().st_size > 50_000_000:
                continue
            scanned += 1
            if rx.search(p.read_text(errors="ignore")):
                hits.append(rel(p))
    git_hits = _git("log", "--all", "--oneline", "-G", git_pattern, "--", ".", ":!data/test_cases.json",
                    ":!data/test_cases_bulk", ":!results_ledger.json", ":!docs/RESULTS.md").splitlines()
    return {"files_scanned": scanned, "file_hits": hits, "git_commit_hits": git_hits}


def measure_unverified(st: Stamp) -> dict:
    pat = r"72\.8\s*%|(?<![\d.])0?\.728(?!\d)"
    found = search_for_figure(pat, r"72\.8 ?%|0?\.728")
    return {
        "unverified.resume_bulk_72_8": st.entry(
            None, kind="unverified", source="resume (not in repo)",
            command="python tools/build_ledger.py   # search_for_figure(%r)" % pat,
            notes=("Claim: 'v1/v3 scored 72.8% at scale'. No artifact in the repo contains this figure, and the only "
                   "2000-case bulk eval (seed 42) is v1 at 51.8%. No bulk eval exists for v2 or v3."),
            search=found),
        "unverified.v1_v3_training_set_composition": st.entry(
            None, kind="unverified", source="README.md (Fine-tuning > Where things actually stand)",
            command="n/a - the training files for v1-v3 are not in the repo",
            notes=("README says v1 trained on 83 examples (77 near-misses scoring 0.83-0.96, 6 perfect), v2 on 33 and "
                   "v3 on 20 perfect examples. Those datasets and the Colab notebook are not in the repo, so these "
                   "counts cannot be re-derived. The grader scores they produced (20/24 each) are recorded under grader.*.")),
        "unverified.perfect_hit_rate_5_12pct": st.entry(
            None, kind="unverified", source="README.md (Known limitations)",
            command="n/a - only one teacher run (training.*) is retained",
            notes="README cites a '~5-12%' perfect-score hit rate across runs. Only the 1000-attempt run is retained (training.perfect_pct)."),
    }


def build_ledger(verify_corpus: bool) -> dict:
    st = Stamp()
    entries: dict = {}
    streamed = sample = redrawn = None
    if verify_corpus and list(CORPUS_DIR.glob("shard_*.jsonl")):
        bulk = json.loads(BULK_EVAL.read_text())
        meta = json.loads(ev.meta_path(EVAL_SET).read_text()) if EVAL_SET.exists() else None
        with ProcessPoolExecutor(max_workers=3) as pool:  # three independent passes over the shards
            f_corpus = pool.submit(_corpus_job, CORPUS_DIR)
            f_bulk = pool.submit(_bulk_job, CORPUS_DIR, bulk["samples"], bulk["seed"])
            f_eval = pool.submit(_eval_set_job, CORPUS_DIR, meta["size"], meta["floor"], meta["seed"]) if meta else None
            streamed, sample = f_corpus.result(), f_bulk.result()
            redrawn = f_eval.result() if f_eval else None
    corpus, verified = measure_corpus(st, verify_corpus, streamed)
    entries.update(corpus)
    entries.update(measure_bulk_eval(st, verify_corpus, sample))
    entries.update(measure_saved_grader_runs(st))
    entries.update(measure_training(st))
    entries.update(measure_dpo_pairs(st))
    entries.update(measure_eval_set(st, verify_corpus, redrawn))
    entries.update(measure_experiments(st))
    entries.update(measure_unverified(st))
    entries.update(measure_tests(st, None))  # provisional: lets the ledger-consistency tests pass inside the real run
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "tools/build_ledger.py",
        "generated_at": st.now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"commit": st.commit, "dirty": st.dirty, "branch": _git("rev-parse", "--abbrev-ref", "HEAD")},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine()},
        "entries": entries,
    }
    write_outputs(ledger)  # tests below run against these files
    entries.update(measure_tests(st, run_suite()))
    write_outputs(ledger)
    return ledger


# --------------------------------------------------------------------------
# rendering (ledger -> README spans, docs/RESULTS.md)
# --------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _corpus_table(E: dict) -> str:
    total = E["corpus.total_cases"]["value"]
    rows = ["", "| Category | Cases | Share of corpus |", "|---|---:|---:|"]
    for c, n in sorted(E["corpus.by_category"]["value"].items(), key=lambda kv: -kv[1]):
        rows.append(f"| {c} | {n:,} | {pct(n / total)} |")
    rows.append(f"| **total** | **{total:,}** | 100.0% |")
    return "\n".join(rows) + "\n"


def _bulk_overall(E: dict) -> str:
    v = E["bulk_eval.primary"]["value"]
    return f"{v['passed']}/{v['total']} = {pct(v['passed'] / v['total'])}"


def _bulk_table(E: dict) -> str:
    e = E["bulk_eval.primary"]
    v = e["value"]
    rows = ["", f"Overall: **{_bulk_overall(E)}** (`{e['model_tag']}`, uniform sample, seed {e['seed']})", "",
            "| Category | Passed | Total | Accuracy |", "|---|---:|---:|---:|"]
    for c, s in sorted(v["by_category"].items(), key=lambda kv: -kv[1]["total"]):
        rows.append(f"| {c} | {s['passed']} | {s['total']} | {pct(s['accuracy'])} |")
    return "\n".join(rows) + "\n"


def _written_fail_rates(E: dict) -> str:
    r = E["training.test_fail_rates"]["value"]
    names = ["compound_written_number", "large_written_million", "written_thousand"]
    return ", ".join(f"`{n}` {pct(r[n]['rate'])}" for n in names)


def _baselines_table(E: dict) -> str:
    rows = ["", "| Model (saved solution) | Weighted accuracy | `written_number` | Hand grader |", "|---|---:|---:|---:|"]
    n = 0
    for k, e in sorted(E.items()):
        if k.startswith("exp.baseline_"):
            v = e["value"]
            rows.append(f"| `{e['model_tag']}` | {pct(v['weighted_accuracy'])} | {pct(v['written_number_accuracy'])} | "
                        f"{v['hand_grader_passed']}/{v['hand_grader_total']} |")
            n += 1
    return "\n".join(rows) + "\n" if n else "\nNo baselines recorded.\n"


def _phase1_status(E: dict) -> str:
    n = E["phase1.runs_recorded"]["value"]
    return "no v4 training run is recorded yet" if n == 0 else f"{n} v4 training runs recorded"


def _dpo_pairs_summary(E: dict) -> str:
    d = E["dpo.pairs.diagnostics"]["value"]
    t, c = d["tier_counts"], d["chars"]
    return (f"{d['n_pairs']} pairs from {d['n_unique_chosen']} perfect attempts and {d['n_unique_rejected']} distinct "
            f"lower-scoring ones (tiers: {t['near']} near, {t['mid']} mid, {t['far']} far; mean score margin "
            f"{d['margin']['mean']:.3f}, minimum {d['margin']['min']:.3f}). The rejected attempt fails a written-number test in "
            f"{pct(d['frac_pairs_rejected_fails_a_written_number_test'])} of pairs, and is longer than the chosen one in "
            f"{pct(c['frac_pairs_rejected_longer'])} of pairs ({c['rejected_mean']:.0f} vs {c['chosen_mean']:.0f} characters on average)")


RENDERERS = {
    "dpo.pairs.summary": _dpo_pairs_summary,
    "tests.collected": lambda E: str(E["tests.collected"]["value"]),
    "tests.files": lambda E: str(E["tests.files"]["value"]),
    "corpus.total_cases": lambda E: f"{E['corpus.total_cases']['value']:,}",
    "corpus.written_number": lambda E: f"{E['corpus.by_category']['value']['written_number']:,}",
    "corpus.written_number_pct": lambda E: pct(E["corpus.written_number_pct"]["value"]),
    "corpus.table": _corpus_table,
    "bulk_eval.overall": _bulk_overall,
    "bulk_eval.table": _bulk_table,
    "training.attempts": lambda E: f"{E['training.attempts']['value']:,}",
    "training.perfect": lambda E: str(E["training.perfect"]["value"]),
    "training.perfect_pct": lambda E: pct(E["training.perfect_pct"]["value"]),
    "training.written_fail_rates": _written_fail_rates,
    "eval_baselines.table": _baselines_table,
    "phase1.table": lambda E: ev.render_ablation_table(E["phase1.ablation"]["value"]),
    "phase1.status": _phase1_status,
}


def replace_spans(text: str, entries: dict, renderers: dict | None = None) -> tuple[str, list[str]]:
    renderers = RENDERERS if renderers is None else renderers
    used: list[str] = []

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in renderers:
            raise KeyError(f"README references unknown ledger id: {key}")
        used.append(key)
        return f"<!--ledger:{key}-->{renderers[key](entries)}<!--/ledger-->"

    return SPAN_RE.sub(sub, text), used


def _v(E: dict, key: str):
    return E[key]["value"]


def render_results_md(ledger: dict) -> str:
    E, g, env = ledger["entries"], ledger["git"], ledger["environment"]
    L: list[str] = []
    a = L.append
    a("# Results ledger")
    a("")
    a("> Generated by `tools/build_ledger.py` from `results_ledger.json`. **Do not edit by hand.**")
    a("> Rule: no number goes in the README or on a resume unless it is an entry in the ledger.")
    a("")
    a(f"- Code commit: `{g['commit'][:10]}` ({'tree was DIRTY' if g['dirty'] else 'clean tree'}), branch `{g['branch']}`")
    a(f"- Generated: {ledger['generated_at']}")
    a(f"- Environment: Python {env['python']}, {env['platform']} ({env['machine']})")
    a("")
    a("Entry kinds: **measured** (computed in this run), **recorded** (read from an artifact of an earlier run that "
      "cannot be re-executed here), **manifest_claim** (taken from a manifest, not re-counted), "
      "**unverified** (claimed somewhere, no artifact supports it).")
    a("")
    a("## Tests")
    a("")
    t = E["tests.collected"]["value"]
    a(f"- pytest collects **{t}** tests across {_v(E, 'tests.files')} files; "
      f"passed {_v(E, 'tests.passed')}, failed {_v(E, 'tests.failed')}, skipped {_v(E, 'tests.skipped')} "
      f"(measured on {env['platform']}, Python {env['python']}).")
    a(f"- `grep 'def test_'` finds {_v(E, 'tests.def_test_functions')} functions. The difference is parametrization "
      "(`test_api` bad filenames, `test_graders` per task), which pytest expands into separate cases.")
    a(f"- On macOS the `RLIMIT_AS` memory test is skipped (documented in `sandbox.py`), so a Mac run reads {t - 1} passed + 1 skipped.")
    a("")
    a("| File | Collected | `def test_` |")
    a("|---|---:|---:|")
    defs = E["tests.def_test_functions"]["per_file"]
    for f, n in sorted(_v(E, "tests.per_file").items()):
        a(f"| {f} | {n} | {defs.get(f, 0)} |")
    a("")
    a("## Test-case corpus")
    a("")
    c = E["corpus.total_cases"]
    if c["kind"] == "measured":
        status = (f"**Verified**: all {c['shard_files']} shards were streamed and the line count and per-category counts "
                  "match the manifest." if c.get("verified_against_manifest")
                  else f"**MISMATCH between manifest and shards**: {c.get('mismatches')}")
    else:
        status = "**Not verified**: shards were unavailable or verification was skipped; numbers are the manifest's claim."
    a(status)
    a("")
    a(_corpus_table(E).strip("\n"))
    a("")
    a(f"Manifest note: {_v(E, 'corpus.manifest_note')}")
    a("")
    a("## Bulk eval (ambigous_spec, 2000-case sample)")
    a("")
    b = E["bulk_eval.primary"]
    m = E["bulk_eval.primary.sample_matches_corpus"]["value"]
    a(f"Source `{b['source']}` (sha256 `{b['artifact_sha256'][:12]}`), model tag `{b['model_tag']}`, seed {b['seed']}. "
      f"Recomputed from per-case results; matches the file's own summary: {b['value']['stored_summary_consistent']}.")
    a("")
    a(f"Re-drawing the seeded sample from the current shards reproduces the eval inputs exactly: "
      f"**{'yes' if m else ('NO' if m is False else 'not checked')}**.")
    a("")
    a("The sample is a **uniform** reservoir sample, so its category mix follows the corpus mix; it is not stratified.")
    a("")
    cats = b["value"]["by_category"]
    corp = _v(E, "corpus.by_category")
    ctot = _v(E, "corpus.total_cases")
    a("| Category | Passed | Total | Accuracy | Share of sample | Share of corpus |")
    a("|---|---:|---:|---:|---:|---:|")
    for cat, s in sorted(cats.items(), key=lambda kv: -kv[1]["total"]):
        a(f"| {cat} | {s['passed']} | {s['total']} | {pct(s['accuracy'])} | "
          f"{pct(s['total'] / b['value']['total'])} | {pct(corp.get(cat, 0) / ctot)} |")
    a(f"| **overall** | {b['value']['passed']} | {b['value']['total']} | {pct(b['value']['accuracy'])} | 100.0% | 100.0% |")
    a("")
    a("This is the only bulk eval in the repo. There is none for v2 or v3.")
    a("")
    a("## Hand-written grader (24 cases), saved runs")
    a("")
    a("| Model tag | Score | Failed tests | Run file |")
    a("|---|---:|---|---|")
    for k, e in sorted(E.items()):
        if k.startswith("grader."):
            v = e["value"]
            a(f"| `{e['model_tag']}` | {v['passed']}/{v['total']} | {', '.join(v['failed_tests']) or 'none'} | `{e['source']}` |")
    a("")
    a("## Teacher data (`training_data.json`)")
    a("")
    a(f"- Attempts: {_v(E, 'training.attempts')} ({_v(E, 'training.attempts_graded')} with per-test results). "
      f"Perfect: {_v(E, 'training.perfect')} ({pct(_v(E, 'training.perfect_pct'))}), "
      f"{_v(E, 'training.perfect_unique_solutions')} distinct solution texts among them.")
    a(f"- File sha256 `{E['training.attempts']['artifact_sha256'][:12]}`. {E['training.attempts']['notes']}")
    a("")
    a("| Grader test | Failed | Of | Failure rate |")
    a("|---|---:|---:|---:|")
    for n, r in sorted(_v(E, "training.test_fail_rates").items(), key=lambda kv: -kv[1]["rate"])[:12]:
        a(f"| {n} | {r['failed']} | {r['total']} | {pct(r['rate'])} |")
    a("")
    a("## Stratified evaluation set")
    a("")
    if "eval_set.strat2000" in E:
        es = E["eval_set.strat2000"]
        v = es["value"]
        r = v["reproduces_from_corpus"]
        a(f"`{es['source']}` (sha256 `{v['cases_sha256'][:12]}`, seed {es['seed']}, floor {v['floor']} per category, "
          f"{v['size']} cases). Re-drawing it from the current shards gives byte-identical cases: "
          f"**{'yes' if r else ('NO' if r is False else 'not checked')}**.")
        a("")
        a("| Category | Cases in eval set | Cases in corpus |")
        a("|---|---:|---:|")
        for cat, n in sorted(v["allocation"].items(), key=lambda kv: -kv[1]):
            a(f"| {cat} | {n} | {_v(E, 'corpus.by_category').get(cat, 0):,} |")
        a("")
        a("Overall accuracy on this set is reweighted by corpus share (`weighted`), so it estimates accuracy on the full "
          "corpus. `invalid` (9) and `zero` (6) are taken whole; their per-category intervals are necessarily wide.")
        a("")
    a("## Baselines on the stratified set (saved v1-v3 solutions)")
    a("")
    a(_baselines_table(E).strip("\n"))
    a("")
    groups: dict = defaultdict(list)
    for k, e in sorted(E.items()):
        if k.startswith("exp.baseline_") and e.get("run_solution_sha256"):
            groups[e["run_solution_sha256"]].append(f"`{e['model_tag']}`")
    same = [" = ".join(v) for v in groups.values() if len(v) > 1]
    if same:
        a("Byte-identical solution texts: " + "; ".join(same) + ".")
        a("")
    a("Bootstrap intervals are degenerate here (a single deterministic solution passes or fails whole categories); "
      "see `evaluation.py`.")
    a("")
    a("## v4 SFT ablation")
    a("")
    a(ev.render_ablation_table(E["phase1.ablation"]["value"]).strip("\n"))
    a("")
    a("## Claims with no supporting artifact")
    a("")
    for k, e in sorted(E.items()):
        if e["kind"] == "unverified":
            a(f"- **{k}**: {e['notes']}")
            if "search" in e:
                s = e["search"]
                a(f"  - Searched {s['files_scanned']} files plus git history: file hits {s['file_hits'] or 'none'}, "
                  f"commit hits {s['git_commit_hits'] or 'none'}.")
    a("")
    a("## Entry index")
    a("")
    a("| Id | Kind | Seed | Model tag | Source |")
    a("|---|---|---|---|---|")
    for k, e in sorted(E.items()):
        a(f"| `{k}` | {e['kind']} | {e['seed'] if e['seed'] is not None else ''} | {e['model_tag'] or ''} | `{e['source']}` |")
    a("")
    return "\n".join(L)


def write_outputs(ledger: dict) -> None:
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, sort_keys=False) + "\n")
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(render_results_md(ledger))
    text = README.read_text()
    new, _ = replace_spans(text, ledger["entries"])
    if new != text:
        README.write_text(new)


def check() -> list[str]:
    """Do the README and RESULTS.md agree with the committed ledger? No measuring."""
    ledger = json.loads(LEDGER_PATH.read_text())
    problems = []
    text = README.read_text()
    try:
        new, used = replace_spans(text, ledger["entries"])
    except KeyError as e:
        return [str(e)]
    if not used:
        problems.append("README has no <!--ledger:...--> spans")
    if new != text:
        problems.append("README numbers differ from results_ledger.json (run tools/build_ledger.py)")
    if not RESULTS_MD.exists() or RESULTS_MD.read_text() != render_results_md(ledger):
        problems.append("docs/RESULTS.md is not what results_ledger.json renders to")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--skip-corpus-verify", action="store_true")
    args = ap.parse_args()
    if args.check:
        problems = check()
        for p in problems:
            print("FAIL:", p)
        print("ledger check:", "FAILED" if problems else "ok")
        return 1 if problems else 0
    ledger = build_ledger(verify_corpus=not args.skip_corpus_verify)
    E = ledger["entries"]
    print(f"commit {ledger['git']['commit'][:10]} dirty={ledger['git']['dirty']}")
    print(f"tests: {E['tests.collected']['value']} collected, {E['tests.passed']['value']} passed, "
          f"{E['tests.failed']['value']} failed, {E['tests.skipped']['value']} skipped")
    print(f"corpus: {E['corpus.total_cases']['value']:,} ({E['corpus.total_cases']['kind']})")
    print(f"bulk eval: {_bulk_overall(E)}  sample_matches_corpus={E['bulk_eval.primary.sample_matches_corpus']['value']}")
    problems = check()
    print("self-check:", "ok" if not problems else problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
