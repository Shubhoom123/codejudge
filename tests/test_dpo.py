"""Phase 2: preference pairs are well-formed, reproducible and non-degenerate; the DPO plan runs without a GPU."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trainer import dpo_pairs, dpo_train  # noqa: E402

TRAINING_DATA = ROOT / "data" / "training_runs" / "ambigous_spec" / "training_data.json"
NO_TORCH = ("import sys; sys.path.insert(0, %r); import trainer.dpo_train as s; "
            "assert 'torch' not in sys.modules and 'trl' not in sys.modules; "
            "raise SystemExit(s.main(['--dry-run']))" % str(ROOT))


def _attempt(i, score, failed=()):
    return {"id": i, "solution": f"def parse_amount(t):\n    return {i}", "score": score, "failed": list(failed)}


def test_tiers_and_pair_construction_on_synthetic_attempts():
    assert [dpo_pairs.tier_of(m) for m in (0.04, 0.08, 0.12, 0.17, 0.21, 0.71)] == [
        "near", "near", "mid", "mid", "far", "far"]
    attempts = [_attempt(0, 1.0), _attempt(1, 0.96), _attempt(2, 0.88), _attempt(3, 0.88), _attempt(4, 0.5),
                _attempt(5, 0.0, ["x"])]
    pairs = dpo_pairs.build_pairs(attempts, seed=0)
    assert [(p["tier"], p["margin"] > 0) for p in pairs] == [("near", True), ("mid", True), ("mid", True), ("far", True)]
    assert all(p["chosen_id"] == 0 and p["chosen"] != p["rejected"] for p in pairs)
    assert pairs == dpo_pairs.build_pairs(attempts, seed=0)          # deterministic
    attempts.append({**_attempt(9, 0.5), "solution": attempts[0]["solution"]})   # same text as the chosen one
    assert all(p["rejected_id"] != 9 for p in dpo_pairs.build_pairs(attempts, seed=0))


def test_real_pairs_are_not_degenerate_and_match_the_committed_meta():
    pairs, meta = dpo_pairs.build(TRAINING_DATA, "ambigous_spec", seed=0)
    d = meta["diagnostics"]
    assert d["n_pairs"] == 468 and d["n_unique_chosen"] == 117 and d["n_grader_crash_excluded"] == 9
    assert all(d["checks"].values()) and d["margin"]["min"] > 0 and d["margin"]["n_distinct"] > 3
    assert set(d["tier_counts"]) == {"near", "mid", "far"} and min(d["tier_counts"].values()) > 0
    assert d["frac_pairs_rejected_fails_a_written_number_test"] > 0.5      # the pairs carry the written-number signal
    committed = json.loads((ROOT / "data" / "dpo" / "ambigous_spec_pairs_seed0.meta.json").read_text())
    assert committed["pairs_sha256"] == meta["pairs_sha256"]
    assert dpo_train.check_pairs_against_meta(meta, 0, "ambigous_spec") == "matches committed meta"
    with pytest.raises(RuntimeError):
        dpo_train.check_pairs_against_meta({**meta, "pairs_sha256": "0" * 64}, 0, "ambigous_spec")


def test_dpo_plan_records_and_cpu_only_dry_run():
    cfg = dpo_train.load_config(ROOT / "trainer" / "configs" / "dpo_v4.json")
    assert dpo_train.plan(cfg, 468)["planned_optimizer_steps"] == 118
    assert dpo_train.sft_stage_config(cfg)["experiment"] == "phase2_sft"      # cannot collide with Phase 1 run ids
    recs = dpo_train.build_pair_records(
        [{"chosen": "a", "rejected": "b"}], "ambigous_spec")
    assert recs[0]["prompt"].startswith("<|im_start|>user\n") and recs[0]["prompt"].endswith("<|im_start|>assistant\n")
    from colab import run_phase2
    assert [c["seed"] for c in run_phase2.matrix(cfg)] == [0, 1, 2]
    p = subprocess.run([sys.executable, "-c", NO_TORCH], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "118 DPO optimizer steps" in p.stdout
