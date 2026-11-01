import json

import pytest

import evaluation as ev

COUNTS = {"a": 90, "b": 10}


def _res(cat, passed, n):
    return [{"id": i, "category": cat, "passed": passed, "detail": ""} for i in range(n)]


def test_weighted_accuracy_uses_corpus_shares_not_sample_shares():
    # sample is 50/50, corpus is 90/10: a passes, b fails -> corpus accuracy is 0.9, not 0.5
    s = ev.summarize(_res("a", True, 50) + _res("b", False, 50), COUNTS, n_boot=50)
    assert s["weighted"]["accuracy"] == 0.9
    assert s["unweighted"]["accuracy"] == 0.5


def test_all_or_nothing_categories_give_degenerate_intervals():
    # documents the limitation in evaluation.py's docstring: one deterministic solution
    s = ev.summarize(_res("a", True, 20) + _res("b", False, 20), COUNTS, n_boot=100)
    assert s["per_category"]["a"]["ci95"] == [1.0, 1.0]
    assert s["weighted"]["ci95"] == [0.9, 0.9]


def test_mixed_category_gets_a_real_interval_that_contains_the_estimate_and_is_seeded():
    results = _res("a", True, 70) + _res("a", False, 30) + _res("b", True, 10)
    s1 = ev.summarize(results, COUNTS, n_boot=500, seed=3)
    s2 = ev.summarize(results, COUNTS, n_boot=500, seed=3)
    lo, hi = s1["per_category"]["a"]["ci95"]
    assert s1 == s2
    assert lo < 0.7 < hi and 0.55 < lo and hi < 0.85


def test_diagnose_flags_fences_truncation_and_missing_parsing_logic():
    code = "def parse_amount(text):\n    return float(text)"
    d = ev.diagnose_generation("```python\n" + code + "\n```", code, n_new_tokens=1536, max_new_tokens=1536,
                               finish_reason="length")
    assert d["has_markdown_fence"] and d["cleaning_changed_code"] and d["parses"] and d["defines_parse_amount"]
    assert d["hit_max_new_tokens"] is True
    assert d["mentions_written_number_words"] is False
    ok = ev.diagnose_generation("x = {'thousand': 1000}", "x = {'thousand': 1000}", n_new_tokens=10, max_new_tokens=100)
    assert ok["mentions_written_number_words"] and ok["hit_max_new_tokens"] is False
    assert ev.diagnose_generation("def (", "def (")["parses"] is False


def test_loss_summary_reports_whether_loss_went_down():
    down = [{"loss": 2.0}, {"loss": 1.5}, {"loss": 1.0}, {"loss": 0.5}]
    flat = [{"loss": 1.0}, {"loss": 1.0}, {"loss": 1.0}, {"loss": 1.0}]
    assert ev.loss_summary(down)["decreased"] is True
    assert ev.loss_summary(flat)["decreased"] is False
    assert ev.loss_summary([])["n_logged"] == 0


def _run(arm, n, seed, acc):
    return {"kind": "sft_run", "arm": arm, "n_examples": n, "seed": seed,
            "metrics": {"eval": {"weighted": {"accuracy": acc, "ci95": [acc, acc]},
                                 "per_category": {"written_number": {"accuracy": acc}}},
                        "hand_grader": {"passed": 20}}}


def test_aggregate_groups_by_arm_and_n_and_reports_spread():
    runs = [_run("scaling", 117, s, a) for s, a in [(0, 0.5), (1, 0.7), (2, 0.6)]] + [_run("scaling", 20, 0, 0.5)]
    agg = ev.aggregate_runs(runs + [{"kind": "baseline"}])
    g = agg["scaling:n117"]
    assert g["seeds"] == [0, 1, 2] and g["weighted_accuracy"]["mean"] == 0.6
    assert g["weighted_accuracy"]["std"] == pytest.approx(0.1)
    assert agg["scaling:n20"]["weighted_accuracy"]["std"] is None
    assert g["per_category"]["written_number"]["min"] == 0.5 and g["per_category"]["written_number"]["max"] == 0.7


def test_ablation_table_renders_rows_and_an_honest_empty_state():
    assert "No v4 training runs" in ev.render_ablation_table({})
    table = ev.render_ablation_table(ev.aggregate_runs([_run("scaling", 20, 0, 0.5)]))
    assert "| scaling | 20 | 1 | 50.0% (std n/a)" in table


def test_score_solution_runs_cases_through_the_sandbox():
    cases = [{"id": 0, "input": "5", "expected": 5.0, "category": "plain"},
             {"id": 1, "input": "abc", "expected": "raises_ValueError", "category": "invalid"},
             {"id": 2, "input": "five", "expected": 5.0, "category": "written"}]
    out = ev.score_solution("def parse_amount(t):\n    return float(t)", cases, workers=2)
    assert {r["category"]: r["passed"] for r in out} == {"plain": True, "invalid": True, "written": False}


def test_eval_set_loader_rejects_a_missing_or_mismatched_hash(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"id": 0}) + "\n")
    p.with_suffix(".meta.json").write_text(json.dumps({"cases_sha256": "0" * 64}))
    with pytest.raises(ValueError):
        ev.load_eval_set(p)
