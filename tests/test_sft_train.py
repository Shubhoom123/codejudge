import json
import subprocess
import sys
from pathlib import Path

import pytest

import config
from trainer import sft_train as st

ROOT = Path(__file__).parent.parent
CFG_PATH = ROOT / "trainer" / "configs" / "sft_v4.json"


@pytest.fixture(scope="module")
def cfg():
    return st.load_config(CFG_PATH)


@pytest.fixture(scope="module")
def perfect(cfg):
    return st.load_perfect_examples(cfg)


def test_perfect_examples_match_the_ledger_and_are_all_perfect(perfect):
    ledger = json.loads((ROOT / "results_ledger.json").read_text())
    assert len(perfect) == ledger["entries"]["training.perfect"]["value"]
    assert all(e["score"] == 1.0 for e in perfect)


def test_subsets_are_nested_for_a_fixed_seed_and_differ_across_seeds(perfect):
    s20, s50, s117 = (st.select_examples(perfect, n, 0) for n in (20, 50, 117))
    ids = lambda xs: [e["id"] for e in xs]  # noqa: E731
    assert ids(s20) == ids(s50)[:20] and ids(s50) == ids(s117)[:50]
    assert ids(st.select_examples(perfect, 20, 0)) == ids(s20)
    assert ids(st.select_examples(perfect, 20, 1)) != ids(s20)
    assert set(ids(s117)) == {e["id"] for e in perfect}


def test_select_rejects_impossible_sizes(perfect):
    for n in (0, len(perfect) + 1):
        with pytest.raises(ValueError):
            st.select_examples(perfect, n, 0)


def test_data_hash_identifies_the_subset_regardless_of_order(perfect):
    a = st.select_examples(perfect, 20, 0)
    assert st.data_sha256(a) == st.data_sha256(list(reversed(a)))
    assert st.data_sha256(a) != st.data_sha256(st.select_examples(perfect, 20, 1))


def test_training_prompt_is_exactly_what_the_evaluators_send(cfg, perfect):
    # regression guard for README bug #2: training and evaluation building "the prompt" separately
    recs = st.build_records(st.select_examples(perfect, 20, 0), cfg["task"])
    expected = config.format_chatml(config.build_eval_prompt(cfg["task"]))
    assert {r["prompt"] for r in recs} == {expected}
    assert config.load_task_prompt(cfg["task"]) in expected and expected.startswith("<|im_start|>user\n")
    assert expected.endswith("<|im_end|>\n<|im_start|>assistant\n")
    assert config.load_task_hint(cfg["task"]) not in expected  # hint.md is teacher-only
    assert all("```" not in r["completion"] for r in recs)


def test_eval_prompt_is_the_evaluators_prompt():
    assert config.build_eval_prompt("ambigous_spec") == (
        config.CODE_SYSTEM_PROMPT + "\n\n" + config.load_task_prompt("ambigous_spec"))


def test_step_planning_for_both_arms(cfg):
    assert st.steps_per_epoch(20, 2, 4) == 3 and st.steps_per_epoch(117, 2, 4) == 15
    assert st.plan_epochs(20, cfg, "scaling", None) == 3
    assert st.plan_epochs(20, cfg, "steps_matched", 45) == 15
    with pytest.raises(ValueError):
        st.plan_epochs(20, cfg, "steps_matched", None)


def test_plan_reports_data_facts_that_matter_for_diagnosis(cfg):
    plan = st.make_plan(cfg, 20, 0, "scaling", None)
    d = plan["data_diagnostics"]
    assert plan["planned_optimizer_steps"] == 9 and plan["effective_batch"] == 8
    assert d["n_unique_prompts"] == 1 and d["n_unique_completions"] == 20
    assert d["n_mentioning_written_number_words"] == 20  # the teacher data does contain the parsing logic


def test_config_hash_tracks_the_recipe_but_not_output_locations(cfg):
    base = st.config_sha256(cfg)
    moved = st.load_config(CFG_PATH, {"mlflow.tracking_uri": "file:/elsewhere", "runs_dir": "other"})
    tuned = st.load_config(CFG_PATH, {"train.learning_rate": 1e-3})
    assert st.config_sha256(moved) == base and st.config_sha256(tuned) != base


def test_config_validation(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({"experiment": "x"}))
    with pytest.raises(ValueError):
        st.load_config(bad)
    with pytest.raises(ValueError):
        st.load_config(CFG_PATH, {"backend": "tensorflow"})
    # Colab's current MLflow raises on the file store ("maintenance mode") before training starts
    assert st.load_config(CFG_PATH)["mlflow"]["tracking_uri"].startswith("sqlite:")


def test_importing_the_trainer_does_not_need_torch_and_dry_run_works_without_a_gpu():
    code = ("import sys; sys.path.insert(0, %r); import trainer.sft_train as s; "
            "assert 'torch' not in sys.modules and 'trl' not in sys.modules; "
            "raise SystemExit(s.main(['--n','20','--seed','0','--dry-run']))" % str(ROOT))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "9 optimizer steps" in p.stdout


def test_colab_matrix_covers_the_requested_ablation(cfg):
    from colab import run_phase1
    cells = run_phase1.matrix(cfg)
    scaling = [c for c in cells if c["arm"] == "scaling"]
    assert {(c["n"], c["seed"]) for c in scaling} == {(n, s) for n in (20, 50, 117) for s in (0, 1, 2)}
    assert all(c["target"] == 45 for c in cells if c["arm"] == "steps_matched")
