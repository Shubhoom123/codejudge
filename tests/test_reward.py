from trainer.reward import build_training_prompt, score_to_reward


def test_perfect_score_gets_full_reward_at_default_threshold():
    assert score_to_reward(1.0, min_score=1.0) == 1.0


def test_near_miss_below_default_threshold_gets_excluded():
    # Regression test: this is the exact bug that let a solution failing
    # an entire grader category (e.g. every written-number test, landing
    # at 20/24 = 0.833) into the training set as "high quality." At the
    # strict default (min_score=1.0), anything short of a perfect score
    # must be excluded, not just down-weighted.
    assert score_to_reward(0.96, min_score=1.0) == 0.0
    assert score_to_reward(0.833, min_score=1.0) == 0.0
    assert score_to_reward(0.8, min_score=1.0) == 0.0


def test_min_score_is_configurable_and_actually_gates_inclusion():
    # Lowering --min-score is a deliberate, explicit choice the caller
    # makes, not a silent default -- verify the threshold is real.
    assert score_to_reward(0.85, min_score=0.8) == 0.85
    assert score_to_reward(0.79, min_score=0.8) == 0.0


def test_low_score_gets_zero_reward():
    assert score_to_reward(0.79, min_score=1.0) == 0.0
    assert score_to_reward(0.0, min_score=1.0) == 0.0


def test_reward_scales_with_score_above_threshold():
    # Reward is the score itself now (not a flat 1.0/0.5/0.0 step
    # function), so two perfect solutions and near-perfect ones aren't
    # all treated identically once they clear the bar.
    assert score_to_reward(1.0, min_score=0.9) == 1.0
    assert score_to_reward(0.92, min_score=0.9) == 0.92


def test_build_training_prompt_shape():
    p = build_training_prompt("  do X  ", "  code  ")
    assert set(p.keys()) == {"instruction", "input", "output"}
    assert p["input"] == "do X"
    assert p["output"] == "code"
    assert "Python code generator" in p["instruction"]
