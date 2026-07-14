from trainer.reward import build_training_prompt, score_to_reward


def test_perfect_score_gets_full_reward():
    assert score_to_reward(1.0) == 1.0


def test_high_score_gets_partial_reward():
    assert score_to_reward(0.85) == 0.5
    assert score_to_reward(0.8) == 0.5


def test_low_score_gets_zero_reward():
    assert score_to_reward(0.79) == 0.0
    assert score_to_reward(0.0) == 0.0


def test_build_training_prompt_shape():
    p = build_training_prompt("  do X  ", "  code  ")
    assert set(p.keys()) == {"instruction", "input", "output"}
    assert p["input"] == "do X"
    assert p["output"] == "code"
    assert "Python code generator" in p["instruction"]
