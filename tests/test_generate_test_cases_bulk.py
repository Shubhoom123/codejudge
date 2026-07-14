"""Tests for tools/generate_test_cases_bulk.py.

These run at a small scale (hundreds of integers, not millions) — the
point isn't to re-verify throughput here, it's to catch correctness
regressions in the number_to_words <-> parse_amount round-trip and the
manifest bookkeeping, which is exactly the kind of thing that silently
breaks during a refactor and corrupts every case generated after it.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "tools"))
import generate_test_cases_bulk as gen  # noqa: E402

from test_case_store import iter_shards, load_manifest


def test_number_to_words_round_trips_through_reference_parser():
    for n in [0, 1, 7, 19, 20, 42, 99, 100, 101, 500, 999, 1000, 1001, 45000, 999999, 1000000, 2500000]:
        words = gen.number_to_words(n)
        assert gen.parse_amount(words) == float(n), f"{n} -> {words!r} didn't round-trip"


def test_number_to_words_zero():
    assert gen.number_to_words(0) == "zero"


def test_fixed_edge_cases_are_validated_not_assumed():
    """Regression test for a real bug: "$$100" was hardcoded as an
    "invalid" candidate, but the reference solution just strips both "$"
    characters and parses "100" fine — it never raises. Every fixed
    edge case must be validated against the reference at generation
    time, not just asserted by whoever wrote the candidate list."""
    cases = gen.build_fixed_edge_cases()
    texts = {text for text, _, _ in cases}
    assert "$$100" not in texts

    for text, expected, category in cases:
        if category == "invalid":
            try:
                gen.parse_amount(text)
                assert False, f"{text!r} was kept as invalid but doesn't raise"
            except ValueError:
                pass
        else:
            assert gen.parse_amount(text) == expected


def test_generate_chunk_only_emits_round_tripped_written_numbers(tmp_path):
    result = gen.generate_chunk((0, 0, 200, str(tmp_path), 1000))
    assert result["by_category"]["written_number"] == 200

    cases = [c for c in iter_shards(tmp_path) if c["category"] == "written_number"]
    for case in cases:
        assert gen.parse_amount(case["input"]) == case["expected"]


def test_generate_chunk_writes_expected_case_shape(tmp_path):
    gen.generate_chunk((0, 0, 50, str(tmp_path), 1000))
    cases = list(iter_shards(tmp_path))
    assert len(cases) > 0
    for case in cases:
        assert set(case.keys()) == {"input", "expected", "category"}


def test_full_generation_small_scale_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["generate_test_cases_bulk.py", "--target", "300", "--workers", "1", "--out-dir", str(tmp_path)],
    )
    gen.main()

    manifest = load_manifest(tmp_path)
    assert manifest["total_cases"] > 0
    assert manifest["target_integers"] == 300

    cases = list(iter_shards(tmp_path))
    assert len(cases) == manifest["total_cases"]

    # manifest category counts must match what's actually on disk
    actual_by_category = {}
    for c in cases:
        actual_by_category[c["category"]] = actual_by_category.get(c["category"], 0) + 1
    assert actual_by_category == manifest["by_category"]

    # every non-"raises_ValueError" case must actually parse to its
    # claimed expected value under the reference solution
    for c in cases:
        if c["expected"] != "raises_ValueError":
            assert gen.parse_amount(c["input"]) == c["expected"], c
        else:
            try:
                gen.parse_amount(c["input"])
                assert False, f"expected ValueError for {c['input']!r}"
            except ValueError:
                pass
