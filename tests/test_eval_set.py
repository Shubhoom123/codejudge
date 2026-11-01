import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "tools"))
import make_eval_set as mes  # noqa: E402

import evaluation as ev  # noqa: E402
from test_case_store import ShardWriter, write_manifest  # noqa: E402

REAL = {"written_number": 14172601, "whitespace": 5451000, "negative": 2180399, "dollar_sign": 2024658,
        "word_dollars": 2024658, "plain_number": 2024658, "comma_thousands": 1288328, "decimal": 833683,
        "invalid": 9, "zero": 6}


def test_allocation_hits_the_budget_and_honours_the_floor():
    a = mes.allocate(REAL, size=2000, floor=100)
    assert sum(a.values()) == 2000
    assert all(a[c] >= 100 for c in REAL if REAL[c] >= 100)


def test_tiny_categories_are_taken_whole_not_padded():
    a = mes.allocate(REAL, size=2000, floor=100)
    assert a["invalid"] == 9 and a["zero"] == 6


def test_allocation_never_exceeds_a_category_and_stays_roughly_proportional_above_the_floor():
    a = mes.allocate(REAL, size=2000, floor=100)
    assert all(a[c] <= REAL[c] for c in REAL)
    assert a["written_number"] > a["whitespace"] > a["negative"] > a["decimal"]


def test_allocation_rejects_impossible_requests():
    with pytest.raises(ValueError):
        mes.allocate({"a": 5, "b": 5}, size=11, floor=1)
    with pytest.raises(ValueError):
        mes.allocate({"a": 100, "b": 100}, size=10, floor=20)


def _tiny_corpus(tmp_path):
    w = ShardWriter(tmp_path, "t", shard_size=40)
    counts = {"big": 200, "mid": 60, "tiny": 3}
    for cat, n in counts.items():
        for i in range(n):
            w.write({"input": f"{cat}-{i}", "expected": float(i), "category": cat})
    w.close()
    write_manifest(tmp_path, {"total_cases": sum(counts.values()), "by_category": counts})
    return counts


def test_build_is_deterministic_and_respects_allocation(tmp_path):
    counts = _tiny_corpus(tmp_path)
    c1, m1 = mes.build(tmp_path, size=40, floor=5, seed=7)
    c2, m2 = mes.build(tmp_path, size=40, floor=5, seed=7)
    c3, _ = mes.build(tmp_path, size=40, floor=5, seed=8)
    assert c1 == c2 and m1["cases_sha256"] == m2["cases_sha256"]
    assert [c["input"] for c in c1] != [c["input"] for c in c3]
    got = {}
    for c in c1:
        got[c["category"]] = got.get(c["category"], 0) + 1
    assert got == m1["allocation"] and got["tiny"] == 3
    assert len({c["input"] for c in c1}) == len(c1)  # without replacement
    assert all(c["input"].startswith(c["category"]) for c in c1)
    assert set(m1["corpus_counts"]) == set(counts)


def test_build_fails_loudly_when_shards_disagree_with_the_manifest(tmp_path):
    _tiny_corpus(tmp_path)
    write_manifest(tmp_path, {"total_cases": 263, "by_category": {"big": 200, "mid": 61, "tiny": 2}})
    with pytest.raises(ValueError):
        mes.build(tmp_path, size=40, floor=5, seed=7)


def test_committed_eval_set_matches_its_metadata_and_the_corpus_manifest():
    path = ROOT / "data" / "eval_sets" / "ambigous_spec_strat2000_seed42.jsonl"
    cases, meta = ev.load_eval_set(path)  # also verifies the sha256
    assert len(cases) == meta["size"] == 2000
    assert len({c["id"] for c in cases}) == 2000
    manifest = json.loads((ROOT / "data" / "test_cases_bulk" / "manifest.json").read_text())
    assert meta["corpus_counts"] == manifest["by_category"]
    got = {}
    for c in cases:
        got[c["category"]] = got.get(c["category"], 0) + 1
    assert got == meta["allocation"]
    assert all(got[c] >= min(meta["floor"], n) for c, n in meta["corpus_counts"].items())


def test_tampering_with_the_eval_set_is_detected(tmp_path):
    src = ROOT / "data" / "eval_sets" / "ambigous_spec_strat2000_seed42.jsonl"
    dst = tmp_path / "set.jsonl"
    dst.write_bytes(src.read_bytes() + b"\n")
    dst.with_suffix(".meta.json").write_bytes(src.with_suffix(".meta.json").read_bytes())
    with pytest.raises(ValueError):
        ev.load_eval_set(dst)
