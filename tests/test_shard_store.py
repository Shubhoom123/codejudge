"""Tests for test_case_store.py — the sharded read/write layer that lets
the corpus scale to millions of cases without one giant JSON file."""
import json

from test_case_store import (
    ShardWriter,
    count_cases,
    iter_shards,
    load_manifest,
    reservoir_sample,
    write_manifest,
)


def test_shard_writer_splits_into_correct_shard_sizes(tmp_path):
    writer = ShardWriter(tmp_path, shard_prefix="t", shard_size=10)
    for i in range(25):
        writer.write({"input": str(i), "expected": float(i), "category": "test"})
    writer.close()

    shard_files = sorted(tmp_path.glob("shard_t_*.jsonl"))
    assert len(shard_files) == 3  # 10 + 10 + 5
    assert writer.total_written == 25


def test_iter_shards_reads_back_everything_in_order(tmp_path):
    writer = ShardWriter(tmp_path, shard_prefix="t", shard_size=5)
    for i in range(12):
        writer.write({"input": str(i), "expected": float(i), "category": "test"})
    writer.close()

    cases = list(iter_shards(tmp_path))
    assert len(cases) == 12
    assert [c["input"] for c in cases] == [str(i) for i in range(12)]


def test_iter_shards_skips_blank_lines(tmp_path):
    (tmp_path / "shard_x_00000.jsonl").write_text(
        json.dumps({"input": "a", "expected": 1.0, "category": "c"}) + "\n\n"
    )
    cases = list(iter_shards(tmp_path))
    assert len(cases) == 1


def test_reservoir_sample_returns_exactly_k_when_corpus_is_larger(tmp_path):
    writer = ShardWriter(tmp_path, shard_prefix="t", shard_size=50)
    for i in range(500):
        writer.write({"input": str(i), "expected": float(i), "category": "test"})
    writer.close()

    sample = reservoir_sample(tmp_path, k=30, seed=1)
    assert len(sample) == 30
    # every sampled case actually came from the corpus
    inputs = {c["input"] for c in sample}
    assert inputs.issubset({str(i) for i in range(500)})


def test_reservoir_sample_is_deterministic_for_a_fixed_seed(tmp_path):
    writer = ShardWriter(tmp_path, shard_prefix="t", shard_size=50)
    for i in range(200):
        writer.write({"input": str(i), "expected": float(i), "category": "test"})
    writer.close()

    a = reservoir_sample(tmp_path, k=10, seed=7)
    b = reservoir_sample(tmp_path, k=10, seed=7)
    assert a == b


def test_reservoir_sample_handles_corpus_smaller_than_k(tmp_path):
    writer = ShardWriter(tmp_path, shard_prefix="t", shard_size=50)
    for i in range(5):
        writer.write({"input": str(i), "expected": float(i), "category": "test"})
    writer.close()

    sample = reservoir_sample(tmp_path, k=100, seed=1)
    assert len(sample) == 5


def test_manifest_roundtrip(tmp_path):
    write_manifest(tmp_path, {"total_cases": 42, "note": "hi"})
    assert load_manifest(tmp_path) == {"total_cases": 42, "note": "hi"}


def test_load_manifest_missing_returns_empty_dict(tmp_path):
    assert load_manifest(tmp_path) == {}


def test_count_cases_prefers_manifest_over_scanning(tmp_path):
    write_manifest(tmp_path, {"total_cases": 999})
    # No actual shard files exist — if count_cases scanned instead of
    # trusting the manifest, this would return 0.
    assert count_cases(tmp_path) == 999


def test_count_cases_falls_back_to_scanning_without_manifest(tmp_path):
    writer = ShardWriter(tmp_path, shard_prefix="t", shard_size=50)
    for i in range(7):
        writer.write({"input": str(i), "expected": float(i), "category": "test"})
    writer.close()
    assert count_cases(tmp_path) == 7
