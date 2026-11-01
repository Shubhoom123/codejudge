"""The results ledger is the only place a headline number may come from.

These tests don't re-measure anything (that needs the gitignored shards and
saved runs). They check that what is *committed* is self-consistent: the
README and docs/RESULTS.md are renderings of results_ledger.json, every
entry carries its provenance, and the test count in the ledger is current.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "tools"))
import build_ledger as bl  # noqa: E402


@pytest.fixture(scope="module")
def ledger():
    return json.loads(bl.LEDGER_PATH.read_text())


def test_every_entry_records_its_provenance(ledger):
    required = {"value", "kind", "source", "command", "commit", "dirty", "date", "seed", "model_tag"}
    for key, entry in ledger["entries"].items():
        assert required <= entry.keys(), f"{key} is missing {required - entry.keys()}"
        assert entry["kind"] in bl.KINDS, key


def test_unverified_entries_carry_no_value(ledger):
    for key, entry in ledger["entries"].items():
        if entry["kind"] == "unverified":
            assert entry["value"] is None, f"{key}: an unverified claim must not store a number"


def test_readme_and_results_md_are_rendered_from_the_ledger():
    assert bl.check() == []


def test_readme_test_counts_are_ledger_spans_not_typed_by_hand():
    text = bl.README.read_text()
    outside_spans = bl.SPAN_RE.sub("", text)
    assert not re.search(r"\b\d+ tests?\b", outside_spans), \
        "a hand-typed test count is in the README; use <!--ledger:tests.collected-->"


def test_ledger_test_count_matches_what_pytest_collects(ledger):
    collected, _ = bl.collect_tests()
    assert collected == ledger["entries"]["tests.collected"]["value"], \
        "tests were added or removed: re-run python tools/build_ledger.py"


def test_replace_spans_rewrites_only_the_span():
    text = "We have <!--ledger:x-->OLD<!--/ledger--> tests."
    new, used = bl.replace_spans(text, {}, {"x": lambda E: "7"})
    assert new == "We have <!--ledger:x-->7<!--/ledger--> tests."
    assert used == ["x"]


def test_replace_spans_rejects_ids_that_are_not_in_the_ledger():
    with pytest.raises(KeyError):
        bl.replace_spans("<!--ledger:nope-->1<!--/ledger-->", {}, {})


def test_cli_check_mode_exits_zero_on_committed_files():
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "build_ledger.py"), "--check"],
                       capture_output=True, text=True, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stdout + p.stderr


def test_fast_reservoir_matches_the_reference_sampler_exactly(tmp_path):
    from test_case_store import ShardWriter, reservoir_sample
    w = ShardWriter(tmp_path, "t", shard_size=17)
    for i in range(200):
        w.write({"input": str(i), "expected": float(i), "category": "c"})
    w.close()
    for seed, k in [(0, 5), (42, 30), (7, 200), (3, 500)]:
        assert bl.reservoir_cases(tmp_path, k, seed) == reservoir_sample(tmp_path, k, seed)
