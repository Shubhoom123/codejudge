"""Tests for the FastAPI wrapper that don't require a live Ollama server.

/evaluate itself isn't exercised end-to-end here (it needs a real model
backend) — these cover routing, validation, and the path-traversal guard
on /results/{filename}, which is the part most worth regression-testing.
"""
import json

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_tasks_matches_config():
    r = client.get("/tasks")
    assert r.status_code == 200
    assert set(r.json()) == set(config.list_tasks())


def test_get_known_task():
    r = client.get("/tasks/ambigous_spec")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "ambigous_spec"
    assert "parse_amount" in body["prompt"]


def test_get_unknown_task_404s():
    r = client.get("/tasks/does_not_exist")
    assert r.status_code == 404


def test_evaluate_unknown_task_404s_before_touching_ollama():
    r = client.post("/evaluate", json={"model": "gemma3:12b", "task": "does_not_exist"})
    assert r.status_code == 404


def test_evaluate_unknown_model_400s():
    # resolve_model will try to hit Ollama's /api/tags to auto-detect;
    # if nothing is listening it fails closed with a 400, not a crash.
    r = client.post("/evaluate", json={"model": "definitely-not-a-real-model", "task": "ambigous_spec"})
    assert r.status_code == 400


@pytest.mark.parametrize("bad_filename", [
    "../../etc/passwd",
    "..%2Fetc%2Fpasswd",
    "not_a_run_file.json",
    "run_123.txt",
    "run_123/../../secret.json",
])
def test_results_path_traversal_is_rejected(bad_filename):
    r = client.get(f"/results/{bad_filename}")
    assert r.status_code in (400, 404)


def test_results_latest_with_no_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "empty_results")
    import api.main as api_main
    monkeypatch.setattr(api_main.config, "RESULTS_DIR", tmp_path / "empty_results")
    r = client.get("/results/latest")
    assert r.status_code == 404


def test_results_roundtrip(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    payload = [{"task": "ambigous_spec", "models": {}}]
    (results_dir / "run_20260101_000000.json").write_text(json.dumps(payload))

    import api.main as api_main
    monkeypatch.setattr(api_main.config, "RESULTS_DIR", results_dir)

    listing = client.get("/results")
    assert listing.json() == ["run_20260101_000000.json"]

    detail = client.get("/results/run_20260101_000000.json")
    assert detail.status_code == 200
    assert detail.json() == payload
