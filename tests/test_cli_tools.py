"""CLI regressions reproduce damaged partials and cancelled queue entries."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(f"k6_test_{name}", ROOT / "tools" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_corrupt_complete_partial_recovers_on_next_attempt(tmp_path, monkeypatch):
    downloader = module("download_models")
    item = {"destination": "test.bin", "size": 4, "sha256": hashlib.sha256(b"GOOD").hexdigest(),
            "repo": "test/model", "revision": "a" * 40, "filename": "test.bin"}
    calls = []

    def fetch(request, timeout):
        calls.append(request)
        return io.BytesIO(b"BAD!" if len(calls) == 1 else b"GOOD")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fetch)
    with pytest.raises(ValueError, match="verification"):
        downloader.download(item, tmp_path)
    assert not (tmp_path / "test.bin").exists()
    downloader.download(item, tmp_path)
    assert len(calls) == 2
    assert (tmp_path / "test.bin").read_bytes() == b"GOOD"


def test_bad_preexisting_partial_is_restarted_without_deleting_installed_model(tmp_path, monkeypatch):
    downloader = module("download_models")
    item = {"destination": "test.bin", "size": 4, "sha256": hashlib.sha256(b"GOOD").hexdigest(),
            "repo": "test/model", "revision": "a" * 40, "filename": "test.bin"}
    (tmp_path / "test.bin.part").write_bytes(b"BAD!")
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b"GOOD"))
    downloader.download(item, tmp_path)
    assert (tmp_path / "test.bin").read_bytes() == b"GOOD"
    (tmp_path / "test.bin").write_bytes(b"BAD!")
    with pytest.raises(ValueError, match="Existing file"):
        downloader.download(item, tmp_path)
    assert (tmp_path / "test.bin").read_bytes() == b"BAD!"


@pytest.mark.parametrize("finish_race", [False, True])
def test_removed_prompt_exits_and_completion_race_is_preserved(tmp_path, monkeypatch, finish_race):
    runner = module("run_workflow")
    workflow = tmp_path / "workflow.json"
    receipt = tmp_path / "receipt.json"
    workflow.write_text("{}")
    checks = []
    history_calls = 0

    def request(url, data=None):
        nonlocal history_calls
        checks.append(url)
        if url.endswith("/prompt"):
            return {"prompt_id": "removed"}
        if url.endswith("/queue"):
            return {"queue_running": [], "queue_pending": []}
        history_calls += 1
        if finish_race and history_calls >= 4:
            return {"removed": {"status": {"status_str": "success", "completed": True}, "outputs": {}}}
        return {}

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(sys, "argv", ["run_workflow", str(workflow), "--receipt", str(receipt)])
    if finish_race:
        runner.main()
    else:
        with pytest.raises(RuntimeError, match="cancelled"):
            runner.main()
    result = json.loads(receipt.read_text())
    assert result["history"]["status"]["status_str"] == ("success" if finish_race else "cancelled")
    assert sum(url.endswith("/queue") for url in checks) == 3
