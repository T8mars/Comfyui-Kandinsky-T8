"""CLI regressions reproduce damaged partials and cancelled queue entries."""
import hashlib
from http.client import IncompleteRead
import importlib.util
import io
import json
from pathlib import Path
import sys
import concurrent.futures
import threading

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


def test_oversized_response_and_preexisting_partial_can_recover(tmp_path, monkeypatch):
    downloader = module("download_models")
    item = {"destination": "test.bin", "size": 4, "sha256": hashlib.sha256(b"GOOD").hexdigest(),
            "repo": "test/model", "revision": "a" * 40, "filename": "test.bin"}
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b"TOOLONG"))
    with pytest.raises(ValueError, match="exceeded"):
        downloader.download(item, tmp_path)
    assert not (tmp_path / "test.bin").exists()
    assert (tmp_path / "test.bin.part").stat().st_size == 0
    # Also exercise oversized partials left by older versions.
    (tmp_path / "test.bin.part").write_bytes(b"TOOLONG")
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b"GOOD"))
    downloader.download(item, tmp_path)
    assert (tmp_path / "test.bin").read_bytes() == b"GOOD"


def test_incomplete_chunked_response_uses_existing_retry_policy(tmp_path, monkeypatch):
    downloader = module("download_models")
    item = {"destination": "test.bin", "size": 4, "sha256": hashlib.sha256(b"GOOD").hexdigest(),
            "repo": "test/model", "revision": "a" * 40, "filename": "test.bin"}
    calls = []

    class Broken(io.BytesIO):
        def read(self, size):
            raise IncompleteRead(b"GO", 2)

    def fetch(request, timeout):
        calls.append(request)
        return Broken() if len(calls) == 1 else io.BytesIO(b"GOOD")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fetch)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    downloader.download(item, tmp_path)
    assert len(calls) == 2 and (tmp_path / "test.bin").read_bytes() == b"GOOD"


def test_duplicate_variant_is_downloaded_once(tmp_path, monkeypatch):
    downloader = module("download_models")
    calls = []
    item = {"destination": "sources/lite.safetensors"}
    monkeypatch.setattr(downloader, "describe", lambda *args: item)
    monkeypatch.setattr(downloader, "download", lambda value, root: calls.append(value) or value)
    monkeypatch.setattr(sys, "argv", ["download_models", "--models", str(tmp_path), "--variants", "lite", "lite", "--workers", "2"])
    downloader.main()
    assert calls == [item]
    assert json.loads((tmp_path / "download_receipts.json").read_text()) == [item]


def download_item():
    return {"destination": "test.bin", "size": 4, "sha256": hashlib.sha256(b"GOOD").hexdigest(),
            "repo": "test/model", "revision": "a" * 40, "filename": "test.bin"}


def test_independent_downloads_cannot_share_partial_writer(tmp_path, monkeypatch):
    downloader = module("download_models")
    ready, release = threading.Event(), threading.Event()
    digest = downloader.digest

    def pause_verified_partial(path):
        result = digest(path)
        if path.suffix == ".part":
            ready.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(downloader, "digest", pause_verified_partial)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b"GOOD"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(downloader.download, download_item(), tmp_path)
        try:
            assert ready.wait(3)
            with pytest.raises(RuntimeError, match="already running"):
                downloader.download(download_item(), tmp_path)
        finally:
            release.set()
        receipt = first.result()
    assert digest(tmp_path / "test.bin") == receipt["actual_sha256"] == download_item()["sha256"]


def test_download_final_install_cannot_replace_external_file(tmp_path, monkeypatch):
    downloader = module("download_models")
    target = tmp_path / "test.bin"
    operation = "rename" if downloader.os.name == "nt" else "link"
    install = getattr(downloader.os, operation)

    def create_before_install(temporary, destination):
        if Path(destination) == target:
            thread = threading.Thread(target=lambda: target.write_bytes(b"USER"))
            thread.start()
            thread.join(5)
            assert not thread.is_alive()
        return install(temporary, destination)

    monkeypatch.setattr(downloader.os, operation, create_before_install)
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b"GOOD"))
    with pytest.raises(FileExistsError):
        downloader.download(download_item(), tmp_path)
    assert target.read_bytes() == b"USER"
    assert (tmp_path / "test.bin.part").read_bytes() == b"GOOD"
    assert not (tmp_path / "test.bin.download.json").exists()


def test_failed_receipt_publication_preserves_previous_complete_json(tmp_path, monkeypatch):
    downloader = module("download_models")
    target, receipt = tmp_path / "test.bin", tmp_path / "test.bin.download.json"
    target.write_bytes(b"GOOD")
    receipt.write_text('{"previous":true}\n')
    before = receipt.read_bytes()
    replace = downloader.os.replace

    def deny_receipt(source, destination):
        if Path(destination) == receipt:
            raise PermissionError("receipt publication denied")
        return replace(source, destination)

    monkeypatch.setattr(downloader.os, "replace", deny_receipt)
    with pytest.raises(PermissionError, match="receipt publication denied"):
        downloader.download(download_item(), tmp_path)
    assert receipt.read_bytes() == before and target.read_bytes() == b"GOOD"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("suffix", ["", ".part", ".download.json"])
def test_download_lock_cannot_alias_existing_artifacts(tmp_path, suffix):
    downloader = module("download_models")
    target = tmp_path / "test.bin"
    artifact = Path(str(target) + suffix)
    artifact.write_bytes(b"GOOD")
    lock = Path(str(target) + ".download.lock")
    lock.hardlink_to(artifact)
    before = artifact.read_bytes()
    with pytest.raises(ValueError, match="Download lock must differ"):
        downloader.download(download_item(), tmp_path)
    assert artifact.read_bytes() == before and lock.read_bytes() == before
