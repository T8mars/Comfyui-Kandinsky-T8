"""Download integrity checks use small local fixtures, never network weights."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("k6_model_downloads_test", ROOT / "model_downloads.py")
downloads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloads)


def entry(data=b"valid model"):
    return {"directory": "text_encoders", "name": "test.safetensors", "repo_id": "test/model",
            "revision": "a" * 40, "filename": "test.safetensors", "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def test_truncated_existing_file_is_repaired(tmp_path, monkeypatch):
    model = entry()
    target = tmp_path / model["directory"] / model["name"]
    target.parent.mkdir()
    target.write_bytes(b"v")
    assert downloads.existing_model(model, tmp_path, {}) is None
    cached = tmp_path / "cached"
    cached.write_bytes(b"valid model")
    monkeypatch.setattr(downloads, "hf_hub_download", lambda **kwargs: cached)
    assert downloads.download_model(model, tmp_path, {}) == "downloaded"
    assert target.read_bytes() == b"valid model"


def test_equal_size_corruption_and_invalidated_hash_cache(tmp_path, monkeypatch):
    model = entry()
    target = tmp_path / model["directory"] / model["name"]
    target.parent.mkdir()
    target.write_bytes(b"valid model")
    monkeypatch.setattr(downloads, "hf_hub_download", lambda **kwargs: pytest.fail("valid file must be reused"))
    assert downloads.download_model(model, tmp_path, {}) == "reused"
    target.write_bytes(b"wrong model")
    assert not downloads.verified_model(target, model)
    cached = tmp_path / "cached"
    cached.write_bytes(b"valid model")
    monkeypatch.setattr(downloads, "hf_hub_download", lambda **kwargs: cached)
    assert downloads.download_model(model, tmp_path, {}) == "downloaded"


def test_corrupt_hub_cache_is_forced_once_and_never_installed(tmp_path, monkeypatch):
    model = entry()
    cached = tmp_path / "cached"
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs["force_download"])
        cached.write_bytes(b"wrong model")
        return cached

    monkeypatch.setattr(downloads, "hf_hub_download", fetch)
    with pytest.raises(ValueError, match="verification"):
        downloads.download_model(model, tmp_path, {})
    assert calls == [False, True]
    assert not (tmp_path / model["directory"] / model["name"]).exists()


def test_valid_extra_model_is_reused(tmp_path, monkeypatch):
    extra = tmp_path / "external"
    extra.mkdir()
    model = entry()
    (extra / model["name"]).write_bytes(b"valid model")
    monkeypatch.setattr(downloads, "hf_hub_download", lambda **kwargs: pytest.fail("extra model must be reused"))
    assert downloads.download_model(model, tmp_path / "models", {"text_encoders": [extra]}) == "reused"


def test_shipped_manifest_has_verified_pinned_components():
    plan = downloads.model_plan(ROOT / "example_workflows")
    assert len(plan) == 6
    sources = json.loads((ROOT / "MODEL_SOURCES.json").read_text(encoding="utf-8"))
    for model in plan:
        source = next(item for item in sources if (item["repo"], item["revision"], item["filename"])
                      == (model["repo_id"], model["revision"], model["filename"]))
        assert model["size"] == source["size"]
        assert model["sha256"] == source["actual_sha256"]
