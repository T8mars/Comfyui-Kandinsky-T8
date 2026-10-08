import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("k6_cleanup_test", Path(__file__).parents[1] / "kandinsky6/runtime_cleanup.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("failed", [None, "prefetch", "model"])
def test_both_cleanups_run_and_cleanup_failure_is_reported(failed):
    calls = []

    def cleanup(name):
        calls.append(name)
        if failed == name:
            raise RuntimeError(name)

    model = SimpleNamespace(cleanup=lambda: cleanup("model"))
    if failed:
        with pytest.raises(RuntimeError, match=failed):
            module.cleanup_sampling(model, lambda: cleanup("prefetch"))
    else:
        module.cleanup_sampling(model, lambda: cleanup("prefetch"))
    assert calls == ["prefetch", "model"]


def test_inference_failure_survives_both_cleanup_failures():
    calls = []

    def cleanup(name):
        calls.append(name)
        raise RuntimeError(name)

    original = ValueError("original inference failure")
    model = SimpleNamespace(cleanup=lambda: cleanup("model"))
    with pytest.raises(ValueError) as result:
        try:
            raise original
        finally:
            module.cleanup_sampling(model, lambda: cleanup("prefetch"))
    assert result.value is original
    assert calls == ["prefetch", "model"]
