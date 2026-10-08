import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("k6_example_inputs_test", Path(__file__).parents[1] / "example_inputs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_copy_and_cleanup_failure_cannot_block_plugin_loading(tmp_path, monkeypatch, caplog):
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.png").write_bytes(b"image")
    monkeypatch.setattr(module, "EXAMPLE_INPUTS", source)

    def copy(source_file, target_file):
        target_file.write(b"PART")
        raise OSError("copy failed")

    def unlink(path, **kwargs):
        raise PermissionError("file is locked")

    monkeypatch.setattr(module.shutil, "copyfileobj", copy)
    monkeypatch.setattr(Path, "unlink", unlink)
    assert module.install_example_inputs(tmp_path / "input") == ()
    assert "Unable to remove incomplete" in caplog.text and "copy failed" in caplog.text


def test_example_install_preserves_user_file(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.png").write_bytes(b"example")
    monkeypatch.setattr(module, "EXAMPLE_INPUTS", source)
    target = tmp_path / "input/kandinsky6/example.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user content")
    assert module.install_example_inputs(tmp_path / "input") == ()
    assert target.read_bytes() == b"user content"
