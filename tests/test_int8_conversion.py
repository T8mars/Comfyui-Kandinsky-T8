"""Small fixtures exercise the real Kitchen kernels and serialized output."""
import importlib.util
import json
import concurrent.futures
from pathlib import Path
import threading

import pytest
import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
from safetensors.torch import load_file, save_file

spec = importlib.util.spec_from_file_location("convert_int8", Path(__file__).parents[1] / "tools" / "convert_int8.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA ConvRot kernels required")
@pytest.mark.parametrize("group,width", [(64, 896), (256, 1792)])
def test_streamed_weight_reloads_and_runs_linear(tmp_path, monkeypatch, group, width):
    torch.manual_seed(61)
    weight = torch.randn(35, width)
    original = {"block.weight": weight, "modulation.weight": torch.randn(3, 7).bfloat16(),
                "scalar": torch.tensor(3.5), "integer": torch.tensor([1, 2, 3])}
    source, target = tmp_path / "source.safetensors", tmp_path / "int8.safetensors"
    save_file(original, source)
    plan = [{"source": key, "key": key, "shape": list(value.shape),
             "dtype": {torch.float32: "F32", torch.bfloat16: "BF16", torch.int64: "I64"}[value.dtype],
             "group": group if key == "block.weight" else None} for key, value in sorted(original.items())]
    # The fixture substitutes only architecture planning; serialization,
    # chunked quantization, verification and the runtime kernel remain real.
    monkeypatch.setattr(converter, "make_plan", lambda *args: (plan, {}))
    manifest = converter.convert(source, target, rows=13)
    loaded = load_file(target)
    for key in original.keys() - {"block.weight"}:
        assert torch.equal(loaded[key], original[key])
        assert loaded[key].dtype == original[key].dtype
    assert loaded["block.weight"].dtype == torch.int8
    marker = json.loads(bytes(loaded["block.comfy_quant"].tolist()))
    assert marker["convrot_groupsize"] == group
    params = TensorWiseINT8Layout.Params(scale=loaded["block.weight_scale"].cuda(), orig_dtype=torch.float32,
                                        orig_shape=tuple(weight.shape), convrot=True, convrot_groupsize=group)
    quantized = QuantizedTensor(loaded["block.weight"].cuda(), "TensorWiseINT8Layout", params)
    inputs = torch.randn(8, width, device="cuda")
    expected = torch.nn.functional.linear(inputs, weight.cuda())
    actual = torch.nn.functional.linear(inputs, quantized)
    assert torch.isfinite(actual).all()
    assert ((actual - expected).norm() / expected.norm()).item() < 0.025
    assert manifest["quantized_layers"] == 1
    assert converter.file_hash(target) == manifest["output_sha256"]
    with pytest.raises(FileExistsError):
        converter.convert(source, target)


def test_incomplete_model_is_rejected(tmp_path):
    source = tmp_path / "incomplete.safetensors"
    save_file({"visual_transformer_blocks.0.attn.to_query.weight": torch.zeros(16, 16)}, source)
    with pytest.raises(ValueError, match="complete"):
        converter.make_plan(source)


def test_sensitive_layer_selection():
    assert converter.group_size("visual_blocks.0.audioT.feed_forward.out_layer.weight", (896, 2688)) == 64
    assert converter.group_size("visual_blocks.0.videoT.self_attention.to_query.weight", (1792, 1792)) == 256
    for key in ("video_time_embeddings.in_layer.weight", "out_layer.out_layer.weight",
                "visual_blocks.0.videoT.modulation.out_layer.weight", "visual_embeddings.in_layer.weight"):
        assert converter.group_size(key, (1792, 1792)) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA ConvRot kernels required")
def test_source_cannot_collide_with_temporary_output(tmp_path, monkeypatch):
    source = tmp_path / "output.safetensors.tmp"
    target = tmp_path / "output.safetensors"
    original = torch.randn(3, 64)
    save_file({"block.weight": original}, source)
    before = source.read_bytes()
    monkeypatch.setattr(converter, "make_plan", lambda *args: (
        [{"source": "block.weight", "key": "block.weight", "shape": [3, 64], "dtype": "F32", "group": 64}], {}))
    converter.convert(source, target, rows=2)
    assert source.read_bytes() == before
    assert target.is_file()
    assert not list(tmp_path.glob("output.safetensors.*.tmp"))


@pytest.mark.parametrize("rows", [0, -1, 1.5])
def test_invalid_chunk_size_fails_before_writing(tmp_path, rows):
    target = tmp_path / "output.safetensors"
    target.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="positive integer"):
        converter.convert(tmp_path / "source.safetensors", target, rows=rows, overwrite=True)
    assert target.read_bytes() == b"existing output"
    assert list(tmp_path.iterdir()) == [target]


def tiny_plan(*args):
    return ([{"source": "weight", "key": "weight", "shape": [2, 2], "dtype": "F32", "group": None}], {})


def test_concurrent_conversion_cannot_overwrite_default_target(tmp_path, monkeypatch):
    source_a, source_b, target = (tmp_path / name for name in ("a.safetensors", "b.safetensors", "out.safetensors"))
    save_file({"weight": torch.zeros(2, 2)}, source_a)
    save_file({"weight": torch.ones(2, 2)}, source_b)
    monkeypatch.setattr(converter, "make_plan", tiny_plan)
    ready, release = threading.Event(), threading.Event()
    original_verify = converter.verify

    def verify(*args):
        original_verify(*args)
        ready.set()
        assert release.wait(5)

    monkeypatch.setattr(converter, "verify", verify)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(converter.convert, source_a, target)
        try:
            assert ready.wait(3)
            with pytest.raises(RuntimeError, match="already running"):
                converter.convert(source_b, target)
        finally:
            release.set()
        manifest = first.result()
    assert torch.equal(load_file(target)["weight"], torch.zeros(2, 2))
    assert converter.file_hash(target) == manifest["output_sha256"]


@pytest.mark.parametrize("existing", [False, True])
def test_manifest_commit_failure_rolls_back_checkpoint(tmp_path, monkeypatch, existing):
    source_a, source_b, target = (tmp_path / name for name in ("a.safetensors", "b.safetensors", "out.safetensors"))
    save_file({"weight": torch.zeros(2, 2)}, source_a)
    save_file({"weight": torch.ones(2, 2)}, source_b)
    monkeypatch.setattr(converter, "make_plan", tiny_plan)
    receipt = target.with_suffix(".safetensors.manifest.json")
    if existing:
        converter.convert(source_a, target)
        old_weight, old_receipt = target.read_bytes(), receipt.read_bytes()
    replace = converter.os.replace

    def fail_receipt(source, destination):
        if Path(destination) == receipt:
            raise PermissionError("manifest directory write denied")
        return replace(source, destination)

    monkeypatch.setattr(converter.os, "replace", fail_receipt)
    with pytest.raises(PermissionError, match="manifest"):
        converter.convert(source_b, target, overwrite=existing)
    if existing:
        assert target.read_bytes() == old_weight and receipt.read_bytes() == old_receipt
        assert json.loads(receipt.read_text())["output_sha256"] == converter.file_hash(target)
    else:
        assert not target.exists() and not receipt.exists()
    assert not any(path.is_dir() for path in tmp_path.iterdir())


@pytest.mark.parametrize("suffix", [".convert.lock", ".manifest.json"])
def test_source_cannot_collide_with_fixed_conversion_auxiliary_files(tmp_path, suffix):
    target = tmp_path / "output.safetensors"
    source = Path(str(target) + suffix)
    save_file({"weight": torch.zeros(2, 2)}, source)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="Source must differ"):
        converter.convert(source, target)
    assert source.read_bytes() == before and not target.exists()


def test_source_hardlink_cannot_be_used_as_conversion_lock(tmp_path):
    source, target = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    save_file({"weight": torch.zeros(2, 2)}, source)
    before = source.read_bytes()
    Path(str(target) + ".convert.lock").hardlink_to(source)
    with pytest.raises(ValueError, match="Source must differ"):
        converter.convert(source, target)
    assert source.read_bytes() == before and not target.exists()


def test_external_writer_cannot_be_overwritten_during_final_install(tmp_path, monkeypatch):
    source, target = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    save_file({"weight": torch.ones(2, 2)}, source)
    monkeypatch.setattr(converter, "make_plan", tiny_plan)
    operation = "rename" if converter.os.name == "nt" else "link"
    install = getattr(converter.os, operation)
    external_bytes = []

    def create_before_install(temporary, destination):
        if Path(destination) == target:
            thread = threading.Thread(target=lambda: save_file({"weight": torch.full((2, 2), 99.)}, target))
            thread.start()
            thread.join(5)
            assert not thread.is_alive()
            external_bytes.append(target.read_bytes())
        return install(temporary, destination)

    monkeypatch.setattr(converter.os, operation, create_before_install)
    with pytest.raises(FileExistsError):
        converter.convert(source, target)
    assert target.read_bytes() == external_bytes[0]
    assert torch.equal(load_file(target)["weight"], torch.full((2, 2), 99.))
    assert not target.with_suffix(".safetensors.manifest.json").exists()
    assert not any(path.is_dir() for path in tmp_path.iterdir())


@pytest.mark.parametrize("conversion_fails", [False, True])
def test_workspace_cleanup_failure_preserves_conversion_result(tmp_path, monkeypatch, capsys, conversion_fails):
    source, target = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    save_file({"weight": torch.ones(2, 2)}, source)
    monkeypatch.setattr(converter, "make_plan", tiny_plan)

    def cleanup_denied(path):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(converter.shutil, "rmtree", cleanup_denied)
    if conversion_fails:
        def verify_failed(*args):
            raise ValueError("original verification error")
        monkeypatch.setattr(converter, "verify", verify_failed)
        with pytest.raises(ValueError, match="original verification error"):
            converter.convert(source, target)
        assert not target.exists()
    else:
        manifest = converter.convert(source, target)
        assert manifest["output_sha256"] == converter.file_hash(target)
        assert json.loads(target.with_suffix(".safetensors.manifest.json").read_text())["output_sha256"] == manifest["output_sha256"]
    assert "Could not remove conversion workspace" in capsys.readouterr().err
    assert any(path.is_dir() for path in tmp_path.iterdir())


@pytest.mark.parametrize("replace_inode", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_manifest_does_not_delete_external_checkpoint_update(tmp_path, monkeypatch, replace_inode, existing):
    source, target = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    external = tmp_path / "external.safetensors"
    save_file({"weight": torch.ones(2, 2)}, source)
    save_file({"weight": torch.full((2, 2), 99.)}, external)
    external_bytes = external.read_bytes()
    monkeypatch.setattr(converter, "make_plan", tiny_plan)
    receipt = target.with_suffix(".safetensors.manifest.json")
    if existing:
        converter.convert(source, target)
        old_weight, old_receipt = target.read_bytes(), receipt.read_bytes()
    replace = converter.os.replace

    def fail_receipt(temporary, destination):
        if Path(destination) == receipt:
            def update():
                if replace_inode:
                    replace(external, target)
                else:
                    target.write_bytes(external_bytes)
            thread = threading.Thread(target=update)
            thread.start()
            thread.join(5)
            assert not thread.is_alive()
            raise PermissionError("receipt denied after external update")
        return replace(temporary, destination)

    monkeypatch.setattr(converter.os, "replace", fail_receipt)
    with pytest.raises(PermissionError, match="receipt denied"):
        converter.convert(source, target, overwrite=existing)
    assert target.read_bytes() == external_bytes
    if existing:
        assert receipt.read_bytes() == old_receipt
        backups = list(tmp_path.glob("output.safetensors.*/previous.safetensors"))
        assert len(backups) == 1 and backups[0].read_bytes() == old_weight
    else:
        assert not receipt.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_rollback_cannot_destroy_file_created_after_ownership_check(tmp_path, monkeypatch, existing):
    source, target = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    external = tmp_path / "external.safetensors"
    save_file({"weight": torch.ones(2, 2)}, source)
    save_file({"weight": torch.full((2, 2), 99.)}, external)
    external_bytes = external.read_bytes()
    monkeypatch.setattr(converter, "make_plan", tiny_plan)
    receipt = target.with_suffix(".safetensors.manifest.json")
    if existing:
        converter.convert(source, target)
        old_weight, old_receipt = target.read_bytes(), receipt.read_bytes()
    replace, unlink = converter.os.replace, Path.unlink
    publish = converter.install_without_overwrite
    failed, changed = False, False

    def external_update():
        nonlocal changed
        thread = threading.Thread(target=lambda: replace(external, target))
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
        changed = True

    def fail_receipt(temporary, destination):
        nonlocal failed
        if Path(destination) == receipt:
            failed = True
            raise PermissionError("receipt denied")
        if failed and existing and Path(temporary).name == "previous.safetensors":
            external_update()  # Regression against destructive old backup restore.
        return replace(temporary, destination)

    def race_restore(temporary, destination):
        if failed and existing and Path(temporary).name == "previous.safetensors":
            external_update()
        return publish(temporary, destination)

    def race_cleanup(path, *args, **kwargs):
        if failed and not existing and (path == target or path.name == "rollback.safetensors"):
            external_update()
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(converter.os, "replace", fail_receipt)
    monkeypatch.setattr(converter, "install_without_overwrite", race_restore)
    monkeypatch.setattr(Path, "unlink", race_cleanup)
    with pytest.raises(PermissionError, match="receipt denied"):
        converter.convert(source, target, overwrite=existing)
    assert changed and target.read_bytes() == external_bytes
    if existing:
        assert receipt.read_bytes() == old_receipt
        backups = list(tmp_path.glob("output.safetensors.*/previous.safetensors"))
        assert len(backups) == 1 and backups[0].read_bytes() == old_weight
    else:
        assert not receipt.exists()
