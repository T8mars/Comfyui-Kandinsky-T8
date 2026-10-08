"""Small fixtures exercise the real Kitchen kernels and serialized output."""
import importlib.util
import json
from pathlib import Path

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
