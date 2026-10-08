"""Damaged quantized weights must fail before native silent float fallback."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("k6_quantization_test", Path(__file__).parents[1] / "kandinsky6/quantization.py")
quantization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantization)


def payload(prefix="", group=64):
    config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}
    return {
        prefix + "layer.weight": torch.ones(64, 64, dtype=torch.int8),
        prefix + "layer.weight_scale": torch.ones(64, 1),
        prefix + "layer.comfy_quant": torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8),
    }


@pytest.mark.parametrize("fault", ["missing_marker", "missing_scale", "negative", "nan", "inf", "shape", "group", "format", "scale_f64"])
def test_damaged_int8_auxiliary_is_rejected(fault):
    state = payload(group=256 if fault == "group" else 64)
    if fault == "missing_marker":
        state.pop("layer.comfy_quant")
    elif fault == "missing_scale":
        state.pop("layer.weight_scale")
    elif fault == "shape":
        state["layer.weight_scale"] = torch.ones(64)
    elif fault == "scale_f64":
        state["layer.weight_scale"] = torch.ones(64, 1, dtype=torch.float64)
    elif fault in {"negative", "nan", "inf"}:
        state["layer.weight_scale"][0] = {"negative": -1, "nan": float("nan"), "inf": float("inf")}[fault]
    elif fault == "format":
        state["layer.comfy_quant"] = torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8)
        state["layer.weight_scale"] = torch.ones(())
    with pytest.raises(ValueError, match="Kandinsky 6"):
        quantization.validate_int8_state_dict(state)


def test_normal_int8_prefix_and_float_passthrough_are_preserved():
    state = payload("model.diffusion_model.")
    state["model.diffusion_model.time.weight"] = torch.ones(3, 3)
    state["vae.layer.weight"] = torch.ones(3, 3, dtype=torch.int8)
    original_keys = set(state)
    quantization.validate_int8_state_dict(state, "model.diffusion_model.")
    assert set(state) == original_keys
    state["model.diffusion_model.layer.weight_scale"] = torch.ones(())
    quantization.validate_int8_state_dict(state, "model.diffusion_model.")


@pytest.mark.parametrize("flag", ["false", 0, 1, None, [], {}])
def test_non_boolean_rotation_flag_is_rejected(flag):
    state = payload()
    state["layer.comfy_quant"] = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "params": {"convrot": flag}}).encode()), dtype=torch.uint8)
    with pytest.raises(ValueError, match="flag must be a boolean"):
        quantization.validate_int8_state_dict(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA INT8 kernel required")
@pytest.mark.parametrize("shape", [(), (1,), (1, 1), (1, 1, 1), (1, 1, 1, 1)])
def test_single_value_scales_run_fast_int8_linear(shape):
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

    state = payload()
    state["layer.comfy_quant"] = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": False}).encode()), dtype=torch.uint8)
    state["layer.weight_scale"] = torch.ones(shape)
    quantization.validate_int8_state_dict(state)
    params = TensorWiseINT8Layout.Params(scale=state["layer.weight_scale"].cuda(),
                                        orig_dtype=torch.float32, orig_shape=(64, 64))
    weight = QuantizedTensor(state["layer.weight"].cuda(), "TensorWiseINT8Layout", params)
    result = torch.nn.functional.linear(torch.ones(2, 64, device="cuda"), weight)
    torch.testing.assert_close(result, torch.full((2, 64), 64., device="cuda"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_float_weight_cannot_silently_truncate_under_int8_marker(dtype):
    state = payload()
    state["layer.weight"] = torch.full((64, 64), 0.25, dtype=dtype)
    with pytest.raises(ValueError, match="must have INT8 storage dtype"):
        quantization.validate_int8_state_dict(state)
    # Unmarked sensitive layers retain their floating-point values.
    state.pop("layer.comfy_quant")
    quantization.validate_int8_state_dict(state)
    assert torch.equal(state["layer.weight"], torch.full((64, 64), 0.25, dtype=dtype))
