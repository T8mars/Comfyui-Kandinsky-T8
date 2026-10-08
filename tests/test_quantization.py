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


@pytest.mark.parametrize("fault", ["missing_marker", "missing_scale", "negative", "nan", "inf", "shape", "group", "format"])
def test_damaged_int8_auxiliary_is_rejected(fault):
    state = payload(group=256 if fault == "group" else 64)
    if fault == "missing_marker":
        state.pop("layer.comfy_quant")
    elif fault == "missing_scale":
        state.pop("layer.weight_scale")
    elif fault == "shape":
        state["layer.weight_scale"] = torch.ones(64)
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
