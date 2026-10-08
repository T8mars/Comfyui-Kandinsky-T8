"""Check native INT8 metadata without copying or scanning weight payloads."""
import json

import torch


def validate_int8_state_dict(state_dict, prefix=""):
    for key, weight in state_dict.items():
        if not key.startswith(prefix) or not key.endswith(".weight") or weight.dtype != torch.int8:
            continue
        stem = key.removesuffix("weight")
        marker = state_dict.get(stem + "comfy_quant")
        if marker is None:
            raise ValueError(f"Kandinsky 6 INT8 weight is missing comfy_quant: {key}")
        if marker.dtype != torch.uint8 or marker.ndim != 1:
            raise ValueError(f"Invalid Kandinsky 6 quantization marker: {key}")
        try:
            config = json.loads(bytes(marker.cpu().tolist()))
        except (ValueError, TypeError, UnicodeError) as error:
            raise ValueError(f"Invalid Kandinsky 6 quantization marker: {key}") from error
        if not isinstance(config, dict):
            raise ValueError(f"Invalid Kandinsky 6 quantization marker: {key}")
        if config.get("format") != "int8_tensorwise":
            raise ValueError(f"Kandinsky 6 INT8 weight requires int8_tensorwise metadata: {key}")
        scale = state_dict.get(stem + "weight_scale")
        if scale is None:
            raise ValueError(f"Kandinsky 6 INT8 weight is missing weight_scale: {key}")
        if weight.ndim != 2 or not scale.is_floating_point() or (
            scale.numel() != 1 and tuple(scale.shape) != (weight.shape[0], 1)
        ):
            raise ValueError(f"Invalid Kandinsky 6 INT8 scale shape/dtype: {key}")
        if not torch.isfinite(scale).all().item() or not (scale > 0).all().item():
            raise ValueError(f"Kandinsky 6 INT8 scale must be positive and finite: {key}")
        params = config.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if config.get("convrot", params.get("convrot", False)):
            group = config.get("convrot_groupsize", params.get("convrot_groupsize", 256))
            if type(group) is not int or group not in (16, 64, 256) or weight.shape[1] % group:
                raise ValueError(f"Invalid Kandinsky 6 INT8 ConvRot group: {key}")
