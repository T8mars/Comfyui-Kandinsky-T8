"""Stream a complete K6 AV DiT into one native ComfyUI INT8 ConvRot file."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from comfy_kitchen.tensor import TensorWiseINT8Layout
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kandinsky6.checkpoint import native_key

RECIPE = "k6-attention-ffn-convrot-v1"
LINEAR = re.compile(
    r"^(?:visual_blocks\.\d+\.(?:"
    r"(?:videoT|audioT)\.(?:"
    r"(?:self_attention|cross_attention)\.(?:to_query|to_key|to_value|out_layer)|"
    r"feed_forward\.(?:in_layer|out_layer))|"
    r"(?:va_cross_attention|av_cross_attention)\.(?:to_query|to_key|to_value|out_layer))|"
    r"(?:video|audio)_text_transformer_blocks\.\d+\.(?:"
    r"self_attention\.(?:to_query|to_key|to_value|out_layer)|"
    r"feed_forward\.(?:in_layer|out_layer)))\.weight$"
)
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
               "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            result.update(block)
    return result.hexdigest()


def group_size(key, shape):
    if not LINEAR.fullmatch(key) or len(shape) != 2:
        return None
    return next((group for group in (256, 64, 16) if shape[1] % group == 0), None)


def make_plan(source, excluded=()):
    tensors = []
    seen = set()
    excluded = set(excluded)
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if "_quantization_metadata" in metadata or any(key.endswith(".comfy_quant") for key in checkpoint.keys()):  # noqa: SIM118
            raise ValueError("Source is already quantized; use the original K6 checkpoint")
        for source_key in checkpoint.keys():  # noqa: SIM118
            key = native_key(source_key)
            if key in seen:
                raise ValueError(f"Conflicting native checkpoint key: {key}")
            seen.add(key)
            view = checkpoint.get_slice(source_key)
            shape, dtype = list(view.get_shape()), view.get_dtype()
            if dtype not in DTYPE_BYTES:
                raise ValueError(f"Unsupported source dtype {dtype}: {source_key}")
            group = None if key in excluded else group_size(key, shape)
            if group and dtype not in {"F32", "F16", "BF16"}:
                raise ValueError(f"Expected floating-point source weight: {source_key}")
            tensors.append({"source": source_key, "key": key, "shape": shape,
                            "dtype": dtype, "group": group})
        shapes = {item["key"]: item["shape"] for item in tensors}
        required = ("visual_embeddings.in_layer.weight", "audio_embeddings.in_layer.weight", "out_layer.out_layer.weight")
        if not all(key in shapes for key in required):
            raise ValueError("Not a complete Kandinsky 6 joint AV DiT")
        dim = shapes[required[0]][0]
        if dim not in (1792, 4096) or shapes[required[1]][0] != dim // 2:
            raise ValueError("Checkpoint does not match released K6 Lite/Pro dimensions")
        if shapes[required[2]][0] not in (64, 640):
            raise ValueError("Checkpoint has an unsupported flow/PiFlow output head")
        blocks, text_blocks = (32, 2) if dim == 1792 else (60, 4)
        expected = set()
        projections = ("to_query", "to_key", "to_value", "out_layer")
        for index in range(blocks):
            for branch in ("videoT", "audioT"):
                stem = f"visual_blocks.{index}.{branch}"
                for attention in ("self_attention", "cross_attention"):
                    expected.update(f"{stem}.{attention}.{projection}.weight" for projection in projections)
                expected.update(f"{stem}.feed_forward.{projection}.weight" for projection in ("in_layer", "out_layer"))
            for attention in ("va_cross_attention", "av_cross_attention"):
                expected.update(f"visual_blocks.{index}.{attention}.{projection}.weight" for projection in projections)
        for branch in ("video", "audio"):
            for index in range(text_blocks):
                stem = f"{branch}_text_transformer_blocks.{index}"
                expected.update(f"{stem}.self_attention.{projection}.weight" for projection in projections)
                expected.update(f"{stem}.feed_forward.{projection}.weight" for projection in ("in_layer", "out_layer"))
        actual = {key for key in seen if LINEAR.fullmatch(key)}
        if actual != expected:
            raise ValueError(f"Incomplete/unreleased K6 block structure: missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}")
        if not any(item["group"] for item in tensors):
            raise ValueError("No K6 attention/FFN layers selected")
        unknown = excluded - seen
        if unknown:
            raise ValueError(f"Unknown excluded layers: {sorted(unknown)}")
    return sorted(tensors, key=lambda item: item["key"]), metadata


def output_schema(plan):
    schema, configs = {}, {}
    for item in plan:
        key, shape = item["key"], item["shape"]
        schema[key] = {"dtype": "I8" if item["group"] else item["dtype"], "shape": shape}
        if item["group"]:
            prefix = key.removesuffix("weight")
            config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": item["group"]}
            encoded = json.dumps(config, separators=(",", ":")).encode("utf-8")
            configs[prefix + "comfy_quant"] = encoded
            schema[prefix + "comfy_quant"] = {"dtype": "U8", "shape": [len(encoded)]}
            schema[prefix + "weight_scale"] = {"dtype": "F32", "shape": [shape[0], 1]}
    offset = 0
    for key in sorted(schema):
        entry = schema[key]
        size = math.prod(entry["shape"]) * DTYPE_BYTES[entry["dtype"]]
        entry["data_offsets"] = [offset, offset + size]
        offset += size
    return schema, configs, offset


def read_header(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return 8 + size, json.loads(stream.read(size))


def region_hash(path, start, length):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        stream.seek(start)
        while length:
            block = stream.read(min(length, 16 << 20))
            if not block:
                raise ValueError("Truncated tensor payload")
            result.update(block)
            length -= len(block)
    return result.hexdigest()


def verify(path, plan, schema, passthrough):
    data_start, header = read_header(path)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        if set(checkpoint.keys()) != set(schema):
            raise ValueError("Output checkpoint keys do not match the manifest")
        for key, entry in schema.items():
            view = checkpoint.get_slice(key)
            if list(view.get_shape()) != entry["shape"] or view.get_dtype() != entry["dtype"]:
                raise ValueError(f"Output shape/dtype mismatch: {key}")
        for item in plan:
            key = item["key"]
            if item["group"]:
                prefix = key.removesuffix("weight")
                scale = checkpoint.get_tensor(prefix + "weight_scale")
                if not torch.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(f"Invalid INT8 scale: {key}")
                marker = json.loads(bytes(checkpoint.get_tensor(prefix + "comfy_quant").tolist()))
                if marker != {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": item["group"]}:
                    raise ValueError(f"Invalid ConvRot marker: {key}")
            else:
                start, end = header[key]["data_offsets"]
                if region_hash(path, data_start + start, end - start) != passthrough[key]:
                    raise ValueError(f"Passthrough tensor changed: {key}")


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def convert(source, destination, rows=256, device="cuda", excluded=(), overwrite=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Destination must differ from source")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    plan, source_metadata = make_plan(str(source), excluded)
    schema, configs, payload_size = output_schema(plan)
    source_sha = file_hash(source)
    metadata = {**source_metadata, "k6_quant_recipe": RECIPE,
                "k6_source_sha256": source_sha, "k6_source_filename": source.name,
                "k6_kitchen_version": importlib.metadata.version("comfy-kitchen"),
                "k6_excluded_layers": json.dumps(sorted(excluded))}
    receipt_path = source.with_suffix(source.suffix + ".download.json")
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("actual_sha256") != source_sha:
            raise ValueError("Source does not match its download receipt")
        for field in ("repo", "revision", "filename"):
            metadata[f"k6_source_{field}"] = receipt[field]
    header = json.dumps({"__metadata__": metadata, **dict(sorted(schema.items()))}, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    data_start = 8 + len(header)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < data_start + payload_size + (1 << 30):
        raise OSError("Not enough free space for the converted checkpoint")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    passthrough, metrics = {}, []
    started = time.monotonic()
    with temporary.open("w+b") as output, safe_open(str(source), framework="pt", device="cpu") as checkpoint:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        output.truncate(data_start + payload_size)
        for key, encoded in configs.items():
            output.seek(data_start + schema[key]["data_offsets"][0])
            output.write(encoded)
        for index, item in enumerate(plan):
            key, shape, group = item["key"], item["shape"], item["group"]
            prefix = key.removesuffix("weight")
            view = checkpoint.get_slice(item["source"])
            tensor_hash = hashlib.sha256()
            if not shape:
                data = tensor_bytes(checkpoint.get_tensor(item["source"]))
                output.seek(data_start + schema[key]["data_offsets"][0])
                output.write(data)
                passthrough[key] = hashlib.sha256(data).hexdigest()
                continue
            row_elements = math.prod(shape[1:])
            step = rows if group else max(1, (16 << 20) // max(1, row_elements * DTYPE_BYTES[item["dtype"]]))
            for start in range(0, shape[0], step):
                stop = min(start + step, shape[0])
                block = view[start:stop]
                if group:
                    weight = block.to(device=device, dtype=torch.float32)
                    if not torch.isfinite(weight).all():
                        raise ValueError(f"Non-finite source weight: {key}")
                    qdata, params = TensorWiseINT8Layout.quantize(
                        weight, per_channel=True, convrot=True, convrot_groupsize=group, stochastic_rounding=0,
                    )
                    output.seek(data_start + schema[prefix + "weight_scale"]["data_offsets"][0] + start * 4)
                    output.write(tensor_bytes(params.scale))
                    if start == 0:
                        sample_count = min(32, stop)
                        sample_params = TensorWiseINT8Layout.Params(
                            scale=params.scale[:sample_count], orig_dtype=torch.float32,
                            orig_shape=(sample_count, shape[1]), convrot=True, convrot_groupsize=group,
                        )
                        reconstructed = TensorWiseINT8Layout.dequantize(qdata[:sample_count], sample_params)
                        reference = weight[:sample_count]
                        relative = ((reconstructed - reference).norm() / reference.norm().clamp_min(1e-30)).item()
                        metrics.append({"layer": key, "group": group, "sample_relative_l2": relative})
                        del reconstructed, reference, sample_params
                    data = tensor_bytes(qdata)
                    del weight, qdata, params
                else:
                    data = tensor_bytes(block)
                    tensor_hash.update(data)
                output.seek(data_start + schema[key]["data_offsets"][0] + start * row_elements * DTYPE_BYTES[schema[key]["dtype"]])
                output.write(data)
                del block, data
            if not group:
                passthrough[key] = tensor_hash.hexdigest()
            if index % 40 == 0:
                print(f"CONVERT {index + 1}/{len(plan)} {key} elapsed={time.monotonic() - started:.1f}s", flush=True)
        output.flush()
        os.fsync(output.fileno())
    verify(str(temporary), plan, schema, passthrough)
    output_sha = file_hash(temporary)
    os.replace(temporary, destination)
    manifest = {"recipe": RECIPE, "source": str(source), "source_sha256": source_sha,
                "output": str(destination), "output_sha256": output_sha,
                "output_bytes": destination.stat().st_size, "kitchen_version": metadata["k6_kitchen_version"],
                "quantized_layers": len(metrics), "groups": dict(Counter(item["group"] for item in metrics)),
                "elapsed_seconds": time.monotonic() - started, "tensors": plan,
                "sample_weight_metrics": metrics, "passthrough_sha256": passthrough,
                "verification": "schema, scales, markers, all passthrough tensor bytes verified"}
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"VERIFIED {destination} bytes={manifest['output_bytes']} layers={len(metrics)} sha256={output_sha}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exclude-layer", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    if args.dry_run:
        plan, _ = make_plan(str(args.source), args.exclude_layer)
        _, _, size = output_schema(plan)
        print(json.dumps({"recipe": RECIPE, "payload_bytes": size,
                          "quantized_layers": sum(bool(item["group"]) for item in plan), "tensors": plan}, indent=2))
    else:
        destination = args.destination or args.source.with_name(args.source.stem + "_int8_convrot.safetensors")
        convert(args.source, destination, args.rows, args.device, args.exclude_layer, args.overwrite)


if __name__ == "__main__":
    main()
