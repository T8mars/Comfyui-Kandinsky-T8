"""Checkpoint key normalization shared by the loader and offline converter."""

from functools import lru_cache
import json
from pathlib import Path

PREFIXES = (
    ("visual_transformer_blocks.", "visual_blocks."),
    ("audio_out_layer.", "audio_outLayer."),
)
INNER_NAMES = (
    (".timestep_embedder.linear_1.", ".in_layer."),
    (".timestep_embedder.linear_2.", ".out_layer."),
    (".video_dec_block.", ".videoT."),
    (".audio_dec_block.", ".audioT."),
    (".feed_forward.net.0.proj.", ".feed_forward.in_layer."),
    (".feed_forward.net.2.", ".feed_forward.out_layer."),
    (".attn.", ".self_attention."),
)


def native_key(key):
    for source, target in PREFIXES:
        if key.startswith(source):
            key = target + key[len(source):]
            break
    for source, target in INNER_NAMES:
        key = key.replace(source, target)
    return key


@lru_cache(maxsize=1)
def released_shapes():
    return json.loads(Path(__file__).with_name("checkpoint_shapes.json").read_text(encoding="utf-8"))["models"]


def validate_released_shapes(shapes):
    """Validate the full AV payload against the fixed released header contract."""
    visual = shapes.get("visual_embeddings.in_layer.weight", ())
    output = shapes.get("out_layer.out_layer.weight", ())
    if len(visual) != 2 or len(output) != 2 or str(visual[0]) not in released_shapes():
        raise ValueError("Not a complete released Kandinsky 6 joint AV DiT")
    if output[0] not in (64, 640):
        raise ValueError("Unsupported Kandinsky 6 output grid size")
    grid = output[0] // 64
    expected = released_shapes()[str(visual[0])]
    optional = "visual_token_type_embeddings.weight"
    missing = set(expected) - set(shapes) - {optional}
    extra = set(shapes) - set(expected)
    wrong = []
    for key, shape in expected.items():
        if key not in shapes:
            continue
        required = list(shape)
        if key in ("out_layer.out_layer.weight", "out_layer.out_layer.bias",
                   "audio_outLayer.out_layer.weight", "audio_outLayer.out_layer.bias"):
            required[0] *= grid
        if list(shapes[key]) != required:
            wrong.append(key)
    if missing or extra or wrong:
        raise ValueError("Incomplete/unreleased K6 tensor structure: "
                         f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}, wrong_shapes={wrong[:5]}")
