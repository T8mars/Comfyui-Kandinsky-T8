"""Full structural validation uses header-sized fixtures, without model weights."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("k6_checkpoint_test", Path(__file__).parents[1] / "kandinsky6/checkpoint.py")
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


def shapes(family="1792", grid=1):
    result = copy.deepcopy(checkpoint.released_shapes()[family])
    for key in ("out_layer.out_layer.weight", "out_layer.out_layer.bias",
                "audio_outLayer.out_layer.weight", "audio_outLayer.out_layer.bias"):
        result[key][0] *= grid
    return result


@pytest.mark.parametrize("family,grid", [("1792", 1), ("1792", 10), ("4096", 1), ("4096", 10)])
def test_released_av_structures_and_optional_reference_types(family, grid):
    structure = shapes(family, grid)
    checkpoint.validate_released_shapes(structure)
    structure.pop("visual_token_type_embeddings.weight", None)
    checkpoint.validate_released_shapes(structure)


@pytest.mark.parametrize("key", ["audio_outLayer.out_layer.weight", "visual_blocks.0.va_modulation.out_layer.weight",
                               "audio_text_transformer_blocks.0.self_attention.query_norm.weight"])
def test_incomplete_payload_is_rejected(key):
    structure = shapes()
    structure.pop(key)
    with pytest.raises(ValueError, match="Incomplete"):
        checkpoint.validate_released_shapes(structure)


def test_mismatched_output_grid_and_projection_are_rejected():
    structure = shapes(grid=10)
    structure["audio_outLayer.out_layer.weight"][0] = 40
    with pytest.raises(ValueError, match="wrong_shapes"):
        checkpoint.validate_released_shapes(structure)
    structure = shapes()
    structure["visual_blocks.0.audioT.self_attention.to_query.weight"] = [16, 16]
    with pytest.raises(ValueError, match="wrong_shapes"):
        checkpoint.validate_released_shapes(structure)
