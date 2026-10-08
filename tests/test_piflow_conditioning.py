import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("k6_piflow_conditioning", Path(__file__).parents[1] / "kandinsky6/piflow_conditioning.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("extra", [{"start_percent": 0.5}, {"end_percent": 0.5}, {"clip_start_percent": 0.1},
                                  {"area": (16, 16, 0, 0)}, {"mask": object()}, {"strength": 0.1},
                                  {"control": object()}, {"hooks": object()}])
def test_unsupported_single_condition_is_rejected(extra):
    with pytest.raises(ValueError, match="does not support"):
        module.joint_conditioning([[object(), {"pooled_output": object(), **extra}]])


def test_full_range_joint_condition_preserves_audio_and_encoder_metadata():
    context = object()
    metadata = {"pooled_output": object(), "start_percent": 0, "end_percent": 1, "strength": 1,
                "attention_mask": object(), "k6_audio_context": object(), "k6_audio_pooled_output": object()}
    actual_context, actual_metadata = module.joint_conditioning([[context, metadata]])
    assert actual_context is context
    assert actual_metadata is metadata


def test_multiple_conditions_and_missing_pooled_are_rejected():
    with pytest.raises(ValueError, match="one joint"):
        module.joint_conditioning([[object(), {}], [object(), {}]])
    with pytest.raises(ValueError, match="pooled"):
        module.joint_conditioning([[object(), {}]])
