import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("k6_piflow_latent_test", Path(__file__).parents[1] / "kandinsky6/piflow_latent.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Joint:
    is_nested = True

    def __init__(self, *streams):
        self.streams = streams

    def unbind(self):
        return self.streams


def canonical():
    video, audio = torch.zeros(1, 16, 3, 2, 2), torch.zeros(1, 5, 40)
    video_mask = torch.ones_like(video)
    video_mask[:, :, -1] = 0
    return {"samples": Joint(video, audio), "noise_mask": Joint(video_mask, torch.ones_like(audio)), "k6_reference_tail": True}


def test_canonical_tail_and_no_mask_are_accepted():
    module.validate_noise_mask(canonical())
    module.validate_noise_mask({"samples": object()})


@pytest.mark.parametrize("fault", ["ordinary", "video", "audio", "tail", "shape"])
def test_ignored_mask_semantics_are_rejected(fault):
    latent = canonical()
    video_mask, audio_mask = latent["noise_mask"].unbind()
    if fault == "ordinary":
        latent["k6_reference_tail"] = False
        latent["noise_mask"] = torch.zeros(2, 2)
    elif fault == "video":
        video_mask[:, :, 0] = 0
    elif fault == "audio":
        audio_mask[:] = 0
    elif fault == "tail":
        video_mask[:, :, -1] = 1
    elif fault == "shape":
        latent["noise_mask"] = Joint(video_mask[:, :, :1], audio_mask)
    with pytest.raises(ValueError, match="reference-tail"):
        module.validate_noise_mask(latent)
