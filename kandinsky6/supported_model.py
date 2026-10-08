"""ComfyUI supported-model registration for Kandinsky 6 (video+audio)."""
import torch

import comfy.supported_models
import comfy.latent_formats

from . import model_base as k6_model_base
from .core_contract import SCHEDULER_DEFAULTS
from .quantization import validate_int8_state_dict


class Kandinsky6(comfy.supported_models.Kandinsky5):
    # matched against the unet_config produced by our detect branch (register.py)
    unet_config = {
        "image_model": "kandinsky6",
    }

    sampling_settings = {
        "shift": float(SCHEDULER_DEFAULTS["scheduler_scale"]),
    }

    latent_format = comfy.latent_formats.HunyuanVideo  # video latent
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        validate_int8_state_dict(state_dict, prefix)
        return k6_model_base.Kandinsky6(self, device=device)

    # clip_target() inherited from Kandinsky5 -> Qwen2.5-7B + CLIP-L
    # (the external text encoder is shared; the DiT's video/audio text projections
    #  are internal and live in the diffusion-model checkpoint).
