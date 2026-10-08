"""Decode Kandinsky 6 audio latents with TOD-VAE and BigVGAN."""
import contextlib

import torch

import comfy.model_management as mm
import comfy.model_patcher

from .mmaudio.ext.autoencoder import AutoEncoderModule
from .core_contract import AUDIO_DEFAULTS

OUTPUT_SAMPLE_RATE = int(AUDIO_DEFAULTS["sample_rate"])
NORMALIZATION_MODE = str(AUDIO_DEFAULTS["normalization_mode"])
DOWNSAMPLE_FACTOR = {
    str(AUDIO_DEFAULTS["mode"]): int(AUDIO_DEFAULTS["latent_downsample_factor"]),
}


class K6AudioVAE:
    def __init__(self, tod_vae_ckpt, bigvgan_dir, mode=str(AUDIO_DEFAULTS["mode"]),
                 scaling_factor=float(AUDIO_DEFAULTS["scaling_factor"]),
                 mean_value=0.0, device=None, dtype=torch.float32):
        self.mode = mode
        self.scaling_factor = scaling_factor
        self.mean_value = mean_value
        self.downsample_factor = DOWNSAMPLE_FACTOR[mode]
        self.sample_rate = OUTPUT_SAMPLE_RATE
        # These names are consumed by Comfy's standard VAEDecodeAudio node.
        self.audio_sample_rate = OUTPUT_SAMPLE_RATE
        self.audio_sample_rate_output = OUTPUT_SAMPLE_RATE
        self.load_device = mm.vae_device()
        self.offload_device = mm.vae_offload_device()
        initial_device = self.offload_device if device is None else torch.device(device)
        self.dtype = dtype

        self.tod = AutoEncoderModule(
            vae_ckpt_path=tod_vae_ckpt,
            vocoder_ckpt_path=bigvgan_dir,
            mode=mode,
            need_vae_encoder=False,
            need_vae_decoder=True,
        ).to(initial_device).eval()
        archive_dtypes = getattr(mm, "archive_model_dtypes", None)
        if archive_dtypes is not None:
            archive_dtypes(self.tod)
        patcher_type = getattr(
            comfy.model_patcher,
            "CoreModelPatcher",
            comfy.model_patcher.ModelPatcher,
        )
        self.patcher = patcher_type(
            self.tod,
            load_device=self.load_device,
            offload_device=self.offload_device,
            size=mm.module_size(self.tod),
        )

    def get_models(self):
        """Expose the decoder to Comfy's prompt model tracker."""
        return [self.patcher]

    @torch.no_grad()
    def decode(self, latent):
        """Decode BTF latents to Comfy's ``[B, samples, channels]`` layout."""
        # TOD-VAE + BigVGAN are sizeable enough that bypassing Comfy's model
        # tracker can OOM directly after the DiT. Let Comfy evict/offload first.
        mm.load_models_gpu(
            [self.patcher],
            memory_required=1024 * 1024 * 1024,
            force_full_load=True,
        )
        cuda_context = getattr(mm, "cuda_device_context", None)
        device_context = (
            cuda_context(self.load_device)
            if cuda_context is not None
            else contextlib.nullcontext()
        )
        with device_context:
            z = latent.to(self.load_device, torch.float32)
            z = z / self.scaling_factor + self.mean_value
            mel = self.tod.decode(z.transpose(1, 2))
            wav = self.tod.vocode(mel)
            if wav.ndim == 2:
                wav = wav.unsqueeze(1)
            if wav.ndim != 3:
                raise RuntimeError(
                    f"Kandinsky 6 audio decoder returned an invalid shape: {tuple(wav.shape)}"
                )
            # BigVGAN uses [B,C,N]; Comfy VAE.decode uses [B,N,C].
            return wav.movedim(1, -1).to(
                device=mm.intermediate_device(), dtype=torch.float32, copy=True
            )

    @torch.no_grad()
    def decode_canonical(self, latent):
        """Decode with the output normalization used by the canonical K6 pipeline."""
        waveform = self.decode(latent)
        if NORMALIZATION_MODE == "clip":
            return waveform.clamp(-1.0, 1.0)
        if NORMALIZATION_MODE == "normalize":
            peak = waveform.abs().amax(dim=(1, 2), keepdim=True).clamp_min_(1e-8)
            return waveform / peak
        raise ValueError(f"unknown audio normalization_mode={NORMALIZATION_MODE}")
