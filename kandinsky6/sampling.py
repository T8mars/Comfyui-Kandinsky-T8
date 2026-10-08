"""Native PiFlow sampling with Comfy-owned model loading and latent formats."""

import comfy.model_management as mm
import comfy.model_prefetch
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import torch

from .piflow_contract import PIFLOW_DEFAULTS
from .piflow_conditioning import joint_conditioning
from .runtime_cleanup import cleanup_sampling
from .piflow_math import DXPolicy, policy_rollout_fm, shift_timesteps


def rollout(dit, video, audio, context, pooled, *, steps, dtype, reference=None, transformer_options=None,
            audio_context=None, audio_pooled=None):
    """Canonical multimodal DX rollout; one DiT evaluation per segment."""
    params = PIFLOW_DEFAULTS
    eps, shift = float(params["eps"]), float(params["shift"])
    final_scale = max(float(params["final_step_size_scale"]), eps)
    base_segment = 1.0 / (steps - (1.0 - final_scale))
    raw_src = 1.0
    batch = video.shape[0]
    options = dict(transformer_options or {})
    options.pop("k6_magcache", None)
    progress = comfy.utils.ProgressBar(steps)
    for index in range(steps):
        mm.throw_exception_if_processing_interrupted()
        segment = base_segment * (final_scale if index == steps - 1 else 1.0)
        raw_dst = max(raw_src - segment, eps)
        sigma_src = shift_timesteps(torch.full((batch,), raw_src, device=video.device), shift)
        if reference is not None:
            video[:, :, -1:] = reference
        predictions = dit(
            [video.to(dtype), audio.to(dtype)],
            sigma_src * 1000.0,
            context=context,
            y=pooled,
            k6_audio_context=audio_context,
            k6_audio_pooled_output=audio_pooled,
            k6_reference_tail=reference is not None,
            transformer_options=options,
        )
        video_grid = predictions[0].reshape(batch, dit.n_grid, video.shape[1], *video.shape[2:])
        audio_grid = predictions[1].reshape(batch, audio.shape[1], dit.n_grid, audio.shape[2]).movedim(2, 1)
        updated = []
        for state, grid in ((video, video_grid), (audio, audio_grid)):
            sigma = sigma_src.reshape(batch, *((state.ndim - 1) * [1]))
            policy = DXPolicy(grid, state, sigma, segment, shift=shift, eps=eps)
            result, _, _ = policy_rollout_fm(
                state,
                sigma,
                torch.full((batch,), raw_src, device=state.device),
                torch.full((batch,), raw_dst, device=state.device),
                int(params["num_policy_substeps"]),
                policy,
                interrupt_check=mm.throw_exception_if_processing_interrupted,
            )
            updated.append(result)
        video, audio = updated
        raw_src = raw_dst
        progress.update(1)
    if reference is not None:
        video[:, :, -1:] = reference
    return video, audio


def sample_piflow(model, latent, positive, seed, steps):
    context, metadata = joint_conditioning(positive)
    device = model.load_device
    dtype = model.model.get_dtype_inference()
    mm.load_models_gpu([model], memory_required=model.model.memory_required(latent["samples"].shape))
    try:
        model.pre_run()
        clean = model.model.process_latent_in(latent["samples"].to(device=device, dtype=torch.float32))
        video, audio = (
            comfy.sample.prepare_noise(latent["samples"], seed, latent.get("batch_index")).to(device).unbind()
        )
        reference = clean.unbind()[0][:, :, -1:].clone() if latent.get("k6_reference_tail", False) else None
        video, audio = rollout(
            model.model.diffusion_model,
            video,
            audio,
            mm.cast_to_device(context, device, dtype),
            mm.cast_to_device(metadata["pooled_output"], device, dtype),
            steps=int(steps),
            dtype=dtype,
            reference=reference,
            transformer_options=model.model_options.get("transformer_options", {}),
            audio_context=mm.cast_to_device(metadata.get("k6_audio_context", context), device, dtype),
            audio_pooled=mm.cast_to_device(metadata.get("k6_audio_pooled_output", metadata["pooled_output"]), device, dtype),
        )
        samples = model.model.process_latent_out(comfy.nested_tensor.NestedTensor([video, audio]))
        result = latent.copy()
        result["samples"] = samples.to(device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
        return result
    finally:
        cleanup_sampling(model, comfy.model_prefetch.cleanup_prefetch_queues)
