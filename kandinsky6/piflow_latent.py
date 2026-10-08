"""PiFlow supports the canonical clean I2VA tail, without spatial inpainting."""
import torch


def validate_noise_mask(latent):
    mask = latent.get("noise_mask")
    if mask is None:
        return
    message = "K6 PiFlow supports only the canonical I2VA reference-tail noise mask; arbitrary latent masks are unsupported."
    if not latent.get("k6_reference_tail", False) or not getattr(mask, "is_nested", False):
        raise ValueError(message)
    masks, samples = mask.unbind(), latent["samples"].unbind()
    if len(masks) != 2 or len(samples) != 2:
        raise ValueError(message)
    video_mask, audio_mask = masks
    video, audio = samples
    if video.ndim != 5 or video.shape[2] < 2 or any(
        not torch.is_tensor(current) or current.shape != sample.shape
        for current, sample in zip(masks, samples)
    ):
        raise ValueError(message)
    if not (video_mask[:, :, :-1] == 1).all().item() or not (video_mask[:, :, -1:] == 0).all().item() or not (audio_mask == 1).all().item():
        raise ValueError(message)
