"""Prompt expansion with ComfyUI's native Qwen3.5 and model management."""

from pathlib import Path

_PROMPTS = Path(__file__).parent / "beautifier_prompts"


def join_captions(video_caption, audio_caption):
    if audio_caption.strip():
        return f"{video_caption} <AUDCAP>{audio_caption}<ENDAUDCAP>"
    return video_caption


def check_tags(text):
    """The plain Python beautifier's original tag-count check, without extra filters."""
    text = text.strip()
    if text.count("<S>") != text.count("<E>") or "<speech>" in text:
        raise ValueError(f"unbalanced speech tags: <S> {text.count('<S>')}, <E> {text.count('<E>')}")
    if text.count("<AUDCAP>") != 1 or text.count("<ENDAUDCAP>") != 1:
        raise ValueError("audio caption tags missing")
    return text


def split_captions(text):
    """Separate a checked full prompt for the two native K6 text-encoder inputs."""
    video, audio = check_tags(text).split("<AUDCAP>")
    audio, trailing = audio.split("<ENDAUDCAP>")
    return (video + trailing).strip(), audio.strip()


def chat_prompt(system_prompt, prompt, with_image=False):
    """Match Qwen3.5's original no-thinking HF chat template for system + user."""
    image = "<|vision_start|><|image_pad|><|vision_end|>" if with_image else ""
    return (
        f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{image}[SCENE DESCRIPTION]: {prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


class Kandinsky6BeautifyPrompt:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 - ComfyUI node API
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {
                        "lazy": True,
                        "tooltip": "Qwen3.5-9B from Load CLIP, not the Kandinsky text encoder.",
                    },
                ),
                "video_caption": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Describe the scene and motion. Put exact spoken lines inside <S>...<E>.",
                    },
                ),
                "audio_caption": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Describe voices and sounds; spoken words belong in the video caption.",
                    },
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "max_tokens": ("INT", {"default": 1500, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_caption", "audio_caption", "full_prompt")
    FUNCTION = "beautify"
    CATEGORY = "Kandinsky 6"
    DESCRIPTION = (
        "Expand video and audio captions with native Qwen3.5-9B. Connect the original "
        "Load Image output for I2VA; leave image disconnected for T2VA. Thinking and "
        "MTP are off. Disable enabled to pass your captions through without generating text."
    )

    def check_lazy_status(self, enabled, clip=None, **kwargs):
        return ["clip"] if enabled and clip is None else []

    def beautify(  # noqa: PLR0913, PLR0917 - ComfyUI node inputs
        self,
        clip,
        video_caption,
        audio_caption,
        enabled,
        seed,
        max_tokens,
        temperature,
        repetition_penalty,
        image=None,
    ):
        original = join_captions(video_caption, audio_caption)
        if not enabled:
            return video_caption, audio_caption, original
        mode = "i2av" if image is not None else "t2av"
        system_prompt = (_PROMPTS / f"{mode}_system.txt").read_text(encoding="utf-8")
        prompt = chat_prompt(system_prompt, original, with_image=image is not None)
        tokens = clip.tokenize(prompt, image=image, thinking=False, min_length=1)
        error = ""
        for attempt in range(3):
            generated = clip.generate(
                tokens,
                do_sample=True,
                max_length=max_tokens,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                min_p=0.0,
                repetition_penalty=repetition_penalty,
                presence_penalty=0.0,
                seed=(seed + attempt) % (1 << 64),
                mtp=False,
            )
            try:
                if len(generated) >= max_tokens and clip.decode(generated[-1:], skip_special_tokens=False) not in (
                    "<|im_end|>",
                    "<|endoftext|>",
                ):
                    raise ValueError("answer cut by max_tokens")
                text = check_tags(clip.decode(generated))
                video, audio = split_captions(text)
            except ValueError as exc:
                error = str(exc)
                continue
            return video, audio, text
        raise RuntimeError(
            f"Kandinsky 6 beautifier failed after 3 attempts: {error}. "
            "Increase max_tokens if truncated, retry with another seed, or set enabled=false."
        )
