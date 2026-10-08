# ComfyUI Kandinsky T8

**English** | [简体中文](README.zh-CN.md)

Native Kandinsky 6 video and audio generation for ComfyUI, with single-file **INT8 ConvRot** models. Supports text-to-video+audio and image-to-video+audio for **Lite, Lite distill and Pro distill**.

[Models](https://huggingface.co/t8star/Kandinsky-Comfy) · [Workflows](example_workflows) · [Validation](VALIDATION.md) · [Upstream](https://github.com/kandinskylab/kandinsky-6)

## Install

Requires **ComfyUI 0.39.0+**, **comfy-kitchen 0.2.37+** and an NVIDIA CUDA GPU with INT8 Tensor Core support.

In ComfyUI-Manager, search **Kandinsky T8** (publisher **t8star**). Alternatively, run in `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/T8mars/Comfyui-Kandinsky-T8.git
cd Comfyui-Kandinsky-T8
python -m pip install -r requirements.txt
```

Use ComfyUI's Python environment, then restart ComfyUI. Disable the original Kandinsky 6 extension if installed: both extensions register the same node names.

## Models

Download from [t8star/Kandinsky-Comfy](https://huggingface.co/t8star/Kandinsky-Comfy). Place one DiT in `ComfyUI/models/diffusion_models`:

| Model | File | Size | Sampling |
|---|---|---:|---|
| Lite | `k6_lite_int8_convrot.safetensors` | 4.87 GB | KSampler, 50 steps, CFG 5 |
| Lite distill | `k6_lite_distill_int8_convrot.safetensors` | 3.77 GB | PiFlow, 10 steps, CFG 1 |
| Pro distill | `k6_pro_distill_int8_convrot.safetensors` | 34.97 GB | PiFlow, 10 steps, CFG 1 |

Each file contains the complete joint video/audio DiT. Text encoders and decoders remain independent:

```text
ComfyUI/models/
  text_encoders/qwen_2.5_vl_7b.safetensors
  text_encoders/clip_l.safetensors
  vae/hunyuan_video_vae_bf16.safetensors
  audio_vae/v1-44.pth
  audio_vae/bigvgan_vocoder/config.json
  audio_vae/bigvgan_vocoder/bigvgan_generator.pt
```

The workflow's **Download models** button downloads these shared components from their pinned upstream repositories and verifies their size and SHA-256 before installation or reuse. DiT downloads are separate. `extra_model_paths.yaml` is supported.

## Run

Import a [Base or PiFlow workflow](example_workflows), choose the matching model in **UNETLoader** and keep `weight_dtype=default`. Use Base for Lite; PiFlow for either distilled model. Choose Text or Image for the input mode.

For six model-selected examples, download the [GUI workflow pack](https://github.com/T8mars/Comfyui-Kandinsky-T8/releases/download/v0.1.4/Kandinsky-T8-0.1.4-GUI-Workflows.zip), unzip it and drag a root-level JSON onto the ComfyUI canvas. The extension installs `kandinsky6_i2va_portrait.png` directly into `ComfyUI/input` for image workflows; you can also choose your own image in LoadImage.

Default output: **864×480, 121 frames, 24 fps**, MP4 with audio. Width and height must be multiples of 16; frame count must be `4*n+1`. PiFlow requires CFG=1, denoise=1 and one global, full-range conditioning; masks are limited to the Image workflow's clean reference tail.

Uses ComfyUI's native **UNETLoader / ModelPatcher**, **DualCLIPLoader (`kandinsky5`)**, **VAELoader**, latent/conditioning interfaces and memory offloading. ComfyUI core files are not modified. Lite uses stock KSampler; distilled models use the dedicated Kandinsky6Sampler for PiFlow.

## Validation and conversion

All three models completed full text and image workflows on **RTX 5090 Laptop 24 GB / 64 GB RAM**. All six INT8 outputs passed video/audio decoding checks. Pro distill relies on dynamic VRAM/offloading; model file size is not peak VRAM usage.

[Validation](VALIDATION.md) includes timings and test evidence. [Technical notes](docs/technical-notes.zh-CN.md) cover quantization, optional source downloads and `tools/convert_int8.py`. SHA-256 checksums and conversion manifests are included in the model repository.

## Credits and licenses

Based on [Kandinsky Lab's official extension](https://github.com/kandinskylab/kandinsky-6), revision `bf676db9b1335d22ebca85089355732687de3d0a`. Code and Kandinsky DiT weights use MIT; original license notices are retained. Shared components have their own licenses, including **CC BY-NC 4.0** for the TOD audio decoder. See the [model card](https://huggingface.co/t8star/Kandinsky-Comfy) for component sources and terms.

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com)

[Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI distribution](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
