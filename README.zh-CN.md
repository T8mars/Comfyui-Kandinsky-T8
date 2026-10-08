# ComfyUI Kandinsky T8

[English](README.md) | **简体中文**

基于 ComfyUI 原生模型管道的 Kandinsky 6 音视频生成节点，提供单文件 **INT8 ConvRot** 模型。支持 **Lite、Lite 蒸馏版、Pro 蒸馏版**的文生音视频与图生音视频。

[模型下载](https://huggingface.co/t8star/Kandinsky-Comfy) · [基础工作流](example_workflows) · [验证记录](VALIDATION.md) · [上游项目](https://github.com/kandinskylab/kandinsky-6)

## 安装

需要 **ComfyUI 0.39.0+**、**comfy-kitchen 0.2.37+**，以及支持 INT8 Tensor Core 的 NVIDIA CUDA 显卡。

在 ComfyUI 节点管理器中搜索 **Kandinsky T8**，发布者为 **t8star**。也可在 `ComfyUI/custom_nodes` 中执行：

```bash
git clone https://github.com/T8mars/Comfyui-Kandinsky-T8.git
cd Comfyui-Kandinsky-T8
python -m pip install -r requirements.txt
```

使用 ComfyUI 自身的 Python 环境安装，然后重启。若已安装原版 Kandinsky 6 扩展，请将其禁用，避免节点重名。

## 模型

从 [t8star/Kandinsky-Comfy](https://huggingface.co/t8star/Kandinsky-Comfy) 下载，将所选 DiT 放入 `ComfyUI/models/diffusion_models`：

| 模型 | 文件 | 大小 | 采样设置 |
|---|---|---:|---|
| Lite | `k6_lite_int8_convrot.safetensors` | 4.87 GB | KSampler，50 步，CFG 5 |
| Lite 蒸馏版 | `k6_lite_distill_int8_convrot.safetensors` | 3.77 GB | PiFlow，10 步，CFG 1 |
| Pro 蒸馏版 | `k6_pro_distill_int8_convrot.safetensors` | 34.97 GB | PiFlow，10 步，CFG 1 |

每个文件均包含完整的联合音视频 DiT。文本编码器与解码模型保持独立：

```text
ComfyUI/models/
  text_encoders/qwen_2.5_vl_7b.safetensors
  text_encoders/clip_l.safetensors
  vae/hunyuan_video_vae_bf16.safetensors
  audio_vae/v1-44.pth
  audio_vae/bigvgan_vocoder/config.json
  audio_vae/bigvgan_vocoder/bigvgan_generator.pt
```

工作流中的 **Download models** 按钮从固定版本的上游仓库下载这些共享组件，并复用已有文件；DiT 单独下载。支持 `extra_model_paths.yaml`。

## 使用

导入 [Base 或 PiFlow 工作流](example_workflows)，在 **UNETLoader** 中选择对应模型，保持 `weight_dtype=default`。Lite 使用 Base；两个蒸馏版使用 PiFlow。按输入方式选择 Text 或 Image 工作流。

默认输出 **864×480、121 帧、24 fps**，生成带音轨的 MP4。宽高需为 16 的倍数，帧数需为 `4*n+1`；PiFlow 使用 CFG=1、denoise=1。

接入官方 **UNETLoader / ModelPatcher**、**DualCLIPLoader（`kandinsky5`）**、**VAELoader**、latent/conditioning 接口及显存卸载机制，无需修改 ComfyUI 核心。Lite 使用官方 KSampler；蒸馏版通过 Kandinsky6Sampler 执行专用 PiFlow 采样。

## 验证与转换

三个模型均在 **RTX 5090 Laptop 24 GB / 64 GB RAM** 上完成完整文生与图生流程，六段 INT8 输出均通过音视频回读检查。Pro 蒸馏版依赖动态显存与卸载；模型文件大小不代表峰值显存。

[验证记录](VALIDATION.md)包含耗时与测试证据。[技术说明](docs/technical-notes.zh-CN.md)包含量化范围、可选原始模型下载及 `tools/convert_int8.py` 用法。模型仓库提供 SHA-256 校验与转换清单。

## 致谢与许可证

基于 [Kandinsky Lab 官方扩展](https://github.com/kandinskylab/kandinsky-6)，版本 `bf676db9b1335d22ebca85089355732687de3d0a`。节点代码与 Kandinsky DiT 权重采用 MIT，保留原始授权声明。共享组件遵循各自许可证，其中 TOD 音频解码器为 **CC BY-NC 4.0**；来源与条款见[模型卡](https://huggingface.co/t8star/Kandinsky-Comfy)。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com)

[在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
