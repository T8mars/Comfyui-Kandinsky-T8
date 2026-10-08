# ComfyUI Kandinsky 6 INT8 ConvRot

基于 Kandinsky Lab 官方 ComfyUI 扩展的原生节点，增加完整联合音视频 DiT 的单文件 INT8 ConvRot 转换、Lite PiFlow 蒸馏支持，以及可直接运行的基础工作流。独立文本编码器、视频 VAE 和音频解码模型保留为独立文件。

代码基线：Kandinsky-6 `bf676db9b1335d22ebca85089355732687de3d0a`。保留原项目 MIT License 和 vendor 授权文件。发布项目为 ComfyUI Kandinsky T8，Publisher ID 为 `t8star`。

## 安装

需要 ComfyUI **0.39.0+**、comfy-kitchen **0.2.37+**，以及支持 INT8 Tensor Core 的 CUDA GPU。开发机器实际环境为 Python 3.12.14、Torch 2.14.1+cu130、RTX 5090 Laptop 24 GB。安装本包到 ComfyUI 的 `custom_nodes/ComfyUI-Kandinsky6-INT8`，在 ComfyUI 使用的 Python 中执行：

```powershell
python -m pip install -r requirements.txt
```

同时加载旧版官方 `kandinsky6` 扩展会造成节点名冲突；请只启用一个版本。本包在启动时注册模型检测分支，扩展原生 supported model 列表，未编辑 ComfyUI 核心源文件。它使用官方 MODEL / ModelPatcher、加载卸载、mixed precision ops、attention、prefetch 和 LATENT / CONDITIONING 契约。

安装与预转换权重下载见仓库根目录的 README。下面的下载与转换命令适用于需要自行转换原始权重的用户。

## 模型目录

```text
ComfyUI/models/
  diffusion_models/
    k6_lite_int8_convrot.safetensors
    k6_lite_distill_int8_convrot.safetensors
    k6_pro_distill_int8_convrot.safetensors
  text_encoders/
    qwen_2.5_vl_7b.safetensors
    clip_l.safetensors
  vae/
    hunyuan_video_vae_bf16.safetensors
  audio_vae/
    v1-44.pth
    bigvgan_vocoder/
      config.json
      bigvgan_generator.pt
```

也支持 `extra_model_paths.yaml` 中的 diffusion_models、text_encoders、vae、audio_vae。音频模型可放在子目录；BigVGAN 的配置和 generator 必须在同一目录。TE 与视频 VAE 使用官方 **DualCLIPLoader(type=kandinsky5)** 和 **VAELoader**。音频解码器返回原生 VAE / AUDIO，并由 Comfy 管理显存。

### 下载与转换

在本包目录中执行，将 `$modelsRoot` 改为实际模型目录。下载器固定模型 revision，支持断点续传，核对文件长度和 LFS SHA-256，保存下载回执。原始权重留在 `sources`，不会自动启动下载。

```powershell
$modelsRoot = 'D:\kandinsky-6\models'
python tools/download_models.py --models "$modelsRoot" --variants lite lite-distill pro-distill --components --workers 2
python tools/convert_int8.py "$modelsRoot\sources\lite.safetensors" "$modelsRoot\diffusion_models\k6_lite_int8_convrot.safetensors"
python tools/convert_int8.py "$modelsRoot\sources\lite-distill.safetensors" "$modelsRoot\diffusion_models\k6_lite_distill_int8_convrot.safetensors"
python tools/convert_int8.py "$modelsRoot\sources\pro-distill.safetensors" "$modelsRoot\diffusion_models\k6_pro_distill_int8_convrot.safetensors"
```

界面的 Download models 按钮只提供工作流所需的独立 TE/VAE/音频组件；DiT 需运行转换器。Pro 普通版属于 Hugging Face gated 仓库，需要账号获得模型访问权；下载授权不等于仓库访问权。当前 CLI 下载器针对公共文件；获得 Pro 普通版访问权后，可通过已登录的 Hugging Face 下载工具取得原始权重，再运行本包转换器。

### 量化范围

- 权重先做归一化分组 Hadamard 旋转，然后按输出通道量化为 INT8；推理时动态旋转和量化激活，运行 W8A8 Linear。
- K 维可整除时依次选择 256、64、16 分组；Lite 的 896 维音频分支使用 64。完整 Lite 有 920 个候选 Linear，完整 Pro 有 1728 个。
- 覆盖视频、音频和内部文本块的 attention Q/K/V/output 与 FFN in/out。保留时间嵌入、modulation、输入投影、最终输出头、norm、bias 和 token embeddings 的原始 dtype 与字节。
- 每层采用原生 `.weight` / `.weight_scale` / `.comfy_quant` schema；所有分支写入一个 safetensors，未拆分 DiT，未删除音频权重。
- 转换按行分块，先写临时文件，回读验证形状、dtype、旋转标记、正有限 scale，以及所有保留张量的原始字节，然后原子替换目标文件。完整 SHA-256 和层级记录写入相邻 `.manifest.json`。
- 默认采用确定性 absmax；不启用 MSE clipping。`--exclude-layer native.layer.weight` 可将指定层保留原始精度。`--dry-run` 只检查结构和输出预算。已有输出默认拒绝覆盖。

实际 Lite 普通版文件约 **4.87 GB**，Lite 蒸馏版约 **3.77 GB**，Pro 蒸馏版约 **34.97 GB**。24 GB 显卡加载 Pro 需要 Comfy 的卸载/动态显存能力，开发机器配有 64 GB RAM。文件大小并不等于峰值显存，视频激活、TE 和解码器还需要空间。

## 基础工作流

`example_workflows` 包含四个 GUI 工作流；对应 API 图在 `example_workflows/api`。旧版包含 SR/Beautifier 的官方示例保留在 `upstream_examples`，基础流程没有这些依赖。

| 模型 | 采样器 | steps / CFG / denoise | 音频 scale |
|---|---|---|---|
| Lite / Pro 普通版 | 官方 KSampler，Euler / simple | 50 / 5 / 1 | 0.5302 |
| Lite / Pro 蒸馏版 | Kandinsky6Sampler，PiFlow DX | 10 / 1 / 1 | 0.417 |
| pretrain | 官方 KSampler | 50 / 5 / 1 | 0.417 |

UNETLoader 的 `weight_dtype` 使用 **default**。PiFlow 是专用多网格积分，节点里的 sampler/scheduler 下拉项对蒸馏模型不生效。普通模型继承原生 flow shift=5；PiFlow 采用官方 shift=5、10 网格、128 子步预算和最后一段 0.5 步长。

默认 864×480、121 帧、24 fps、batch=1。宽高需为 16 的倍数，帧数需为 `4*n+1`。图生流程用官方 LoadImage / ImageScale / VAEEncode，追加干净参考尾帧，采样后用 RemoveReferenceLatent 移除，再用官方 VAEDecode。音频按生成帧数裁剪至相同时间长度。官方 CreateVideo / SaveVideo 输出带音轨的 MP4。

正负提示词的不同 token 长度分别求值；禁止重复较短上下文，因为模型内部文本块使用 RoPE。支持 Comfy 官方分块替换和 attention 选项，但没有声称未经验证的 ControlNet、区域提示、任意 LATENT 图像节点或 PiFlow 的任意 scheduler 兼容。

PiFlow 限制：CFG=1、denoise=1、单个联合条件；负提示不参与 CFG=1。支持中断与资源清理，遵循模型加载和 latent 缩放；专用 DX rollout 直接调用 DiT，不经过普通 KSampler 的全部 guider/hook 路径。MagCache 仅沿用官方普通 Pro 校准；基础工作流关闭缓存、compile 和额外注意力量化。

## 验证

```powershell
python -m pytest --confcutdir=tests -q
python tools/run_workflow.py "example_workflows/api/K6 INT8 PiFlow Text to Video+Audio.json" --url http://127.0.0.1:8188 --receipt validation/run.json
```

转换测试运行真实 Kitchen CUDA kernel，回读单文件并实际执行 Linear；覆盖 64/256 分组、分块边界、BF16/标量保留、敏感层选择和不完整模型拒绝。实际模型前向、KSampler / PiFlow、不同条件长度与参考帧验证已完成。

RTX 5090 Laptop 24 GB、64 GB RAM 上，Lite 普通版、Lite 蒸馏版、Pro 蒸馏版的 **六条完整 T2AV / I2AV 流程均成功**：864×480、121 帧、24 fps、H.264 + AAC。全部视频和音轨回读通过，并查看了代表帧。另有相同提示词和种子的未量化 Lite 蒸馏完整视频对照。GUI 导出的 API 图与测试图一致。详细结果见 `VALIDATION.md`；回执、转换 manifest 和 QA 代表帧随节点包保存在 `evidence`，源文件版本及校验信息见 `MODEL_SOURCES.json`。

本次发布覆盖 Lite、Lite 蒸馏版和 Pro 蒸馏版，普通 Pro 不在交付范围。以上性能与质量结果仅覆盖所列版本、硬件、默认尺寸和示例提示词。
