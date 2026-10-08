# 运行证据

2026-10-08 验证完成。基线：ComfyUI `d91ed5f5b7fa60fa18464c2ad7c80254da2f0f29` / 0.39.0，comfy-kitchen 0.2.37，Torch 2.14.1+cu130，Windows，RTX 5090 Laptop 24 GB、64 GB RAM。ComfyUI 核心 checkout 保持干净；模型扩展通过原生注册、ModelPatcher 和动态显存接口接入。

| 检查 | 当前结果 |
|---|---|
| Kitchen ConvRot 64 / 256 CUDA kernels | 通过，真实量化、反量化与 Linear |
| 转换测试 | 14 通过，含并发提交、manifest 回滚及源文件别名保护 |
| 新工具 lint / Python 编译 | 通过 |
| Lite 单文件转换 | 4,868,860,174 bytes，920 INT8 layers；全部保留张量原始字节通过 |
| Lite distill 单文件转换 | 3,773,931,918 bytes，920 INT8 layers；全部保留张量原始字节通过 |
| Pro distill 单文件转换 | 34,970,473,976 bytes，1728 INT8 layers（全部 256 分组）；形状、dtype、正有限 scale、旋转标记、全部保留张量原始字节与整文件 SHA 校验通过 |
| 源模型与独立组件 | 9 个文件全部下载并验证；源模型 LFS SHA-256 一致 |
| Lite 原生 ModelPatcher 双分支前向 | 通过，视频与音频均为有限值 |
| Lite distill 原生 ModelPatcher 双分支前向 | 通过，10 网格与参考尾帧均可前向 |
| KSampler，256×256，17 帧，2 steps，CFG=5 | 通过，包含不同长度正负条件（32 / 15） |
| Lite PiFlow，256×256，17 帧，2 segments | 通过 |
| I2AV 干净参考保存 | KSampler 最大误差 9.54e-7；PiFlow 最大误差 2.38e-7（原生 latent 缩放往返） |
| 原始 FP32 加载 | 普通 Lite 691 个参数、Lite distill 423 个参数，均与源文件逐项一致；offload dtype 同步保留 |
| 官方 LoRA loader / ModelPatcher | 真实补丁生效，INT8 ConvRot 64 分组保留，Linear 输出有限 |
| ComfyUI GUI 模板 | 已识别四个模板；GUI 导出的 API 图与测试图的节点、参数、连线一致 |
| Lite distill 默认尺寸 T2AV | 成功，163.12 秒，864×480 / 121 帧 / 24 fps，H.264 + AAC |
| Lite distill 默认尺寸 I2AV | 成功，136.39 秒，864×480 / 121 帧 / 24 fps，H.264 + AAC |
| 普通 Lite 默认尺寸 T2AV | 成功，1057.68 秒，50 steps / CFG=5，864×480 / 121 帧 / 24 fps，H.264 + AAC |
| 普通 Lite 默认尺寸 I2AV | 成功，1075.14 秒，50 steps / CFG=5，864×480 / 121 帧 / 24 fps，H.264 + AAC |
| 六段 INT8 完整媒体回读 | 121 个可解码帧，44.1 kHz / 222338 个音频样本，有限且非静音，时长差 < 一个采样点；代表帧已查看 |
| 原始 Lite distill 完整视频对照 | 成功，192.42 秒；与 INT8 使用相同节点、参数、提示词、种子和连线（仅模型文件与输出前缀不同），两段媒体均通过回读 |
| Pro distill 默认尺寸 T2AV | 成功，745.43 秒，864×480 / 121 帧 / 24 fps，H.264 + AAC；121 帧和完整音轨回读通过，代表帧已查看 |
| Pro distill 默认尺寸 I2AV | 成功，945.40 秒，864×480 / 121 帧 / 24 fps，H.264 + AAC；121 帧和完整音轨回读通过，参考人物与转头代表帧正常 |

上述六条 INT8 完整流程与一条未量化对照全部返回 ComfyUI `execution_success`。Pro 使用原生动态显存运行，采样期间观察到约 18–20 GB 显存使用；该值来自采样快照，不是严格峰值测量。声音检查覆盖有限值、RMS、削顶和时长，未据此宣称已人工听审或验证口型同步。

同输入、同随机数、同原生前向比较 Lite distill 原始权重与 INT8（两者均保留 FP32 参数）：视频相对 L2=0.009784、cosine=0.999887；音频相对 L2=0.034142、cosine=0.999425。该检查采用小尺寸随机 context，验证数值差异，不能代替真实提示词的完整音视频质量评估。

完整视频对照已另外生成并查看 0 / 2.5 / 5 秒代表帧。两版均有清晰的双鹦鹉与森林场景，姿态与动作存在差异。121 帧 RGB 输出差异 PSNR=21.25 dB；该值用于记录输出差异，不是感知质量分数。单个提示词对照不能保证所有题材无质量损失。

节点包 `evidence/summary.json` 汇总七条完整运行；`evidence/full_runs` 为 ComfyUI 实际回执，`evidence/media` 为媒体回读结果，`evidence/qa` 为代表帧，`evidence/conversion` 为三个转换 manifest。独立组件与源模型的仓库、固定 revision、字节数和 SHA-256 位于 `MODEL_SOURCES.json`。完整日志、视频和原始权重保留在开发工作区 `logs`、`.research/ComfyUI/output`、`models`。

本次发布与验证覆盖 Lite、Lite 蒸馏版和 Pro 蒸馏版。普通 Pro 未完成下载，按交付范围不纳入发布，也未进行运行验证。未对其他硬件、所有题材、外部 ControlNet 或 PiFlow 的任意 guider/hook 作兼容保证。

2026-10-08，`0.1.1` 完成另一组实际 20 轮子 Agent 交叉检查，逐轮结论保存在 [evidence/crosscheck_20.json](evidence/crosscheck_20.json)。修复 10 类问题：工作流下载按钮、损坏组件复用、错误哈希 partial 的重试、移除队列后的等待、完整 AV checkpoint 结构检查、临时路径覆盖源文件、PiFlow 条件元数据、异常清理、AV 交叉注意力选项传递，以及中断后的 MagCache 生命周期。

新增针对性回归与原有转换测试合计 **43 项通过**；前端按钮绑定及 NDJSON 断流测试通过。真实原生检测器验证三份源模型与三份 INT8 模型均兼容；Lite KSampler 与 Lite 蒸馏 PiFlow 另外重跑 `256×256 / 17 帧 / 2 steps` I2AV，两路输出均为有限值，参考最大误差分别为 `9.54e-7 / 2.38e-7`。可选 MagCache 生命周期使用真实 CPU ModelPatcher 验证；未为此下载普通 Pro。本次没有重新量化或修改三份已发布权重，也未重复此前六条默认尺寸完整媒体运行。

`0.1.2` 完成新一组实际 20 轮交叉检查，记录位于 [evidence/crosscheck_second_20.json](evidence/crosscheck_second_20.json)。修复模型目录优先级、下载取消及文件移走的竞争、超长断点与重复下载参数、转换并发与失败回滚、INT8 参数完整性、PiFlow 遮罩与空音频条件，以及示例输入清理失败和转换源硬链接保护；工作流备注明确三个交付模型，下载入口指向已发布 INT8。

完整回归 **77 项通过，无跳过**；前端测试、四套模板的 72 条原生连线及 Python 3.10 AST 检查通过。主项目 Ruff F 检查通过，排除两个既有 vendor 导出文件中的四条 F401/F403 诊断。三份已发布 INT8 的全部 marker/scale 实值通过校验；另用真实原生 Linear、LoRA、ModelPatcher 和 TOD/BigVGAN 音频解码复核，标准音频节点和专用时长裁剪均正常。三份 DiT 和六个独立组件保持不变，未重新量化，也未重跑此前六条默认尺寸完整媒体。

最终源码再次跑通 Lite 蒸馏 `256×256 / 17 帧 / 2 steps` 原生 I2AV，视频、音频均有限，参考误差 `2.38e-7`；重启实际服务器后，六个组件经 SHA-256 校验全部复用，没有新下载。

回归命令（使用已安装依赖的 Python；CUDA 检查需要实际 CUDA 环境）：

```bash
python -m pytest --import-mode=importlib --confcutdir=tests tests -q
node tests/test_downloads.mjs
```

`0.1.3` 的新一组交叉检查记录见 [evidence/crosscheck_third_20.json](evidence/crosscheck_third_20.json)。修复转换和下载的原子禁止覆盖、并发断点损坏、回执提交及外部文件回滚保护、最低版本锁文件别名截断、临时目录清理异常，以及 INT8 单值 scale、存储 dtype 和旋转标记校验。错误编码的下载确认请求返回 400；Hugging Face 双语下载示例改用七个明确文件参数，兼容最低与当前 CLI。

完整回归 **109 项通过，无跳过**。原生快速 Linear、模型保存重载、LoRA、FP32 参数归档、预取中断清理和音频接口检查通过；PiFlow 的 1/2/10 步数学结果与固定上游一致。四套 GUI/API 模板的 72 条连线有效，前端下载状态与按钮检查通过。最低 `filelock 3.13.1` 下的模型、断点与回执硬链接保护已独立复验。转换回滚先原子捕获目标文件，再检查身份并禁止覆盖地恢复；Windows rename 与 POSIX hard-link 分支共 12 个真实小文件竞争场景通过复验。

最新源码重跑 Lite 蒸馏原生 I2AV：`256×256 / 17 帧 / 2 steps`，视频、音频潜变量均有限，参考帧误差 `2.38e-7`，采样耗时 `24.219` 秒。重新启动服务器后，六个组件经 SHA-256 校验全部复用，没有新下载。Hugging Face 仅更新两份模型卡；三份 DiT 与六个独立组件保持不变。本轮未重新量化或重复六条默认尺寸完整媒体运行。

`0.1.4` 增加六份已选好模型的 GUI 工作流下载包，并将图生示例图安装到 `ComfyUI/input` 根目录，兼容官方 LoadImage 的文件枚举与新版前端的媒体检查。已有用户图片不会被覆盖。六份文件均在实际 ComfyUI 前端 `1.55.14` 逐份打开并导出 API 图，模型、参数与 108 条连线和原先六条成功记录一致；图生路径改为同 SHA-256 的根目录示例图。

本次另外从实际 GUI 点击运行完整 Lite 蒸馏图生流程，`864×480 / 121 帧 / 24 fps / 10 steps`，耗时 **157.632 秒**，返回 `execution_success`。新生成 MP4 的 121 帧、H.264 视频及 44.1 kHz AAC 音轨全部回读通过，音频有限且非静音，代表帧已查看。完整回归再次 **109 项通过，无跳过**，四个前端下载按钮与断流检查通过。证据见 [GUI 工作流核对](evidence/gui_workflow_verification.json)和[本次界面运行回执](evidence/full_runs/lite_distill_i2av_gui_0.1.4_full.json)。另五条完整媒体沿用原始成功记录，本次没有重复生成，也没有修改模型权重。
