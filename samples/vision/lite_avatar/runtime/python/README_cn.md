[English](README.md) | 简体中文

# LiteAvatar Python 运行时

[示例](../../README_cn.md) · [模型产物](../../model/README_cn.md) · [转换](../../conversion/README_cn.md)

<a id="overview"></a>
## 概述

面向 S100、S100P 与 S600 的离线音频驱动数字人渲染：一个 Python 运行时把
四个已发布 HBM 产物（音频编码器、audio2mouth、人脸编码器、人脸解码器）
串成整段音频的 `predict`，并把帧交给视频写入器。`main.py` 保持流程可见——解析 → 选择 →
构建 → 预测 → 交付；写入器对所有资源（FFmpeg 管道、带共享内存环的 S100
硬件编码 worker）设上界并在失败时清理。

| 目标 | 语言 | 视频编码器 | 状态 |
| --- | --- | --- | --- |
| s100 | python | S100 硬件 H.264 worker 或 FFmpeg 管道 | supported |
| s100p | python | FFmpeg 管道 | supported |
| s600 | python | FFmpeg 管道 | supported |

线程策略（实测板端配置）：入口在首个重导入之前导出
`OPENBLAS_NUM_THREADS=1` 与 `OMP_NUM_THREADS=4`（显式导出的环境优先），
并固定 `torch.set_num_threads(2)` / `cv2.setNumThreads(2)`，使 Python
渲染、BPU 运行时与编码 worker 各自让开核心。仅导入流水线 API 不会改变
任何进程级设置。

<a id="directory"></a>
## 目录结构

```text
runtime/python/
├── README.md         # 英文说明
├── README_cn.md      # 中文说明
├── main.py           # 入口：解析 -> 选择 -> 构建 -> 预测 -> 交付
├── cli.py            # 选项、发布选择、视频编码器、交付
├── avatar.py         # 阶段、流水线、渲染器、张量契约
├── frontend.py       # 固定的 Paraformer 音频前端（fbank、LFR、CMVN）
├── encoder_worker.py # S100 硬件 H.264 编码 worker（环/ACK 协议）
└── run.sh            # 启动器
```

<a id="environment"></a>
## 环境要求

上板运行需要板端镜像提供的 `hbm_runtime` SDK、板端 Python 栈与
`ffmpeg`：

```bash
# 板上执行（镜像 Python：S100/S100P 为 3.10，S600 为 3.12；aarch64）：
# 在用户数据盘上建一个受限 venv，固定为已验证版本（PyPI 标准 aarch64 轮子）
python3 -m venv --system-site-packages /mnt/sdcard/lite-avatar-venv
/mnt/sdcard/lite-avatar-venv/bin/pip install \
    torch==2.3.1 torchaudio==2.3.1 numpy==1.26.4 \
    opencv-python==4.11.0.86 scipy soundfile pyyaml
. /mnt/sdcard/lite-avatar-venv/bin/activate
# （--system-site-packages 保持镜像 SDK（hbm_runtime）可导入）
```

PATH 上的 `ffmpeg` 对**两种**编码器的视频输出都是必需的——软件写入器
把原始帧管道送入 ffmpeg libx264 进程，硬件路径的捆绑
`encoder_worker.py` 同样经由 ffmpeg 把 H.264 码流与音频混流；
`--no-encode` 不需要它。

渲染器的背景缓存经 `ctypes` 库发现加载系统 liblz4：板端镜像自带
（S100 已验证），Ubuntu 主机由 `liblz4-1` 包提供，macOS 主机
`brew install lz4`（python.org 解释器还需
`DYLD_LIBRARY_PATH="$(brew --prefix lz4)/lib"`）。

x86 导出/校准主机使用另一套配方（`torch==2.3.1+cpu` 轮子），见
[转换文档](../../conversion/README_cn.md#toolchain-targets)。`--help`、
`--list-models` 与 `--dry-run` 在任意工作站上无需板端包即可运行。

<a id="usage"></a>
## 用法

在仓库根目录运行：

```bash
# 不依赖 SDK 的产物清单
bash samples/vision/lite_avatar/runtime/python/run.sh --list-models

# 不加载 SDK 解析四模型选择
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 --dry-run

# 使用已验证默认值的完整渲染（S100 上为硬件编码器）
# 输入：准备好的产物与数字人数据；输出：outputs/lite_avatar/avatar.mp4
# 成功：退出码 0 且出现 "Video written" 汇总行
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 \
    --audio samples/vision/lite_avatar/test_data/asr_example.wav \
    --output outputs/lite_avatar/avatar.mp4
```

退出码：`0` 成功、`1` dry-run 发现产物缺失、`2` 用法/校验/运行时错误
（含完整运行时产物缺失，此时会列出需要准备的确切路径）。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | auto | 执行目标：`auto`、`x5`、`s100`、`s100p`、`s600`；LiteAvatar 发布 s100/s100p/s600 三套产物，须与检测到的硬件一致 |
| `--list-models` | flag | false | 不上板列出清单支持的产物引用 |
| `--dry-run` | flag | false | 不加载 SDK 解析/校验四模型选择 |
| `--audio` | string | null | 输入 WAV（任意采样率；立体声取第一声道） |
| `--avatar-data` | string | samples/vision/lite_avatar/test_data/preload | 数字人数据目录（`bg_video.mp4`、`face_box.txt`、`ref_frames/`） |
| `--cmvn-path` | string | samples/vision/lite_avatar/model/am.mvn | 音频前端的 CMVN 文件（固定 `am.mvn`） |
| `--output` | string | outputs/lite_avatar/avatar.mp4 | 输出视频路径 |
| `--encoder` | choice | auto | `auto`（S100 硬件，其余软件）、`software`（FFmpeg 管道）或 `hardware`（S100 H.264 worker） |
| `--no-encode` | flag | false | 完全跳过视频写盘（仅渲染） |
| `--model-dir` | string | null | 存放清单相对模型路径的目录 |
| `--models-manifest` | string | null | 标识本地重编译产物的 JSON `{filename: sha256}` 清单 |
| `--priority` | int | null | 运行时调度优先级（0-255） |
| `--bpu-cores` | int list | null | 运行时 BPU 核心索引 |

以上默认值即 `build_parser`（[cli.py](cli.py)）中的定义。
`--priority`/`--bpu-cores` 为四个模型设置 BPU 调度；请求会在每个模型首次
推理前生效，并在每次重载时恢复。

<a id="results"></a>
## 结果

运行会打印音频/渲染各阶段耗时、渲染速率（捆绑示例为
`Rendered 166 frames ... in ...s (... fps)`）与 `Video written: <输出路径>
(encoder: hardware|software)`，随后退出码 0。输出为可解码的 H.264 MP4：
背景循环帧与动画嘴部 ROI 的数字人几何，30 fps，输入音频已混流。
`--no-encode` 时不写文件，仅打印渲染汇总。

捆绑示例（166 帧，890x1920）实测：S100 硬件编码渲染 22.0 fps；S100P 软件
编码渲染 **36.4 fps**（端到端 13.2 秒）、`--no-encode` 38.0 fps，软件编码
MP4 回解码面部平均 PSNR 38.7 dB；S600 软件编码渲染 **39.4 fps**、
`--no-encode` 40.9 fps，回解码面部平均 PSNR 38.7 dB。

<a id="integration-example"></a>
## 集成示例

前提：产物已按[模型文档](../../model/README_cn.md)准备，且上述板端包可导入。
示例中的每个输入变量都有定义：

```python
from samples.vision.lite_avatar.runtime.python.avatar import (
    AvatarInput, LiteAvatarPipeline,
)
from samples.vision.lite_avatar.runtime.python.cli import (
    read_audio, resolve_selections,
)

selections = resolve_selections("s100")
pipeline = LiteAvatarPipeline.from_models(
    selections,
    cmvn_path="samples/vision/lite_avatar/model/am.mvn",
    data_dir="samples/vision/lite_avatar/test_data/preload",
)
waveform, sample_rate = read_audio(
    "samples/vision/lite_avatar/test_data/asr_example.wav")
result = pipeline.predict(AvatarInput(waveform, sample_rate))
for frame in result.frames():   # uint8 BGR，30 fps 每帧一个
    ...
```

`result.render_frame(i)` 与 `result.frames()` 返回自有帧；合成算术与原
流水线保持一致。会话语义：`predict` 运行音频阶段并释放音频模型（下次
`predict` 时重载并恢复调度），渲染器惰性构建（参考特征金字塔只编码一次、
跨 predict 复用；解码器按渲染帧加载）。

<a id="stage-io"></a>
## 阶段输入输出

流水线为四个具名阶段，各自的 `preprocess → infer → postprocess` 链在
[avatar.py](avatar.py) 中清晰可读；`predict` 串联的正是这些步骤。

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| 音频前端（CPU，Torch） | `float32` 单声道波形，任意采样率 | 每 4 秒块的 LFR 堆叠、CMVN 归一化 fbank 特征 `float32 [1,400,560]` |
| `audio_encoder` | 特征 `[1,400,560]` | 隐状态 `[1,400,50,512]`（50 路堆叠） |
| `audio2mouth` | 滑动窗口 `[1,30,50,512]` | 嘴部参数 `[1,30,32]` + feat；窗口交叉淡化并 Butterworth 平滑成一条 `float32 [N,32]` 曲线 |
| `facegen_encoder`（每个参考帧一次） | RGB `[1,3,384,384]` | 特征金字塔 `f0..f3`（每个参考只编码一次并复用） |
| `facegen_decoder`（每个输出帧） | 金字塔 + 嘴部参数 `[1,32]` | 嘴部图像 `[1,3,384,384]`，按羽化掩码与乒乓环顺序合成到背景帧上 |

所有 HBM 阶段在运行时边界都是 FLOAT32 张量；INT16 量化在产物内部。

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 检查 |
| --- | --- |
| `Missing model artifacts`（退出码 2） | 按[模型文档](../../model/README_cn.md)准备 `model/<目标>/` 下四个文件；运行 `model/verify.py --target <t>`。 |
| `--dry-run` 退出码 1 | 同样的准备缺口；打印的路径即缺失文件。 |
| 目标选择错误 | LiteAvatar 每个目标各发布一套四文件产物（`s100`、`s100p`、`s600`）；目标须与检测到的板卡一致（或用 `auto`），其他目标显式报错。 |
| `FFmpeg is required for video output (not on PATH)` | **两种**编码器写视频都要求 PATH 上有 `ffmpeg`（硬件路径同样要经它混流）；安装板端镜像的 ffmpeg 软件包，或改用 `--no-encode`。 |
| `Hardware encoder worker exited ... (exit N)` | 捆绑的 worker 在协议中途崩溃；运行会带 worker 日志路径中止——检查 `dmesg`/日志中的编码器设备状态后重试。 |
| 输出与参考渲染不同 | 先对齐产物、音频、数字人数据与编码器再判断质量；用 [evaluator/evaluate_frames.py](../../evaluator/evaluate_frames.py) 对比参考帧。 |
