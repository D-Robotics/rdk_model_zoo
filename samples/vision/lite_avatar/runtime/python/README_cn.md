[English](README.md) | 简体中文

# LiteAvatar Python 运行时

[样例](../../README_cn.md) · [模型产物](../../model/README_cn.md) · [转换](../../conversion/README_cn.md)

<a id="overview"></a>
## 概述

面向 S100、S100P 与 S600 的离线音频驱动数字人渲染:一个 Python
运行时将四个已发布的 HBM 产物(音频编码器、audio2mouth、人脸编码器、
人脸解码器)串联为整段 `predict`,并把帧交付给视频编码器。视频 I/O
全部走 VPU:预处理背景由硬件 H.264 `Decoder` 解码,每个输出帧由工作
进程中的硬件 H.264 `Encoder` 编码,中间的缩放、色彩转换与合成由
NumPy/Pillow/SciPy 完成。FFmpeg 以 `-c:v copy` 把基本流封装为 MP4
并混入 AAC 音频;随后 `ffprobe` 校验交付文件确实同时持有两条轨道,
运行才会报告成功。

| 目标平台 | 语言 | 背景解码 | 视频编码 | 状态 |
| --- | --- | --- | --- | --- |
| s100 | python | VPU H.264(文件模式) | VPU H.264 工作进程 | 支持 |
| s100p | python | VPU H.264(文件模式) | VPU H.264 工作进程 | 支持 |
| s600 | python | VPU H.264(文件模式) | VPU H.264 工作进程 | 支持 |

线程策略(实测板端配置):入口在首个重导入之前导出
`OPENBLAS_NUM_THREADS=1` 与 `OMP_NUM_THREADS=4`(显式导出的环境变量
优先),并固定 `torch.set_num_threads(2)`;这是限制 CPU 线程数量,
不设置 CPU 亲和性。仅导入流水线 API 不会改变任何进程级设置。

<a id="directory"></a>
## 目录结构

```text
runtime/python/
├── README.md         # 英文说明
├── README_cn.md      # 中文说明
├── main.py           # 入口:解析 -> 选择 -> 构造 -> predict -> 交付
├── cli.py            # 选项、已发布选择、硬件写视频器、交付
├── avatar.py         # 阶段、流水线、渲染器、张量契约
├── frontend.py       # 固定的 Paraformer 音频前端(fbank、LFR、CMVN)
├── media_worker.py   # VPU 解码辅助 + 硬件 H.264 编码工作进程
└── run.sh            # 启动脚本
```

<a id="environment"></a>
## 环境

板端执行需要板卡镜像中的 `hbm_runtime` 与 `hobot_vio` SDK 模块、
板端 Python 栈与 `ffmpeg`(含 `ffprobe`):

```bash
# 在板端(镜像 Python:S100/S100P 为 3.10,S600 为 3.12;aarch64):
# 在用户数据盘上建一个独立 venv,固定到已验证版本
# (PyPI 标准 aarch64 轮子)
python3 -m venv --system-site-packages /mnt/sdcard/lite-avatar-venv
/mnt/sdcard/lite-avatar-venv/bin/pip install \
    torch==2.3.1 torchaudio==2.3.1 numpy==1.26.4 \
    scipy soundfile pillow
. /mnt/sdcard/lite-avatar-venv/bin/activate
# (--system-site-packages 保持镜像 SDK 模块 hbm_runtime 与
#  hobot_vio 可导入)
```

视频输出需要 PATH 上有 `ffmpeg`(含 `ffprobe`):它写出带 AAC 音频的
MP4 容器,封装后由 `ffprobe` 校验两条交付轨道。`--no-encode` 则不
需要。

渲染器的背景缓存通过 `ctypes` 库发现加载系统 liblz4:板卡镜像自带
(已在 S100 验证),Ubuntu 主机由 `liblz4-1` 包提供,macOS 主机需要
`brew install lz4`(python.org 解释器还需要
`DYLD_LIBRARY_PATH="$(brew --prefix lz4)/lib"`)。

x86 导出/校准主机使用另一套配方(`torch==2.3.1+cpu` 轮子):见
[转换](../../conversion/README_cn.md#toolchain-targets)。`--help`、
`--list-models` 与 `--dry-run` 可在任何工作站运行,无需板端包。

<a id="background-prep"></a>
## 背景资产预处理(一次性,在主机上)

数字人数据自带的 `bg_video.mp4` 为 MPEG-4 Part 2,RDK 视频解码器
不支持。每个数字人只需在任意有 FFmpeg 的主机上运行一次预处理脚本:
脚本只读源 `.mp4`,转码出 `bg_video.h264`,实测生成的码流(帧数、
几何信息、SHA-256)并在旁边写出配套的 `bg_video.json` sidecar:

```bash
# 在主机上
python3 samples/vision/lite_avatar/test_data/prepare_background.py \
    --data samples/vision/lite_avatar/test_data/preload
```

随后把 `bg_video.h264` 与 `bg_video.json` 拷入板上的数字人数据目录。
运行时在解码前逐字段校验 sidecar(`codec`、可见 `width`、
`coded_width`、`height`、`fps`、`frames`、`sha256`),SHA-256 不匹配
即拒绝。可见宽度补齐到 16 像素倍数以匹配解码器的 NV12 亮度步长
(自带资产 890 补到 896);写 MP4 时补齐列会被裁掉。

<a id="usage"></a>
## 使用

在仓库根目录运行:

```bash
# 不依赖 SDK 的产物清单
bash samples/vision/lite_avatar/runtime/python/run.sh --list-models

# 不加载 SDK 即可解析四模型选择
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 --dry-run

# 按已验证默认值完整渲染
# 输入:预处理产物 + 数字人数据;输出:outputs/lite_avatar/avatar.mp4
# 成功:退出码 0 并打印 "Video written" 摘要行
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 \
    --audio samples/vision/lite_avatar/test_data/asr_example.wav \
    --avatar-data /path/to/prepared-avatar-data \
    --output outputs/lite_avatar/avatar.mp4
```

退出码:`0` 成功,`1` dry-run 发现产物缺失,`2` 用法/校验/运行时
错误(完整运行缺产物时也会列出需要准备的确切路径)。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | 选择 | auto | 执行目标:`auto`、`x5`、`s100`、`s100p`、`s600`;LiteAvatar 发布 s100/s100p/s600 三套,目标必须与检测到的硬件一致 |
| `--list-models` | 开关 | false | 无需板端访问即可列出清单背书的产物引用 |
| `--dry-run` | flag | false | 不加载 SDK 即解析/检查四模型选择 |
| `--audio` | 字符串 | null | 输入 WAV(任意采样率;立体声取第一声道) |
| `--avatar-data` | 字符串 | samples/vision/lite_avatar/test_data/preload | 数字人数据目录(`bg_video.h264` + `bg_video.json`、`face_box.txt`、`ref_frames/`) |
| `--cmvn-path` | 字符串 | samples/vision/lite_avatar/model/am.mvn | 音频前端 CMVN 文件(固定 `am.mvn`) |
| `--output` | 字符串 | outputs/lite_avatar/avatar.mp4 | 输出视频路径 |
| `--encoder` | 选择 | auto | `auto` 或 `hardware`(VPU H.264) |
| `--no-encode` | 开关 | false | 完全跳过视频写入(仅渲染) |
| `--model-dir` | 字符串 | null | 存放清单相对模型路径的目录 |
| `--models-manifest` | 字符串 | null | 标识本地重建产物的 JSON `{filename: sha256}` 清单 |
| `--priority` | 整数 | null | 运行时调度优先级(0-255) |
| `--bpu-cores` | 整数列表 | null | 运行时 BPU 核编号 |

以上默认值即 `build_parser`([cli.py](cli.py))中的定义。
`--priority`/`--bpu-cores` 为全部四个模型设置 BPU 调度;请求在每个
模型首次推理前生效,并在每次重载时恢复。

<a id="results"></a>
## 结果

运行会打印音频/渲染各阶段耗时、渲染速率(捆绑示例为
`Rendered 166 frames ... in ...s (... fps)`)与
`Video written: <output> (encoder: hardware)`,随后退出码 0。输出是
H.264 High 的 MP4 —— 捆绑示例为 890x1920、30 fps、有限范围
BT.601、166 帧 —— 输入音频以 AAC 混入。

捆绑示例以普通入口实测(torch 2.3.1,所有目标硬件编码):

| 目标 | 渲染速率(166 帧窗口) | 端到端耗时 |
| --- | --- | --- |
| s100p | 18.7 fps | 24.7 秒 |
| s600 | 20.5 fps | 28.3 秒 |
| s100 | 15.3 fps | 28.6 秒 |

渲染速率窗口覆盖全部 166 帧渲染加硬件编码器排空与 MP4 封装;此前的
背景硬件解码、音频前端与渲染器初始化等一次性阶段不在窗口内。端到端
耗时为完整 CLI 运行,含初始化。

帧级输出与参考渲染的对比见
[evaluator/evaluate_frames.py](../../evaluator/evaluate_frames.py)。

<a id="integration-example"></a>
## 集成示例

前提:产物已准备(见 [model/README_cn.md](../../model/README_cn.md))
且上述板端包可导入。示例中每个输入变量都有定义:

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
for frame in result.frames():   # uint8 BGR,每个 30 fps 输出帧一个
    ...
```

`result.render_frame(i)` 与 `result.frames()` 返回自持帧。会话语义:
`predict` 运行音频各阶段后释放音频模型(下次 `predict` 按恢复的调度
重载),并惰性构建渲染器。背景仅硬件解码一次并以压缩态持有;解码出
的帧列表在参考金字塔加载前即已释放,峰值内存因此有界;参考金字塔每
份只编码一次并可跨 predict 复用;解码器逐渲染帧加载。

<a id="stage-io"></a>
## 阶段 I/O

流水线为四个命名阶段,每个阶段在 [avatar.py](avatar.py) 内都有可读
的 `preprocess → infer → postprocess` 链;`predict` 恰好按这些步骤
串联。

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| 音频前端(CPU,Torch) | 波形 `float32` 单声道,任意采样率 | LFR 堆叠、CMVN 归一化的 fbank 特征 `float32 [1,400,560]`(每 4 秒一块) |
| `audio_encoder` | 特征 `[1,400,560]` | 隐状态 `[1,400,50,512]`(50 路堆叠) |
| `audio2mouth` | 滑窗 `[1,30,50,512]` | 嘴部参数 `[1,30,32]` + 特征;窗口交叉淡化并经 Butterworth 平滑为一条 `float32 [N,32]` 曲线 |
| `facegen_encoder`(逐参考帧) | RGB `[1,3,384,384]` | 特征金字塔 `f0..f3`(每份参考只编码一次,复用) |
| `facegen_decoder`(逐输出帧) | 金字塔 + 嘴部参数 `[1,32]` | 嘴部图像 `[1,3,384,384]`,按羽化掩膜与乒乓环形顺序合成到背景帧上 |

所有 HBM 阶段在运行时边界暴露 FLOAT32 张量;INT16 量化内置于产物。

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 检查 |
| --- | --- |
| `Missing model artifacts`(退出码 2) | 按 [model/README_cn.md](../../model/README_cn.md) 准备 `model/<target>/` 下的四个文件;运行 `model/verify.py --target <t>`。 |
| `--dry-run` 退出码 1 | 同样的产物缺口;打印的路径就是缺失的确切文件。 |
| 目标选择错误 | LiteAvatar 每个目标(`s100`、`s100p`、`s600`)发布一套四文件;目标必须与检测到的板卡一致(或用 `auto`),其他目标显式报错。 |
| `No prepared background ... only a bg_video.mp4 file is present` | 数字人数据还没有预处理背景;执行上文一次性主机预处理,把 `bg_video.h264` + `bg_video.json` 放到数据旁边。 |
| `bg_video.h264 SHA-256 ... does not match the sidecar` | 码流与 sidecar 来自不同的转码(拷贝不完整/被改动);重跑主机预处理,让两文件出自同一次转码。 |
| `FFmpeg is required for video output (not on PATH)` | 最终封装需要 `ffmpeg`;安装板卡镜像的 ffmpeg 包,或改用 `--no-encode`。 |
| `Hardware decoder close failed ...` / `Hardware encoder worker exited ... (exit N)` | VPU 报告了清理/协议失败;运行会中止并给出工作进程日志路径 —— 检查 `dmesg`/日志中的编解码设备状态后重试。 |
| 输出与参考渲染不一致 | 先对齐产物、音频、数字人数据与目标平台再评判质量;用 [evaluator/evaluate_frames.py](../../evaluator/evaluate_frames.py) 对比参考帧。 |
