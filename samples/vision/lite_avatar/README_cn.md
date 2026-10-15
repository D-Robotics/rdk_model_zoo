[English](README.md) | 简体中文

# LiteAvatar 音频驱动数字人

在 RDK S100、S100P 与 S600 上离线渲染音频驱动数字人：输入语音音频，
输出合成好的数字人视频。

来源：[HumanAIGC/lite-avatar](https://github.com/HumanAIGC/lite-avatar)（MIT）

<a id="overview"></a>

## 概述

本示例面向 S100、S100P 与 S600 提供一个 Python 运行时。`LiteAvatarPipeline` 类把四个已
发布 HBM 产物串成每个阶段清晰可读的 `preprocess → infer → postprocess`
流程与一次整段音频的 `predict`：音频前端提取 Paraformer 特征（Kaldi
fbank、7/6 低帧率堆叠、CMVN），音频编码器输出 50 路堆叠隐层状态，
audio2mouth 把滑动的 30 帧窗口映射为嘴部参数（交叉淡化与
Butterworth 平滑），人脸解码器以参考帧特征金字塔为条件逐帧渲染。
每帧按羽化掩膜合成到背景循环帧上，交给板端 VPU 的 H.264 编码器；
FFmpeg 把输入音频以 AAC 混入 MP4，`ffprobe` 校验交付文件确实同时
持有两条轨道。完整 CLI 与 API 参考见
[运行时文档](runtime/python/README_cn.md)。

### 算法背景

LiteAvatar（HumanAIGC）用语音特征驱动轻量人脸生成网络：paraformer 编码器
的中间隐层状态携带发音信息，小型 audio2mouth 网络将其解码为每个视频帧
32 维嘴部参数；解码网络（InstanceNorm 风格模块与 x2 上采样）合成当前
姿态下的嘴部区域并融合进背景视频。按目标发布的四个产物全节点按配置 INT16 量化
（编译器仍会给个别算子分配其他精度）；解码器的归一化路径经其 21 节点
`run_on_cpu` 列表保持 FP32（见[转换文档](conversion/README_cn.md)）。

<a id="directory"></a>
## 目录结构

```text
lite_avatar/
├── conversion/  # 导出、变换、量化校准与编译配置
├── evaluator/   # 帧级输出质量对比
├── model/       # 已发布产物的身份、准备与校验
├── runtime/     # Python 推理（main.py、cli.py、avatar.py、frontend.py、media_worker.py）
├── test_data/   # 捆绑的示例音频；数字人数据准备
├── tests/       # 宿主机测试（SDK 替身，无需上板）
├── README.md    # 英文说明
├── README_cn.md # 中文说明
└── requirements-host.txt  # 宿主机 Python 依赖
```

<a id="support-matrix"></a>
## 支持矩阵

本示例提供 S100、S100P、S600 的 Python 运行入口。每个目标需要 `encoder`、
`audio2mouth`、`net_encode`、`net_decode` 四个匹配的模型文件，放在
`model/<target>/`。模型通过本地包或自行编译准备，见
[模型准备](model/README_cn.md)。下表列出各目标的能力，`—` 表示不适用。

| 目标 | 所需模型 | 语言 | 背景解码 | 视频编码器 |
| --- | --- | --- | --- | --- |
| s100 | encoder + audio2mouth + net_encode + net_decode（`*_s100.hbm`） | python | VPU H.264（文件模式） | VPU H.264 工作进程 |
| s100p | encoder + audio2mouth + net_encode + net_decode（`*_s100p.hbm`） | python | VPU H.264（文件模式） | VPU H.264 工作进程 |
| s600 | encoder + audio2mouth + net_encode + net_decode（`*_s600.hbm`） | python | VPU H.264（文件模式） | VPU H.264 工作进程 |
| x5 | — | — | — | — |

<a id="prerequisites"></a>
## 环境要求

- 按[模型准备](model/README_cn.md)完成校验的四个 HBM 产物，位于
  `model/<目标>/`（本地包或自行编译；`verify.py --target <t>`）。
- 按上游固定版本 `sample_data.zip` 解压出的数字人数据目录（见
  [测试数据](test_data/README_cn.md)；默认位置 `test_data/preload`），外加
  **一次性主机背景预处理**：自带的 `bg_video.mp4` 为 MPEG-4 Part 2，需在
  主机上转码一次生成 `bg_video.h264` 与 `bg_video.json` sidecar（见
  [运行时背景预处理](runtime/python/README_cn.md#background-prep)）。
- [运行时文档](runtime/python/README_cn.md)中的板端依赖（`hbm_runtime`、
  `hobot_vio`、Torch 2.3.1/torchaudio、NumPy、SciPy、Pillow、soundfile）、
  系统 `liblz4`，以及 PATH 上的 `ffmpeg`（含 `ffprobe`）——写视频需要；
  仅 `--no-encode` 不需要。

<a id="quickstart"></a>
## 快速开始

命令均从仓库根目录执行。第 1–2 步在任意有 FFmpeg 的主机上准备数据；
第 3–4 步在拷贝好数据的板卡上运行。

在主机上：

```bash
# 1. 下载并校验数字人数据（一次性，约 5 MB）
#    成功：shasum 输出 "sample_data.zip: OK"
cd <任意目录>
wget https://github.com/HumanAIGC/lite-avatar/raw/5b7ec850945e03d56fb290b05fb68440c359fa86/data/sample_data.zip
echo "d1c77b2df48b7fd7f19e6eef52bc5107e3131dd9777162faf2576f9451f69d7b  sample_data.zip" | shasum -a 256 -c -
unzip sample_data.zip -d <repo>/samples/vision/lite_avatar/test_data/
cd <repo>

# 2. 一次性准备背景（脚本只读原始 .mp4）
#    成功：bg_video.h264 + bg_video.json 位于 test_data/preload/
python3 samples/vision/lite_avatar/test_data/prepare_background.py \
    --data samples/vision/lite_avatar/test_data/preload
```

把预处理好的数字人数据（含 `bg_video.h264` 与 `bg_video.json`）和模型
产物拷到板上，然后在板上：

```bash
# 3. 校验产物集（S100P/S600 加 --target s100p / s600）
#    成功：verify.py 退出码 0 并逐项打印摘要匹配
python3 samples/vision/lite_avatar/model/verify.py --target s100

# 4. 渲染捆绑的示例语音
#    输出：outputs/lite_avatar/avatar.mp4（166 帧 @ 30 fps，已混音）
#    成功：退出码 0 且出现 "Video written" 汇总行
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 \
    --audio samples/vision/lite_avatar/test_data/asr_example.wav
```

`--target` 选择产物集且须与板卡一致（其他板卡用 `s100p` / `s600`）；
默认 `--encoder auto` 在所有受支持目标上解析为 VPU H.264 编码器，
`--no-encode` 只渲染不写盘。

<a id="expected-results"></a>
## 预期结果

运行会打印音频/渲染各阶段耗时、渲染速率（`Rendered 166 frames ... in
...s (... fps)`）与 `Video written: outputs/lite_avatar/avatar.mp4
(encoder: hardware)`，随后退出码 0。输出为可解码的 H.264 MP4：背景循环
帧与动画嘴部 ROI 的数字人几何，30 fps，输入音频已混流。产物缺失时以退出码
2 失败并给出准备指引；自行编译的产物集按[模型准备](model/README_cn.md)
通过 `--models-manifest` 运行。

输出质量用 [evaluator/evaluate_frames.py](evaluator/evaluate_frames.py)
与参考渲染对比衡量；自行编译产物集的产物级校验见
[转换文档](conversion/README_cn.md#validation)。

### 性能

捆绑 166 帧语音以普通入口实测（torch 2.3.1，所有目标硬件编码）：

| 目标 | 渲染速率（166 帧窗口） | 端到端耗时 |
| --- | --- | --- |
| s100p | 18.7 fps | 24.7 秒 |
| s600 | 20.5 fps | 28.3 秒 |
| s100 | 15.3 fps | 28.6 秒 |

渲染速率窗口覆盖全部 166 帧渲染加硬件编码器排空与 MP4 封装；此前的
背景硬件解码、音频前端与渲染器初始化等一次性阶段不在窗口内。端到端
耗时为完整 CLI 运行，含初始化。

<a id="entry-points"></a>
## 入口

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行：[runtime/python/README_cn.md](runtime/python/README_cn.md)
- 转换：[conversion/README_cn.md](conversion/README_cn.md)
- 评估：[evaluator/README_cn.md](evaluator/README_cn.md)
- 测试数据：[test_data/README_cn.md](test_data/README_cn.md)

<a id="license"></a>
## 许可

样例代码遵循仓库顶层 LICENSE（Apache-2.0）。上游项目为
[lite-avatar](https://github.com/HumanAIGC/lite-avatar)（MIT，固定于
`5b7ec850945e03d56fb290b05fb68440c359fa86`），其完整许可证文本以
[LICENSE-LiteAvatar](LICENSE-LiteAvatar) 随本样例分发。转换脚本调用 FunASR
前端处理 CMVN（`am.mvn`），该前端为 MIT，见
[conversion/LICENSE-FunASR](conversion/LICENSE-FunASR)；示例音频
`test_data/asr_example.wav` 随 ModelScope paraformer 包分发（Apache-2.0，
见 [test_data/README.md](test_data/README.md)）。模型权重从上游 ModelScope
仓库 [HumanAIGC-Engineering/LiteAvatarGallery](https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery)
下载；其模型卡未声明权重许可证，因此权重始终作为用户自备的转换输入
（见[转换文档](conversion/README_cn.md)）。
