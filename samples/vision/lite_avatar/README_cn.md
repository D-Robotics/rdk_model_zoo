[English](README.md) | 简体中文

# LiteAvatar 音频驱动数字人

在 RDK S100 上离线渲染音频驱动数字人：输入语音音频，输出合成好的数字人
视频。

来源：[HumanAIGC/lite-avatar](https://github.com/HumanAIGC/lite-avatar)（MIT）

<a id="overview"></a>

## 概述

本示例面向 S100 提供一个 Python 运行时。`LiteAvatarPipeline` 类把四个已
发布 HBM 产物串成每个阶段清晰可读的 `preprocess → infer → postprocess`
流程与一次整段音频的 `predict`：音频前端提取 Paraformer 特征（Kaldi
fbank、7/6 低帧率堆叠、CMVN），音频编码器输出 50 路堆叠隐层状态，
audio2mouth 把滑动的 30 帧窗口映射为嘴部参数（与上游完全一致的交叉淡化与
Butterworth 平滑），人脸解码器以参考帧特征金字塔为条件逐帧渲染。合成阶段
按原始羽化掩码与乒乓环顺序把解码出的嘴部区域贴回背景循环帧，随后 CLI 把
帧交付给 S100 硬件 H.264 编码器（或 FFmpeg 管道），写入器资源有界并在
失败时清理（见
[运行时文档](runtime/python/README_cn.md)）。

### 算法背景

LiteAvatar（HumanAIGC）用语音特征驱动轻量人脸生成网络：paraformer 编码器
的中间隐层状态携带发音信息，小型 audio2mouth 网络将其解码为每个视频帧
32 维嘴部参数；解码网络（InstanceNorm 风格模块与 x2 上采样）合成当前
姿态下的嘴部区域并融合进背景视频。四个 S100 产物在 BPU 上运行 INT16
激活；解码器中的 21 个 InstanceNorm 子图节点为保证数值一致性保留在 CPU
上以 FP32 运行（见[转换文档](conversion/README_cn.md)）。

<a id="directory"></a>
## 目录结构

```text
lite_avatar/
├── conversion/  # 导出、变换、量化校准与编译配置
├── evaluator/   # 帧级输出质量对比
├── model/       # 已发布产物的身份、准备与校验
├── runtime/     # Python 推理（main.py、cli.py、avatar.py、frontend.py）
├── test_data/   # 捆绑的示例音频；数字人数据准备
├── tests/       # 宿主机测试（SDK 替身，无需上板）
├── README.md    # 英文说明
├── README_cn.md # 中文说明
└── requirements-host.txt  # 宿主机 Python 依赖
```

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 语言 | 状态 |
| --- | --- | --- |
| s100 | python | supported |

<a id="prerequisites"></a>
## 环境要求

- 按[模型准备](model/README_cn.md)完成校验的四个 HBM 产物，位于
  `model/s100/`（本地包或自行编译）。
- 按上游固定版本 `sample_data.zip` 解压出的数字人数据目录（见
  [测试数据](test_data/README_cn.md)；默认位置 `test_data/preload`）。
- [运行时文档](runtime/python/README_cn.md)中的板端依赖（`hbm_runtime`、
  Torch 2.3.1/torchaudio、NumPy、SciPy、OpenCV、soundfile），以及 PATH 上的
  `ffmpeg`——**两种**编码器写视频都需要（硬件路径同样经由 ffmpeg 混流
  H.264 码流）；仅 `--no-encode` 不需要。

<a id="quickstart"></a>
## 快速开始

在 S100 板上从仓库根目录执行的完整路径：

```bash
# 1. 校验已准备好的产物集
#    成功：verify.py 退出码 0 并逐项打印摘要匹配
python3 samples/vision/lite_avatar/model/verify.py

# 2. 准备数字人数据（一次性，约 5 MB）
#    成功：shasum 输出 "sample_data.zip: OK"
cd <任意目录>
wget https://github.com/HumanAIGC/lite-avatar/raw/5b7ec850945e03d56fb290b05fb68440c359fa86/data/sample_data.zip
echo "d1c77b2df48b7fd7f19e6eef52bc5107e3131dd9777162faf2576f9451f69d7b  sample_data.zip" | shasum -a 256 -c -
unzip sample_data.zip -d <repo>/samples/vision/lite_avatar/test_data/
cd <repo>

# 3. 渲染捆绑的示例语音
#    输出：outputs/lite_avatar/avatar.mp4（166 帧 @ 30 fps，已混音）
#    成功：退出码 0 且出现 "Video written" 汇总行
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 \
    --audio samples/vision/lite_avatar/test_data/asr_example.wav
```

默认 `--encoder auto` 在 S100 上解析为硬件 H.264 编码器；`--encoder
software` 切换 FFmpeg 管道，`--no-encode` 只渲染不写盘。完整 CLI 与 API
参考见[运行时文档](runtime/python/README_cn.md)。

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

以普通入口在 S100 上实测（torch 2.3.1、硬件编码器、默认线程策略）：
捆绑的 166 帧语音渲染用时 **7.54 s——22.01 fps**（`Rendered 166
frames ... in 7.54s (22.01 fps)`）。计时窗口覆盖全部 166 帧渲染加编码器
排空与 MP4 混流；此前的音频前端与渲染器初始化等一次性离线阶段不在窗口
内。

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
