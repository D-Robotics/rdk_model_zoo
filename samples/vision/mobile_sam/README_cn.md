[English](README.md) | 简体中文

# MobileSAM

<a id="overview"></a>
## 算法与来源

MobileSAM 使用 TinyViT 图像编码器和 mask decoder 完成框提示图像分割。编码器把归一化的 `512x512` RGB 图像转换为 `1x256x32x32` embedding；解码器接收 embedding 和一个 `[x1,y1,x2,y2]` 框，返回三个 mask 候选及 IoU，运行时选择最佳候选并上采样。

- 论文：<https://arxiv.org/abs/2306.14289>
- 官方仓库：<https://github.com/ChaoningZhang/MobileSAM>
- source 基线：`platforms/s/samples/vision/mobile_sam` 和 `platforms/x5/samples/vision/mobile_sam`

输入直接拉伸至 512×512，框和结果 mask 都使用该坐标系，不反变换回原图。RGB 数值使用 mean `[123.675,116.28,103.53]`、std `[58.395,57.12,57.375]` 归一化；选中 mask 的阈值为 logits `>0`。

<a id="support-matrix"></a>
## 支持矩阵

| Target | Variant | Python | C++ |
|---|---|---|---|
| x5 | default `.bin` pair | supported-not-run | not-supported |
| s100 | nash-e `.hbm` pair | supported-not-run | not-supported |
| s100p | nash-m `.hbm` pair | supported-not-run | not-supported |
| s600 | nash-p `.hbm` pair | supported-not-run | not-supported |

主机 fixture 使用注入 runner；板端执行和 `hbm_runtime` 兼容性仍为 not-run。

<a id="prerequisites"></a>
## 环境前提

主机检查使用仓库 `.venv`、Python、NumPy、OpenCV 和 PyYAML；记录的主机 fixture 版本为 Python 3.14.7、NumPy 2.5.3、OpenCV 4.14.0、PyYAML 6.0.3，NumPy/OpenCV 版本未由本 sample 固定。runtime 语法要求 Python 3.10 或更新版本。板端需要匹配的 RDK 镜像与 `hbm_runtime`，其系统版本未知且未运行。两个模型制品必须显式准备，推理期间不会下载；转换材料见[转换说明](conversion/README_cn.md)。

板端 SDK 应由匹配的系统镜像提供，不从无关主机环境安装 `hbm_runtime`。在仓库根目录检查必要依赖：

```bash
# cwd: 所选板卡上的仓库根目录
python3 -c "import numpy, cv2, yaml, hbm_runtime; print('runtime dependencies available')"
```

若仅缺普通 Python 依赖，在板端 SDK 实际使用的 Python 环境安装（`python3 -m pip install numpy opencv-python PyYAML`）。源配方未固定板端依赖版本，应保留镜像与 SDK 的兼容约束。上述命令仅检查导入可用性；磁盘/RAM需求尚未测量，两个模型都须能由目标 runtime 同时加载。

<a id="quickstart"></a>
## 快速体验

在仓库根目录准备模型对，再在匹配板卡上运行。默认框是 resize 后 `512x512` 坐标中的 `[185,120,380,445]`。

```bash
# cwd：仓库根目录；前置：仅此准备步骤需要网络
python3 samples/vision/mobile_sam/model/download.py --target s100
# 预期：model/nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm 与 decoder_512_nashe.hbm

# cwd：仓库根目录；前置：已准备模型对和匹配板端 runtime
python3 samples/vision/mobile_sam/runtime/python/main.py --target s100 --box 185,120,380,445
# 预期：stdout 输出 JSON，test_data/ 下生成 mobile_sam_full_mask_result.jpg 和 mobile_sam_binary_mask_result.png
```

<a id="expected-results"></a>
## 预期结果

默认图像为 `test_data/dogs.jpg`。成功运行会生成 `512x512` overlay 和二值 mask；IoU 与 mask index 是模型输出，`mobile_sam_binary_mask.png` 是保留的 source 参考。

<a id="directory"></a>
## 目录职责

```text
mobile_sam/
├── model/                 # manifest 驱动的 encoder/decoder 准备
├── runtime/python/        # binding、runner、pipeline、CLI、可视化
├── test_data/             # dogs 图像和保留的 source mask
├── conversion/            # source 转换材料
├── evaluator/             # 评测流程与边界
├── README.md              # 英文总览
└── README_cn.md           # 中文总览
```

<a id="entry-points"></a>
## 入口索引

- [模型](model/README_cn.md)：target 制品、准备步骤和校验值。
- [Python runtime](runtime/python/README_cn.md)：CLI 与 `MobileSAMPipeline` API。
- [转换](conversion/README_cn.md)：导出、校准和编译材料。
- [评测](evaluator/README_cn.md)：参考流程和未测边界。
- C++：未提供。

<a id="license"></a>
## 许可

runtime 代码遵循仓库 Apache-2.0 许可。MobileSAM checkpoint、ONNX 和模型制品遵循上游条款，再分发前应确认其许可。
