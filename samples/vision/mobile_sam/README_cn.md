[English](README.md) | 简体中文

# MobileSAM

<a id="overview"></a>
## 算法与来源

MobileSAM 使用 TinyViT 图像编码器和 mask decoder 完成框提示图像分割。编码器把归一化的 `512x512` RGB 图像转换为 `1x256x32x32` embedding；解码器接收 embedding 和一个 `[x1,y1,x2,y2]` 框，返回三个 mask 候选及 IoU，运行时选择最佳候选并上采样。

- 论文：<https://arxiv.org/abs/2306.14289>
- 官方仓库：<https://github.com/ChaoningZhang/MobileSAM>

输入直接拉伸至 512×512，框和结果 mask 都使用该坐标系，不反变换回原图。RGB 数值使用 mean `[123.675,116.28,103.53]`、std `[58.395,57.12,57.375]` 归一化；选中 mask 的阈值为 logits `>0`。

<a id="directory"></a>
## 目录结构

```text
mobile_sam/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="support-matrix"></a>
## 支持矩阵

| Target | Variant | Python | C++ |
|---|---|---|---|
| x5 | default `.bin` pair | supported | not-supported |
| s100 | nash-e `.hbm` pair | supported | not-supported |
| s100p | nash-m `.hbm` pair | supported | not-supported |
| s600 | nash-p `.hbm` pair | supported | not-supported |

<a id="prerequisites"></a>
## 环境前提

使用 Python 3.10+、NumPy、OpenCV 和 PyYAML，以及目标 RDK 镜像提供的 `hbm_runtime`。运行前准备 encoder 与 decoder 两个模型文件；目标 OE 工具链说明见转换指南。

板端 SDK 应由匹配的系统镜像提供，不从无关主机环境安装 `hbm_runtime`。在仓库根目录检查必要依赖：

```bash
# cwd: 所选板卡上的仓库根目录
python3 -c "import numpy, cv2, yaml, hbm_runtime; print('runtime dependencies available')"
```

在板端 SDK 使用的 Python 环境安装依赖（`python3 -m pip install numpy opencv-python PyYAML`）。为 encoder 和 decoder 同时驻留预留内存。

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

<a id="entry-points"></a>
## 入口索引

- [模型](model/README_cn.md)：target 制品、准备步骤和校验值。
- [Python runtime](runtime/python/README_cn.md)：CLI 与 `MobileSAMPipeline` API。
- [转换](conversion/README_cn.md)：导出、校准和编译材料。
- [评测](evaluator/README_cn.md)：评测流程与参考记录。
- C++：未提供。

<a id="license"></a>
## 许可

runtime 代码遵循仓库 Apache-2.0 许可。MobileSAM checkpoint、ONNX 和模型制品遵循上游条款，再分发前应确认其许可。
