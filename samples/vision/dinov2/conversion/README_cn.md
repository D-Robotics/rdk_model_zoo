[English](./README.md) | 简体中文

# DINOv2 模型转换

本目录提供 FAIR DINOv2 ViT-S/14 视觉编码器在 RDK S100/S100P/S600 上的
ONNX 导出与 PTQ 转换流程。

## 文件

| 文件 | 说明 |
|---|---|
| `mapper.py` | 一键转换入口：ONNX 导出 + 校准数据准备 + hb_compile。 |
| `onnx_export/export_dinov2.py` | PyTorch 到 ONNX 的 BPU 友好导出脚本。 |

## 量化配方

经验证的配方为 **featuremap float32 输入 + 全图 int16 + 默认（KL）校准**。
三个要素缺一不可，已在 `mapper.py` 中固化：

| 要素 | 原因 |
|---|---|
| `input_type_rt: featuremap` | NV12 输入链在 embedding 模型上会摧毁执行精度（实测模拟 0.999、经 YUV420 uint8 往返后执行仅 0.01-0.12）。 |
| `all_node_type: int16` | int8 激活无论何种校准均不达标（实测上限 0.91）。权重 int8 几乎无损失，激活 int8 不行。 |
| 默认校准（不写 `calibration_type`） | hmct 的 modelwise KL 搜索能驯服原始自监督骨干的注意力 logit 离群（logits-minus-max 张量范围达 -345）；max+percentile 校准会崩到 0.18。 |

## 实测矩阵（nash-e，OE 3.7.0，hmct 2.6.5 / hbdk 4.7.5）

| 配置 | cls cosine | patch cosine | 结论 |
|---|---|---|---|
| int8 + softmax-int32，featuremap，max | 0.081 | 0.803 | 不通过 |
| int8 + softmax-int32，featuremap，KL | 0.892 | 0.894 | 不通过 |
| int16，featuremap，max + 0.9999 | 0.184 | 0.840 | 不通过 |
| int16，nv12，KL | 0.999（模拟） | 0.999（模拟） | 不通过（执行 0.01 / 0.12） |
| int16，featuremap，KL（本配方） | **0.9989** | **0.9983** | **通过** |

register-token（`_reg4`）变体已实测且有意不发布：在 per-tensor 校准下其
量化 cosine（0.80）劣于 plain 变体（0.999），尽管论文推荐其稠密特征质量。

## 使用方法

在 x86 主机的 OE docker 镜像内执行：

```bash
# 1. 获取 Apache-2.0 权重与源码仓。
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
git clone https://github.com/facebookresearch/dinov2.git

# 2. 将 50 张多样的真实图片（如 COCO val2017）放入 ./cal_images。

# 3. 转换。
python3 mapper.py \
    --weights ./dinov2_vits14_pretrain.pth \
    --repo ./dinov2 \
    --cal-images ./cal_images \
    --march nash-e \
    --output-dir ./output
```

校准图片必须是真实照片。随机或合成数据会破坏该骨干的 int16 校准。

## OE 资源

转换在 x86 Linux 主机的 RDK S100/S600 OpenExplore 环境（OE 3.7.0，镜像
`ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`）中执行，不支持在板上运行。

```bash
sudo docker run -it --rm --network host --shm-size=15g \
    -v "$(pwd)":/workspace -w /workspace \
    registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 \
    /bin/bash
```

## 许可

源权重为 Meta AI 发布的 Apache-2.0 许可
[DINOv2](https://github.com/facebookresearch/dinov2) 产物。见
[../../../../LICENSE](../../../../LICENSE)。
