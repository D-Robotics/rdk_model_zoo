[English](./README.md) | 简体中文

# MobileSAM 转换说明

本目录说明 MobileSAM 在 RDK X5 上的完整转换流程。运行时 sample 使用两个量化 `.bin`：image encoder 和 box-prompt decoder。

## 目录结构

```text
.
|-- configs
|   |-- mobile_sam_image_encoder_norm_512x512_config.yaml
|   `-- mobile_sam_decoder_512_box_default_config.yaml
|-- scripts
|   |-- download_assets.py
|   |-- export_encoder_onnx.py
|   |-- export_decoder_onnx.py
|   |-- prepare_calibration.py
|   |-- prepare_decoder_calibration.py
|   `-- quantize.py
|-- README.md
`-- README_cn.md
```

ONNX、`*_quant_info.json`、校准 tensor 和量化 `.bin` 都由下面的转换步骤生成，不随 sample 提交。生成文件请保留在转换工作目录或输出目录中；运行 demo 时使用 `model/download_model.sh` 下载模型，或将本地生成的 `.bin` 复制到 `../model/`。

## 1. 克隆官方仓库

官方源码：https://github.com/ChaoningZhang/MobileSAM

```bash
cd samples/vision/mobile_sam/conversion
python3 scripts/download_assets.py --workspace ./workspace
```

生成：

```text
workspace/MobileSAM
workspace/MobileSAM/weights/mobile_sam.pt
```

## 2. 导出 Encoder ONNX

Encoder 导出脚本加载 `weights/mobile_sam.pt`，固定输入尺寸为 `512x512`，归一化在模型外完成，并导出固定 shape 的 opset-11 ONNX。

```bash
python3 scripts/export_encoder_onnx.py \
  --repo ./workspace/MobileSAM \
  --weights ./workspace/MobileSAM/weights/mobile_sam.pt \
  --output ./mobile_sam_image_encoder_norm_512_op11.onnx
```

接口：

```text
input:  normalized_images, 1x3x512x512, float32 NCHW
output: image_embeddings, 1x256x32x32, float32 NCHW
```

## 3. 导出 Decoder ONNX

Decoder 导出脚本加载同一 checkpoint，导出固定输入尺寸、box prompt 作为运行时输入的 opset-11 ONNX。

```bash
python3 scripts/export_decoder_onnx.py \
  --repo ./workspace/MobileSAM \
  --weights ./workspace/MobileSAM/weights/mobile_sam.pt \
  --output ./mobile_sam_decoder_512_box_op11.onnx
```

接口：

```text
inputs:
  image_embeddings, 1x256x32x32, float32 NCHW
  boxes, 1x4, float32
outputs:
  low_res_masks, 1x3x128x128, float32
  iou_predictions, 1x3, float32
```

## 4. 准备校准数据

Encoder 校准使用代表性图片，输出归一化后的 `1x3x512x512` float32 raw tensor：

```bash
python3 scripts/prepare_calibration.py \
  --src /path/to/calibration/images \
  --out ./calibration_data_norm_512 \
  --num 30
```

Decoder 校准使用真实 encoder embedding，并对 box 做确定性扰动：

```bash
python3 scripts/prepare_decoder_calibration.py \
  --embedding /path/to/mobile_sam_image_embeddings_f32.bin \
  --out ./decoder_calibration \
  --num 30
```

## 5. 搭配 YAML 量化

在 OE Docker `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` 中，从当前目录运行。

Encoder YAML：

```text
configs/mobile_sam_image_encoder_norm_512x512_config.yaml
```

Decoder YAML：

```text
configs/mobile_sam_decoder_512_box_default_config.yaml
```

量化命令：

```bash
python3 scripts/quantize.py --config configs/mobile_sam_image_encoder_norm_512x512_config.yaml
python3 scripts/quantize.py --config configs/mobile_sam_decoder_512_box_default_config.yaml
```

生成的 `.bin` 复制到 `../model/`。

## 6. 输出

将生成的 encoder 和 decoder `.bin` 文件复制到 `../model/`，用于运行时推理和性能验证。
