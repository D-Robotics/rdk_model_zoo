[English](./README.md) | 简体中文

# EfficientSAM-Tiny 转换说明

本目录说明 EfficientSAM-Tiny 在 RDK X5 上的完整转换流程。运行时 sample 使用两个量化 `.bin`：image encoder 和 fixed-prompt decoder。

## 目录结构

```text
.
|-- configs
|   |-- efficient_sam_vitt_encoder_featuremap_config.yaml
|   `-- efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml
|-- scripts
|   |-- download_assets.py
|   |-- export_encoder_onnx.py
|   |-- export_decoder_onnx.py
|   |-- prepare_calibration.py
|   |-- prepare_efficient_decoder_calibration.py
|   `-- quantize.py
|-- README.md
|-- README_cn.md
|-- QUANTIZATION_STATUS.md
`-- VALIDATION.md
```

ONNX、`*_quant_info.json`、校准 tensor 和量化 `.bin` 都由下面的转换步骤生成，不随 sample 提交。生成文件请保留在转换工作目录或输出目录中；运行 demo 时使用 `model/download_model.sh` 下载模型，或将本地生成的 `.bin` 复制到 `../model/`。

## 1. 克隆官方仓库

官方源码：https://github.com/yformer/EfficientSAM

```bash
cd samples/vision/efficient_sam/conversion
python3 scripts/download_assets.py --workspace ./workspace
```

生成：

```text
workspace/EfficientSAM
workspace/EfficientSAM/weights/efficient_sam_vitt.pt
```

## 2. 导出 Encoder ONNX

Encoder 导出脚本加载 `weights/efficient_sam_vitt.pt`，固定输入尺寸为 `512x512`，并导出适配工具链的 split QKV opset-11 ONNX。

```bash
python3 scripts/export_encoder_onnx.py \
  --repo ./workspace/EfficientSAM \
  --weights ./workspace/EfficientSAM/weights/efficient_sam_vitt.pt \
  --output ./efficient_sam_vitt_encoder_512_splitqkv_op11.onnx
```

接口：

```text
input:  batched_images, 1x3x512x512, float32 NCHW
output: image_embeddings, 1x256x32x32, float32 NCHW
```

## 3. 导出 Fixed-Prompt Decoder ONNX

Decoder 导出脚本加载同一 checkpoint，并将两个 positive point prompt 固化到 ONNX 中，使板端 demo 能以单 decoder 输入运行静态 full-mask 流程。

```bash
python3 scripts/export_decoder_onnx.py \
  --repo ./workspace/EfficientSAM \
  --weights ./workspace/EfficientSAM/weights/efficient_sam_vitt.pt \
  --output ./efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx
```

接口：

```text
input:  image_embeddings, 1x256x32x32, float32 NCHW
outputs:
  low_res_masks, 1x3x128x128, float32
  iou_predictions, 1x3, float32
```

## 4. 准备校准数据

Encoder 校准使用代表性图片，输出 RGB CHW `1x3x512x512` float32 raw tensor：

```bash
python3 scripts/prepare_calibration.py \
  --src /path/to/calibration/images \
  --out ./calibration_data_rgbchw_512 \
  --num 30
```

Decoder 校准使用真实 encoder embedding：

```bash
python3 scripts/prepare_efficient_decoder_calibration.py \
  --embedding /path/to/efficient_sam_image_embeddings_f32.bin \
  --out ./decoder_calibration \
  --num 30
```

## 5. 搭配 YAML 量化

在 OE Docker `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` 中，从当前目录运行。

Encoder YAML：

```text
configs/efficient_sam_vitt_encoder_featuremap_config.yaml
```

它使用 `efficient_sam_vitt_encoder_512_splitqkv_op11.onnx` 和 `calibration_data_rgbchw_512`，并生成：

```text
bpu_model_output_512_default_none/efficient_sam_vitt_encoder_512x512_default_none.bin
```

Decoder YAML：

```text
configs/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml
```

它使用 `efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx` 和 decoder embedding 校准数据，并生成：

```text
bpu_model_output_decoder_fixedprompt/efficient_sam_vitt_decoder_fixedprompt_512_default.bin
```

量化命令：

```bash
python3 scripts/quantize.py --config configs/efficient_sam_vitt_encoder_featuremap_config.yaml
python3 scripts/quantize.py --config configs/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml
```

本地验证时可将生成的 `.bin` 复制到 `../model/`，也可以使用 `../model/download_model.sh` 下载已发布模型。