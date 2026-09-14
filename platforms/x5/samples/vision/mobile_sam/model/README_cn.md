[English](./README.md) | 简体中文

# 模型文件

本目录保存 MobileSAM 完整 mask 推理所需的 RDK X5 运行时模型。

## 下载

运行 `bash download_model.sh` 可从 RDK X5 Model Zoo archive 下载 `.bin` 文件，也可以按转换流程重新生成。如果交付包不可用，请按 `../conversion/README_cn.md` 克隆官方 MobileSAM 仓库、导出 ONNX，并使用配套 YAML 量化生成。

- 官方源码仓库：https://github.com/ChaoningZhang/MobileSAM
- 权重来源：官方 MobileSAM `weights/mobile_sam.pt`
- Encoder 下载地址：https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobile_sam/mobile_sam_image_encoder_norm_512x512_allint16.bin
- Encoder 量化 YAML：`../conversion/configs/mobile_sam_image_encoder_norm_512x512_config.yaml`
- Decoder 下载地址：https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobile_sam/mobile_sam_decoder_512_box_default.bin
- Decoder 量化 YAML：`../conversion/configs/mobile_sam_decoder_512_box_default_config.yaml`

## 文件

| 文件 | 说明 |
| --- | --- |
| `mobile_sam_image_encoder_norm_512x512_allint16.bin` | `bayes-e` 上的量化 TinyViT image encoder |
| `mobile_sam_decoder_512_box_default.bin` | `bayes-e` 上的量化 box-prompt mask decoder |

## 接口

Encoder：

- 输入：
ormalized_images`，`1x3x512x512`，float32 NCHW
- 输出：`image_embeddings`，`1x256x32x32`

Decoder：

- 输入：`image_embeddings`，`1x256x32x32`；`boxes`，`1x4x1x1`
- 输出：`low_res_masks`，`1x3x128x128`；`iou_predictions`，`1x3x1x1`
