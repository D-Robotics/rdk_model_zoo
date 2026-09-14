[English](./README.md) | 简体中文

# MobileSAM 模型说明

本 sample 提供 RDK X5 上 MobileSAM 完整 mask 分割 demo。TinyViT image encoder 和 box-prompt mask decoder 都已量化为 `.bin`，并通过板端 `hbm_runtime` 运行。

## 原始仓库

- 官方仓库：https://github.com/ChaoningZhang/MobileSAM
- 权重：官方 MobileSAM 发布的 `weights/mobile_sam.pt`
- 导出输入尺寸：`512x512`

## 算法概述

- **任务**：提示式图像分割
- **提示**：box prompt `[185, 120, 380, 445]`
- **Encoder 输入**：`normalized_images`，`1x3x512x512`，NCHW float32 featuremap
- **Encoder 输出**：`image_embeddings`，`1x256x32x32`
- **Decoder 输入**：`image_embeddings` 和 `boxes`
- **Demo 输出**：完整 `512x512` 二值 mask 和叠加图

## 算法功能

- 在 RDK X5 上运行单图 MobileSAM 分割 demo。
- 使用固定尺寸 `512x512` 输入图，以及 resize 后图像坐标系中的一个 box prompt。
- 将 TinyViT image encoder 和 box-prompt mask decoder 作为两个量化 `.bin` 模型串联推理。
- 保存二值 mask 和 mask 叠加可视化图。

## 算法特点

- 默认 prompt 是固定 box `[185, 120, 380, 445]`，可通过运行时 `--box` 参数修改。
- 本 sample 聚焦基于 `hbm_runtime` 的板端 encoder + decoder 双模型推理。
- 本 demo 不包含交互式多 prompt UI、point prompt 或 batch image processing。

## 目录结构

```text
.
|-- conversion      # ONNX 导出、校准数据准备、量化脚本和 YAML
|   |-- configs     # 量化 YAML，不包含生成的 ONNX 或 quant_info
|   `-- scripts     # 资源下载、ONNX 导出、校准准备和量化入口脚本
|-- evaluator       # hrt_model_exec perf 多线程性能验证说明
|-- model           # download_model.sh 和模型说明，`.bin` 通过脚本下载
|-- runtime/python  # run.sh 和 hbm_runtime Python 推理 demo
|-- test_data       # 输入图片、完整 mask 输出图和二值 mask
`-- README_cn.md
```

## 快速开始

在带 `hbm_runtime` 的 RDK X5 板端运行：

```bash
cd samples/vision/mobile_sam/runtime/python
bash run.sh
```

输出文件：

- `test_data/mobile_sam_full_mask_result.jpg`
- `test_data/mobile_sam_binary_mask.png`

## 转换流程概要

1. 使用 `conversion/scripts/download_assets.py` 克隆/下载官方 MobileSAM 仓库和权重。
2. 使用 `conversion/scripts/export_encoder_onnx.py` 导出 `mobile_sam_image_encoder_norm_512_op11.onnx`。
3. 使用 `conversion/scripts/export_decoder_onnx.py` 导出 `mobile_sam_decoder_512_box_op11.onnx`。
4. 使用 `conversion/scripts/prepare_calibration.py` 准备 encoder 校准数据。
5. 使用 `conversion/scripts/prepare_decoder_calibration.py` 准备 decoder 的 embedding/box 校准数据。
6. 量化 YAML：
   - `conversion/configs/mobile_sam_image_encoder_norm_512x512_config.yaml`
   - `conversion/configs/mobile_sam_decoder_512_box_default_config.yaml`

完整命令见 `conversion/README.md`。

## 模型评测

多线程 `hrt_model_exec perf` 性能结果见 `evaluator/README_cn.md`。

## 验证结果

- Encoder 最终 cosine：`0.961277`。
- Decoder cosine：`low_res_masks=0.997539`，`iou_predictions=0.999972`。
- 已在板端通过 `hbm_runtime` 双 `.bin` 推理验证。

## 许可证

本 sample 遵循上游 MobileSAM 项目和 RDK Model Zoo 仓库的许可证要求。第三方模型和 checkpoint 的使用限制请以官方 MobileSAM 仓库说明为准。
