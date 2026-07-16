[English](./README.md) | 简体中文

# EfficientSAM-Tiny 模型说明

本 sample 提供 RDK X5 上 EfficientSAM-Tiny 完整 mask 分割 demo。Image encoder 和固定 prompt mask decoder 都已量化为 `.bin`，并通过板端 `hbm_runtime` 运行。

## 原始仓库

- 官方仓库：https://github.com/yformer/EfficientSAM
- 权重：官方 EfficientSAM 发布的 `weights/efficient_sam_vitt.pt`
- 导出输入尺寸：`512x512`

## 算法概述

- **任务**：提示式图像分割
- **提示**：已固化到 decoder ONNX 的正点 prompt：`(248,210)` 和 `(302,315)`
- **Encoder 输入**：`batched_images`，`1x3x512x512`，NCHW float32 featuremap
- **Encoder 输出**：`image_embeddings`，`1x256x32x32`
- **Decoder 输入**：`image_embeddings`
- **Demo 输出**：完整 `512x512` 二值 mask 和叠加图

## 算法功能

- 在 RDK X5 上运行单图 EfficientSAM-Tiny 分割 demo。
- 使用固定尺寸 `512x512` 输入图，以及固化到 decoder 模型中的两个 positive point prompt。
- 将 image encoder 和 fixed-prompt decoder 作为两个量化 `.bin` 模型串联推理。
- 保存二值 mask 和 mask 叠加可视化图。

## 算法特点

- 默认 prompt 是 resize 后图像坐标系中的固定点 `(248,210)` 和 `(302,315)`。
- 本 sample 聚焦基于 `hbm_runtime` 的板端 encoder + decoder 双模型推理。
- 本 demo 不包含交互式多 prompt UI、运行时 point prompt 选择或 batch image processing。

## 目录结构

```text
.
|-- conversion      # ONNX 导出、校准数据准备、量化脚本和 YAML
|   |-- configs     # 量化 YAML，不包含生成的 ONNX 或 quant_info
|   `-- scripts     # 资源下载、ONNX 导出、校准准备和量化入口脚本
|-- evaluator       # hrt_model_exec perf 性能验证说明
|-- model           # download_model.sh 和模型说明，`.bin` 通过脚本下载
|-- runtime/python  # run.sh 和 hbm_runtime Python 推理 demo
|-- test_data       # 输入图片、完整 mask 输出图和二值 mask
`-- README_cn.md
```

## 快速开始

在带 `hbm_runtime` 的 RDK X5 板端运行：

```bash
cd samples/vision/efficient_sam/runtime/python
bash run.sh
```

输出文件：

- `test_data/efficient_sam_full_mask_result.jpg`
- `test_data/efficient_sam_binary_mask.png`

## 转换流程概要

1. 使用 `conversion/scripts/download_assets.py` 克隆/下载官方 EfficientSAM 仓库和权重。
2. 使用 `conversion/scripts/export_encoder_onnx.py` 导出 `efficient_sam_vitt_encoder_512_splitqkv_op11.onnx`。
3. 使用 `conversion/scripts/export_decoder_onnx.py` 导出 `efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx`。
4. 使用 `conversion/scripts/prepare_calibration.py` 准备 encoder 校准数据。
5. 使用 `conversion/scripts/prepare_efficient_decoder_calibration.py` 准备 decoder 的 embedding 校准数据。
6. 使用以下 YAML 量化：
   - `conversion/configs/efficient_sam_vitt_encoder_featuremap_config.yaml`
   - `conversion/configs/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml`

ONNX、`*_quant_info.json`、校准 tensor 和量化 `.bin` 都由转换流程生成，不随 sample 提交。详见 `conversion/README_cn.md`。

## 模型评测

本 sample 不包含数据集级评估。性能数据见 `evaluator/README_cn.md`，量化精度记录见 `conversion/VALIDATION.md`。

## 验证

- Encoder final cosine：`0.968013`。
- Fixed-prompt decoder cosine：`low_res_masks=0.965641`，`iou_predictions=0.997313`。
- 已在板端通过 `hbm_runtime` 双 `.bin` 推理验证。

## License

本 sample 遵循上游 EfficientSAM 项目和 RDK Model Zoo 仓库的许可证条款。第三方模型和 checkpoint 使用要求请参考官方 EfficientSAM 仓库。