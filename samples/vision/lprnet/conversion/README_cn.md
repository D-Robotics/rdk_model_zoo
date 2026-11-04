[English](README.md) | 简体中文

# LPRNet 转换

## Reproducible Sirius S conversion / 可复现的 S 转换

The 2026-11-04 S profiles use source commit `7c976664b3f3879efabeaff59c7a117e49d5f29e` from [sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch) (Apache-2.0), trained checkpoint `weights/Final_LPRNet_model.pth` SHA-256 `d1ca22ca7a251a90702a166c8ad8ce7f05acb39f5ef8055139a33d462716d1ad`. Keep all weights, ONNX and tensors on dev-server.

```bash
# Repository root, pinned source already present on dev-server; fresh output only.
# torch 2.6.0+cpu, numpy 1.26.4, onnx 1.17.0, onnxruntime 1.20.1, opencv 4.11
python3 -m samples.vision.lprnet.conversion.export --source /path/to/pinned/LPRNet_Pytorch --output /path/to/fresh/export
cp samples/vision/lprnet/conversion/recipes/sirius-68class-s100.yaml /path/to/fresh/export/compile.yaml
# Cached S SDK 3.7.0 container, cwd the fresh export directory:
hb_compile -c compile.yaml
hb_model_info output/lprnet_sirius_68class_24x94_bgr_featuremap_int16_cpuptq_o1_20261104_nashe.hbm
```

Use the target's own YAML: S100 nash-e, S100P nash-m, S600 nash-p. Actual tools: hb_compile 3.5.3 / HMCT 2.6.5 / HBDK 4.7.5. CPU-only PTQ, FEATUREMAP float32 NCHW, no fused preprocessing, INT16 max percentile 0.9999, O1, one core, two compiler jobs. Each march has an independent calibration/quantization run. Do not relabel an old BIN/HBM or share another march's BC.

The exporter verifies the trained checkpoint, source definition and 68-class order; exactly rewrites 4D-unbatched MaxPool3d to channel slicing plus spatial pooling, retaining bit-exact source outputs. It exports all `(1,68,18,1)` logits and checks three real inputs plus random against source PyTorch at atol=1e-4/rtol=1e-3. The cached CPU SDK's full-output parity gate also passed, with zero observed difference.

Calibration uses 200 distinct real plate images, seed42, selected from the pinned source's 1,000-image test directory. This is explicitly a PTQ partition; the other 800 are held out for future accuracy evaluation. BGR linear resize94×24, `(pixel-127.5)/128`, float32 NCHW `.npy`; no duplicated or random calibration. Dataset accuracy is not measured.

`converted_targets_20261104.json` freezes actual source/checkpoint/ONNX/effective-config/recipe hashes, CPU reference gates and compiler receipts. The original Workbench producer and effective configuration remain in the run evidence; this portable exporter is a subsequent repository entry. Compiler ABI and complete originalNodeInfo/OE reports are collected for every artifact. Board smoke, accuracy, Runtime1/2 and C++1/2 are all `not-run`.

The following sections preserve the historical X5 delivery; they do not describe the new S checkpoints.

<a id="source-model"></a>
## 历史 X5 源模型

源交付提供 X5 部署制品和协议说明，但没有仓库内 ONNX 导出脚本、权重文件、校准集或 PTQ YAML。源 conversion README 使用外部 OE 包和用户提供的 `your_lprnet_config.yaml`；该占位文件不在本仓库，不能视为可复现配方。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

源文档指向带有 `hb_mapper` 和 `hrt_model_exec` 的 RDK X5 OpenExplorer 环境。该历史交付仅对应 X5；新的 S 平台独立转换使用上文说明的训练权重。

<a id="export"></a>
## 导出

没有导出配方。用户必须自行提供 ONNX/checkpoint 流程，并产出 `input float32 NCHW (1,3,24,94)`、`output float32 (1,68,18,1)` 的部署图——即发布版 `lpr.bin` 的原生 logits；CTC 解码在移除单元素轴后消费 `(68,18)` 载荷。

<a id="calibration"></a>
## 校准

没有校准数据集或校准生成器。已提交的 `test_input.dat` 是运行时输入示例输入，不是校准集，不能称为图像数据或代表性校准数据。

<a id="compile"></a>
## 编译

用户在本 conversion 目录补齐外部 ONNX 和 YAML 后，源文档给出以下 OE 命令；它们是条件模板，不是本仓可直接运行的完整配方：

```bash
# cwd: samples/vision/lprnet/conversion；输入：外部 your_lprnet_config.yaml
hb_mapper checker --model-type onnx --config ./your_lprnet_config.yaml
hb_mapper makertbin --model-type onnx --config ./your_lprnet_config.yaml
# 成功需 OE 工具报告成功并生成 X5 .bin 制品
```

确切输出文件名、校准选项、量化设置和源权重文件均未知。

<a id="validation"></a>
## 转换后验证

可用 `hrt_model_exec model_info --model_file ./lpr.bin` 检查生成模型，再与运行时模型绑定对照。转换后在板端运行[评估器](../evaluator/README_cn.md)对照器检查两套实现的 logits 一致性；转换配方的重建以本目录列出的前提为准。

<a id="artifacts"></a>
## 产物

历史 X5 清单制品是 `x5:lprnet:lpr.bin`；运行时期望路径为 `../model/lpr.bin`。ONNX、权重文件、校准数据和编译日志属于外部输入，不纳入仓库。

<a id="known-gaps"></a>
## 补充准备

- 源没有导出脚本、权重文件版本、校准生成器、PTQ YAML 或可复现 OE 包。
- `your_lprnet_config.yaml` 是源占位符，不是仓库文件。
- 发布者 SHA-256 未知。已发布制品的板端一致性记录在 evaluator README 中，不延伸到重建。
