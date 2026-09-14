[English](./README.md) | [简体中文](./README_cn.md)

# RDK X5 UNet 模型转换

本目录只保留五个 UNet ResNet 变体从浮点 checkpoint 到 X5 的转换链路；训练
代码继续维护在 Model Zoo 仓库之外。

## 目录

```text
conversion/
├── mapper.py
├── onnx_export/
│   ├── export_unet.py
│   └── model/
└── ptq_yamls/
```

`onnx_export/model` 是训练、评测和导出共同使用的唯一 PyTorch 模型源码。
`ptq_yamls` 为每个 backbone 保存一份经过审阅的 `bayes-e` 模板；`mapper.py`
把模板绑定到本次 ONNX、校准集和全新输出目录，再执行 checker、makertbin 与
`hb_model_info`。

## 1. 导出 ONNX

checkpoint 必须与所选 backbone 一致。导出器会严格加载权重，生成固定 shape
的 opset 11 ONNX，执行 ONNX checker，并使用同一份确定性输入比较 PyTorch 与
ONNX Runtime。已有 ONNX 和报告不会被覆盖。

```bash
python onnx_export/export_unet.py \
  --backbone resnet18 \
  --checkpoint /models/unet_resnet18_voc_best.pth \
  --output /models/unet_resnet18_voc_512x512.onnx
```

只有数值比较通过的导出报告才能进入 `mapper.py`。`--skip-runtime-check` 仅用于
结构预检，生成的报告会明确标记为不能进入 X5 PTQ。

## 2. 准备校准张量

建议选取约 100 张有代表性的 Pascal VOC 训练图像。每个校准文件必须是无文件
头的小端 float32 `.bin`，保存一份 shape 为 `[3, 512, 512]` 的 RGB CHW 张量，
数值范围为 `[0, 255]`。数据脚本不要除以 255；PTQ YAML 通过
`data_scale=1/255` 统一承担归一化，板端 NV12 输入也使用同一规则。

`mapper.py` 会读取全部张量，拒绝文件大小错误、NaN/Inf 和越界值，并在本次
运行的 reports 目录生成带哈希的 `calibration-manifest.json`。

## 3. 编译 X5 模型

在同时提供 `hb_mapper` 与 `hb_model_info` 的 OpenExplorer Mapper 环境执行；
`--output` 指定的目录必须不存在。

```bash
python mapper.py \
  --backbone resnet18 \
  --onnx /models/unet_resnet18_voc_512x512.onnx \
  --calibration /data/unet/calibration_data_rgb_f32_512 \
  --output /output/unet_resnet18_x5_run_001
```

受门禁保护的执行顺序如下：

```text
导出报告 → 校准审计 → hb_mapper checker → hb_mapper makertbin
         → 唯一 .bin → hb_model_info 确认 BPU march: bayes-e
```

运行目录会保留解析后的 YAML、checker/build/model-info 日志、校准 manifest、
复制出的制品、哈希、工具版本与 `run-receipt.json`。编译成功后仍需使用
`../evaluator/eval_unet.py` 完成精度评测和板端 Runtime 验证。
