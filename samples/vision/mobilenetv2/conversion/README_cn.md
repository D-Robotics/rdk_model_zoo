[English](./README.md) | 简体中文

# MobileNetV2 模型转换说明

本目录提供 MobileNetV2 在 RDK S100 上的量化 YAML 配置和完整转换流程说明。

> 仓库附带的 `mobilenetv2_config.yaml` 针对 **S100**（`march: "nash-e"`）。
> 发布到 `rdk_s600/MobileNet/` 的 S600 版本基于同一份源 ONNX 和同一套量化配置，
> 只需把 `march` 改成 `nash-p` 即可。

## 源模型

MobileNetV2 使用 timm 库（PyTorch Image Models）。安装依赖：

```bash
pip install timm onnx
```

使用 `runtime/python/get_mobilenetv2_onnx.py` 脚本下载并导出 ONNX 模型。
该脚本从 Hugging Face（`timm/mobilenetv2_100.ra_in1k`）拉取，需要登录：

```bash
huggingface-cli login
python runtime/python/get_mobilenetv2_onnx.py
```

如果无法配置代理，可从 [timm/mobilenetv2_100.ra_in1k](https://huggingface.co/timm/mobilenetv2_100.ra_in1k)
手动下载模型，然后用以下脚本转换：

```bash
python runtime/python/timm2onnx_local.py
```

导出后输出模型元信息：

```text
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model is valid.
Simplified model saved to mobilenetv2_100.onnx
Total number of parameters in the model: 3487818
```

## 数据集准备

使用 [ImageNet](https://image-net.org/) ILSVRC2012 数据集。

| 数据集 | 类别数 | 图片数 |
|---|---|---|
| 训练集 | 1000 | 约 120 万 |
| 验证集 | 1000 | 50,000 |
| 测试集 | 1000 | 100,000 |

推荐目录结构：

```text
imagenet/
├── calibration_data/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...  (100 images)
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...
└── val.txt
```

生成校准数据（100 张图片 → `/calibration_data_rgb`）：

```bash
python runtime/python/get_calibration_data.py
```

## 模型验证

在完整编译前快速验证 ONNX 模型：

```bash
hb_compile --model mobilenetv2_100.onnx --march nash-e
```

## 模型编译

使用参考 YAML 在校准数据集上进行量化编译：

```bash
hb_compile --config conversion/mobilenetv2_config.yaml
```

YAML 文件 `mobilenetv2_config.yaml` 在本目录中提供。编译后产物：

```text
model_output/mobilenetv2_224x224_nv12.hbm
```

## 量化精度

量化后的余弦相似度：

```text
+------------+-------------------+------------------+
| TensorName | Calibrated Cosine | Quantized Cosine |
+------------+-------------------+------------------+
| output     | 0.993383          | 0.988877         |
+------------+-------------------+------------------+
```

## 工具链性能参考

```text
FPS (1 core): 4968.89
Latency: 0.2 ms (201.3 us)
BPU conv original OPs per run: 601,548,544
```

## 推理脚本

编译后提供两个推理脚本：

- `runtime/python/x86_inference.py` — 在 X86 上用 ONNX/HBIR/HBM 格式推理，支持验证集精度评测
- `runtime/python/s100_inference.py` — 在板端用 HBM 格式推理，使用 HB_HBMRuntime API

X86 推理示例：

```bash
python3 runtime/python/x86_inference.py \
  -m model_output/mobilenetv2_224x224_nv12_quantized_model.bc \
  -i test_data/zebra_cls.jpg
```

精度验证示例：

```bash
python3 runtime/python/x86_inference.py \
  -m model_output/mobilenetv2_224x224_nv12_quantized_model.bc \
  --validate \
  -d ../../../imagenet/val \
  -l ../../../imagenet/val.txt
```

## OE 资源

模型转换请在 x86 Linux 主机的 RDK S100 OpenExplore 环境中完成。

- OE 资源入口：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链在线手册：<https://toolchain.d-robotics.cc/>

## License

本目录基于 [Apache 2.0 License](../../../../LICENSE) 授权。