English | [简体中文](./README_cn.md)

# MobileNetV2 Model Conversion Guide

This directory provides the quantization YAML configuration and full conversion workflow notes for MobileNetV2 on RDK S100.

## Source Model

MobileNetV2 uses the timm library (PyTorch Image Models). Install dependencies:

```bash
pip install timm onnx
```

Download and export the ONNX model using the script in `runtime/python/get_mobilenetv2_onnx.py`. This script pulls from Hugging Face (`timm/mobilenetv2_100.ra_in1k`), so a HuggingFace login is required:

```bash
huggingface-cli login
python runtime/python/get_mobilenetv2_onnx.py
```

If you cannot configure a proxy, download the model manually from [timm/mobilenetv2_100.ra_in1k](https://huggingface.co/timm/mobilenetv2_100.ra_in1k) and convert with:

```bash
python runtime/python/timm2onnx_local.py
```

After exporting, the script prints model metadata:

```text
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model is valid.
Simplified model saved to mobilenetv2_100.onnx
Total number of parameters in the model: 3487818
```

## Dataset Preparation

The model uses the [ImageNet](https://image-net.org/) ILSVRC2012 dataset.

| Dataset | Classes | Images |
|---|---|---|
| Training Set | 1000 | ~1.2 million |
| Validation Set | 1000 | 50,000 |
| Test Set | 1000 | 100,000 |

Recommended directory structure:

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

Generate calibration data (100 images → `/calibration_data_rgb`):

```bash
python runtime/python/get_calibration_data.py
```

## Model Verification

Quick verify the ONNX model before full compilation:

```bash
hb_compile --model mobilenetv2_100.onnx --march nash-e
```

## Model Compilation

Run quantization compilation with the calibration dataset using the reference YAML:

```bash
hb_compile --config conversion/mobilenetv2_config.yaml
```

The YAML file `mobilenetv2_config.yaml` is provided in this directory. After compilation, the deployment file is:

```text
model_output/mobilenetv2_224x224_nv12.hbm
```

## Quantization Accuracy

Cosine similarity after quantization:

```text
+------------+-------------------+------------------+
| TensorName | Calibrated Cosine | Quantized Cosine |
+------------+-------------------+------------------+
| output     | 0.993383          | 0.988877         |
+------------+-------------------+------------------+
```

## Toolchain Performance Reference

```text
FPS (1 core): 4968.89
Latency: 0.2 ms (201.3 us)
BPU conv original OPs per run: 601,548,544
```

## Model Inference Scripts

After compilation, two inference scripts are available:

- `runtime/python/x86_inference.py` — inference on X86 with ONNX/HBIR/HBM formats, supports val dataset accuracy validation
- `runtime/python/s100_inference.py` — inference on board with HBM format using HB_HBMRuntime API

Example x86 inference:

```bash
python3 runtime/python/x86_inference.py \
  -m model_output/mobilenetv2_224x224_nv12_quantized_model.bc \
  -i test_data/zebra_cls.jpg
```

Example accuracy validation:

```bash
python3 runtime/python/x86_inference.py \
  -m model_output/mobilenetv2_224x224_nv12_quantized_model.bc \
  --validate \
  -d ../../../imagenet/val \
  -l ../../../imagenet/val.txt
```

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore environment.

- OE resource entry point: <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
