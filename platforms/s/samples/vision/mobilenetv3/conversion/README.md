English | [简体中文](./README_cn.md)

# Model Conversion

This directory preserves the original MobileNetV3 conversion assets from the
S100 sample and uses commands suitable for the current sample layout.
The Model Zoo already provides pre-compiled HBM models for S100 and S600 in
`../model/`.

> The shipped `mobilenetv3_config.yaml` targets **S100** (`march: "nash-e"`).
> The S600 build published under `rdk_s600/MobileNet/` is produced from the
> same source ONNX with the same quantization configuration — only `march`
> is changed to `nash-p`.

## Files

| File | Description |
| --- | --- |
| `mobilenetv3_config.yaml` | S100 conversion configuration for NV12 runtime input |
| `get_mobilenetv3_onnx.py` | Export `mobilenetv3_large_100` from timm to ONNX |
| `timm2onnx_local.py` | Legacy local-weight ONNX export helper; edit model name and weights path before use |
| `get_calibration_data.py` | Generate float32 BGR calibration `.npy` files |

## ONNX Export

The original sample uses the timm implementation:

- Model: `mobilenetv3_large_100`
- HuggingFace model page: `timm/mobilenetv3_large_100.ra_in1k`
- Required packages: `timm`, `onnx`, `onnxsim`, `torch`

Online export:

```bash
pip install timm onnx onnxsim
huggingface-cli login
python3 get_mobilenetv3_onnx.py
```

The script reports:

```text
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model is valid.
Simplified model saved to mobilenetv3_large_100.onnx
Total number of parameters in the model: 5470832
```

For local weights, edit `timm2onnx_local.py` before running it. The file is kept
from the original sample as a template and requires the user to set the model
name and weights path.

## Calibration Data

The original sample uses ImageNet calibration data. Prepare 100 calibration
images and generate BGR float32 `.npy` files:

```bash
python3 get_calibration_data.py
```

The script writes calibration tensors to:

```text
./calibration_data_bgr/
```

## Compile

Validate the ONNX model:

```bash
hb_compile --model mobilenetv3_large_100.onnx --march nash-e
```

Compile with the provided YAML:

```bash
hb_compile --config mobilenetv3_config.yaml
```

Key conversion settings:

| Item | Value |
| --- | --- |
| Source model | `mobilenetv3_large_100.onnx` |
| Runtime input | NV12, two inputs: Y plane and UV plane |
| Training input | BGR, NCHW |
| Calibration data | `./calibration_data_bgr`, float32 |
| Target march | `nash-e` for S100 |
| Output prefix | `mobilenetv3_224x224_nv12` |

The original YAML also preserves `node_info` entries that force selected nodes
to BPU with int16 input/output. Keep those entries unless the source ONNX graph
changes.

Original quantization record:

```text
TensorName: output
Calibrated Cosine: 0.911233
Quantized Cosine: 0.909042
```

Original toolchain performance reference:

```text
FPS (1 core): 2616.81
latency: 0.38 ms (382.1 us)
BPU conv original OPs per run: 433,179,520
```

This sample uses the public S100 HBM model. Use the conversion reference above
when regenerating the model.

## Conversion Reference

- ONNX export
- PTQ configuration generation

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore environment. Model conversion is not intended to run on the board.

- OE resource entry point (Docker + OE development package): <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

Download the OpenExplore CPU Docker image for RDK S100/S100P from the OE
resource entry point, then load the actual image file:

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
sudo docker images
```

Start the container with the repository mounted and enough shared memory for compilation:

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```
