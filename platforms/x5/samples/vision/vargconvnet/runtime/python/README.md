# VargConvNet Image Classification Python Sample

This sample demonstrates how to use the quantized VargConvNet model on BPU for ImageNet-1k image classification.

## Directory Structure

```text
.
|-- main.py
|-- vargconvnet.py
|-- README.md
|-- README_cn.md
`-- run.sh
```

## Parameter Description

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | Path to the quantized `.bin` model file. | `../../model/vargconvnet_224x224_nv12.bin` |
| `--label-file` | Path to the ImageNet label file. | `../../../../../datasets/imagenet/imagenet_classes.names` |
| `--priority` | Model priority in the range `0~255`. | `0` |
| `--bpu-cores` | BPU core indexes used for inference. | `0` |
| `--test-img` | Path to the test input image. | `../../test_data/box_turtle.JPEG` |
| `--img-save-path` | Path to save the output visualization image. | `../../test_data/inference.png` |
| `--resize-type` | Resize strategy (`0`: direct resize, `1`: letterbox). | `1` |
| `--topk` | Number of Top-K classes to display. | `5` |

## Quick Run

```bash
chmod +x run.sh
./run.sh
```

## Manual Run

Prepare the default model before running `main.py` directly. You can either run
`./run.sh` once, or run `../../model/download.sh` from this directory to download
`../../model/vargconvnet_224x224_nv12.bin`.

- Run with default parameters:

```bash
python3 main.py
```

- Run with explicit parameters:

```bash
python3 main.py \
    --model-path ../../model/vargconvnet_224x224_nv12.bin \
    --test-img ../../test_data/box_turtle.JPEG \
    --img-save-path ../../test_data/inference.png \
    --topk 5
```

## Interface Description

- **VargConvNetConfig**: Encapsulates model path, label file, and inference parameters.
- **VargConvNet**: Implements preprocessing, BPU execution, and Top-K classification post-processing.
