# GoogLeNet Image Classification Python Sample

This sample demonstrates how to use the quantized GoogLeNet model on BPU for ImageNet-1k image classification.

## Directory Structure

```text
.
|-- main.py
|-- googlenet.py
|-- README.md
|-- README_cn.md
`-- run.sh
```

## Parameter Description

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | Path to the quantized `.bin` model file. | `/opt/hobot/model/x5/basic/googlenet_224x224_nv12.bin` |
| `--label-file` | Path to the ImageNet label file. | `../../test_data/ImageNet_1k.json` |
| `--priority` | Model priority in the range `0~255`. | `0` |
| `--bpu-cores` | BPU core indexes used for inference. | `0` |
| `--test-img` | Path to the test input image. | `../../test_data/indigo_bunting.JPEG` |
| `--img-save-path` | Path to save the output visualization image. | `../../test_data/result.jpg` |
| `--resize-type` | Resize strategy (`0`: direct resize, `1`: letterbox). | `1` |
| `--topk` | Number of Top-K classes to display. | `5` |

## Quick Run

```bash
chmod +x run.sh
./run.sh
```

## Manual Run

- Run with default parameters:

```bash
python3 main.py
```

- Run with explicit parameters:

```bash
python3 main.py \
    --model-path ../../model/googlenet_224x224_nv12.bin \
    --test-img ../../test_data/indigo_bunting.JPEG \
    --img-save-path ../../test_data/result.jpg \
    --topk 5
```

## Interface Description

- **GoogLeNetConfig**: Encapsulates model path, label file, and inference parameters.
- **GoogLeNet**: Implements preprocessing, BPU execution, and Top-K classification post-processing.
