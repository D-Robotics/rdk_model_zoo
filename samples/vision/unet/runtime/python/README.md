[English](./README.md) | [简体中文](./README_cn.md)

# UNet Python Runtime

This RDK X5 sample reads one OpenCV BGR image, converts it to packed NV12,
executes UNet on the BPU, and writes a class-index mask, colored overlay, and
machine-readable JSON report.

## Requirements

- RDK X5 with RDK OS 3.5.0 or newer
- Python 3.10 or newer
- The board-provided `hbm_runtime` package
- OpenCV and NumPy
- A `bayes-e` UNet BIN produced by
  [`conversion/mapper.py`](../../conversion/mapper.py)

On the board, verify the Runtime and install only missing general dependencies:

```bash
python3 -c "from hbm_runtime import HB_HBMRuntime; print(HB_HBMRuntime.version)"
sudo apt update
sudo apt install -y python3-opencv python3-numpy
```

Do not install an arbitrary same-named `hbm_runtime` wheel for another platform.

## Directory Structure

```text
python/
├── unet.py       # UNetConfig and reusable UNet inference wrapper
├── main.py       # CLI, visualization, and JSON report entry
├── run.sh        # One-command launcher
├── README.md
└── README_cn.md
```

## Default Assets

The code defines repository-standard default paths:

- model: `../../model/unet_resnet18_voc_512x512_nv12.bin`
- image: `../../test_data/2007_000033.jpg`

The sample image is included in the repository. When the default model is
missing and no `--model-path` is supplied, `run.sh` automatically calls
`../../model/download_model.sh resnet18` before starting inference.

## Command-Line Arguments

Run `python3 main.py --help` to print the current interface.

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | X5 UNet BIN model path | `../../model/unet_resnet18_voc_512x512_nv12.bin` |
| `--test-img` | Input image readable by OpenCV | `../../test_data/2007_000033.jpg` |
| `--mask-save-path` | Raw uint8 class-index PNG | `unet_mask.png` |
| `--img-save-path` | Colored segmentation overlay | `unet_result.png` |
| `--report-path` | Runtime metadata and result JSON | `unet_runtime_report.json` |
| `--priority` | Optional BPU scheduler priority | unset |
| `--bpu-core` | Optional BPU core index | unset |
| `--alpha` | Segmentation color weight in the overlay | `0.55` |

## Quick Run

### Default paths

The default command downloads ResNet18 when necessary and then runs inference:

```bash
cd samples/vision/unet/runtime/python
./run.sh
```

### Explicit paths

```bash
cd samples/vision/unet/runtime/python
./run.sh \
  --model-path /path/to/unet_resnet18_voc_512x512_nv12.bin \
  --test-img /path/to/image.jpg \
  --mask-save-path unet_mask.png \
  --img-save-path unet_result.png \
  --report-path unet_runtime_report.json
```

Scheduling is optional. Omitting both `--priority` and `--bpu-core` leaves the
board Runtime scheduler unchanged.

Supplying `--model-path` disables the automatic default-model download. The
specified model must already exist.

## Outputs

| Output | Contents |
| --- | --- |
| `unet_mask.png` | UInt8 Pascal VOC class IDs `0..20`, shape `[512, 512]` |
| `unet_result.png` | VOC-colored mask blended over the resized source image |
| `unet_runtime_report.json` | Runtime version, model I/O metadata, classes present, elapsed time, and output paths |

The Runtime requires exactly one model, one NV12 input, and one semantic-logit
output. Unsupported image types or incompatible model contracts raise an
exception instead of silently producing a result.

## Python API

`unet.py` follows the repository Config/Model interface:

| Interface | Responsibility |
| --- | --- |
| `UNetConfig` | Model path, input size, and class count |
| `UNet.__init__` | Load the BIN and extract/validate model metadata |
| `set_scheduling_params` | Apply optional priority or BPU core affinity; no-op when unset |
| `pre_process` | Resize BGR uint8 input and return an `hbm_runtime.run`-compatible NV12 dictionary |
| `forward` | Return the direct output of `HB_HBMRuntime.run` |
| `post_process` | Dequantize when needed and return the uint8 class-index mask |
| `predict` / `__call__` | Execute preprocessing, inference, and postprocessing |

For generated API documentation, follow the
[source documentation guide](../../../../../docs/source_reference/README.md).

## Notes

- The input is OpenCV BGR uint8. Other layouts and dtypes are rejected.
- The output mask stays at the model resolution; it is not resized back to the
  original image size.
- Pure BPU performance should be measured with `hrt_model_exec`. The elapsed
  value in the JSON report includes Python preprocessing, inference output
  transfer, and postprocessing.
