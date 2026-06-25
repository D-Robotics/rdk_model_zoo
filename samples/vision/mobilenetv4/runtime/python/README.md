English | [简体中文](./README_cn.md)

# MobileNetV4 Python Runtime

This sample runs MobileNetV4 image classification with `hbm_runtime` and prints
Top-K ImageNet predictions. It supports the S100 / S600 small and medium HBM
models, with the SoC auto-detected at runtime.

## Directory Structure

```text
.
|-- README.md
|-- README_cn.md
|-- main.py
|-- mobilenetv4.py
`-- run.sh
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--model-variant` | Model variant: `small` or `medium` | `small` |
| `--model-path` | HBM model path. Empty value resolves to the sample-local model for `--model-variant` | `small`: `../../model/<soc>/mobilenetv4_small_224x224_nv12.hbm` |
| `--test-img` | Input image path | `../../test_data/zebra_cls.jpg` |
| `--label-file` | ImageNet label file | `../../test_data/imagenet_classes.names` |
| `--top-k` | Number of classification results to print | `5` |
| `--priority` | Runtime priority, 0 is lowest | `0` |
| `--bpu-cores` | BPU core index list | `0` |

## Quick Run

Small model:

```bash
bash run.sh
```

Medium model:

```bash
bash run.sh medium
```

The script downloads models through `../../model/download_model.sh` and uses the
sample-local `../../model/<soc>/` directory (`<soc>` ∈ {`s100`, `s600`}).

## Direct Run

Substitute `<soc>` with `s100` or `s600`:

```bash
python3 main.py \
  --model-variant small \
  --model-path ../../model/<soc>/mobilenetv4_small_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

```bash
python3 main.py \
  --model-variant medium \
  --model-path ../../model/<soc>/mobilenetv4_medium_256x256_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

## Runtime Interface

`mobilenetv4.py` provides:

- `MobileNetV4Config`
- `MobileNetV4.set_scheduling_params(...)`
- `MobileNetV4.pre_process(...)`
- `MobileNetV4.forward(...)`
- `MobileNetV4.post_process(...)`
- `MobileNetV4.predict(...)`
- `MobileNetV4.__call__(...)`

The wrapper converts the resized BGR image to NV12 Y and UV planes and feeds
the two fixed input tensors to `HB_HBMRuntime`.

Expected result for `zebra_cls.jpg`:

```text
Top-5 Classification Results:
  [0] zebra: ...
```

For source code documentation conventions, see
`../../../../../docs/source_reference/README.md`.
