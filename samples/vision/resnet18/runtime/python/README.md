English | [简体中文](./README_cn.md)

# ResNet18 Python Runtime

This sample runs ResNet18 image classification with `hbm_runtime` and prints
Top-K ImageNet predictions.

## Directory Structure

```text
.
|-- README.md
|-- README_cn.md
|-- main.py
|-- resnet18.py
`-- run.sh
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--model-path` | HBM model path | `../../model/s100/resnet18_224x224_nv12.hbm` |
| `--test-img` | Input image path | `../../test_data/zebra_cls.jpg` |
| `--label-file` | ImageNet label file | `../../../../../datasets/imagenet/imagenet_classes.names` |
| `--top-k` | Number of classification results to print | `5` |
| `--priority` | Runtime priority, 0 is lowest | `0` |
| `--bpu-cores` | BPU core index list | `0` |

## Quick Run

```bash
bash run.sh
```

The script downloads the model through `../../model/download_model.sh` and uses
the sample-local `../../model/s100/` directory.

## Direct Run

```bash
python3 main.py \
  --model-path ../../model/s100/resnet18_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

## Runtime Interface

`resnet18.py` provides:

- `Resnet18Config`
- `Resnet18.set_scheduling_params(...)`
- `Resnet18.pre_process(...)`
- `Resnet18.forward(...)`
- `Resnet18.post_process(...)`
- `Resnet18.predict(...)`
- `Resnet18.__call__(...)`

The wrapper converts the resized BGR image to NV12 Y and UV planes and feeds
the two fixed input tensors to `HB_HBMRuntime`.

Expected result for `zebra_cls.jpg`:

```text
Top-5 Classification Results:
  [0] zebra: ...
```

For source code documentation conventions, see
`../../../../../docs/source_reference/README.md`.
