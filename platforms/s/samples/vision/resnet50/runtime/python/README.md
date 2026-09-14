English | [简体中文](./README_cn.md)

# ResNet50 Python Runtime

This sample runs ResNet50 image classification with `hbm_runtime` and prints
Top-K ImageNet predictions.

## Directory Structure

```text
.
|-- README.md
|-- README_cn.md
|-- main.py
|-- resnet50.py
`-- run.sh
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--model-path` | HBM model path | `../../model/s100/resnet50_224x224_nv12.hbm` |
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
the sample-local `../../model/s100/` directory by default. For RDK S600, run
`bash ../../model/download_model.sh s600` first and override `--model-path` to
`../../model/s600/resnet50_224x224_nv12.hbm`.

## Direct Run

```bash
python3 main.py \
  --model-path ../../model/s100/resnet50_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

## Runtime Interface

`resnet50.py` provides:

- `Resnet50Config`
- `Resnet50.set_scheduling_params(...)`
- `Resnet50.pre_process(...)`
- `Resnet50.forward(...)`
- `Resnet50.post_process(...)`
- `Resnet50.predict(...)`
- `Resnet50.__call__(...)`

The wrapper converts the resized BGR image to NV12 Y and UV planes and feeds
the two fixed input tensors to `HB_HBMRuntime`.

Expected result for `zebra_cls.jpg`:

```text
Top-5 Classification Results:
  [0] zebra: ...
```
