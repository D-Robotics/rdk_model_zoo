English | [简体中文](./README_cn.md)

# Python Runtime

This directory contains the MobileNetV1 Python inference entry point and wrapper.
`main.py` handles CLI parsing, image and label loading, runtime configuration,
prediction, and result printing. `mobilenetv1.py` implements the
`Config + Wrapper + predict()` pipeline.

## Files

```text
.
|-- main.py
|-- mobilenetv1.py
|-- run.sh
|-- README.md
`-- README_cn.md
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--model-path` | HBM model path | `../../model/s100/mobilenetv1_224x224_nv12.hbm` |
| `--test-img` | Input image path | `../../test_data/zebra_cls.jpg` |
| `--label-file` | ImageNet label file | `../../test_data/imagenet_classes.names` |
| `--top-k` | Number of classification results to print | `5` |
| `--priority` | Runtime priority, 0 is lowest | `0` |
| `--bpu-cores` | BPU core index list | `0` |

## Run

One-command run:

```bash
bash run.sh
```

Direct run:

```bash
python3 main.py \
  --model-path ../../model/s100/mobilenetv1_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

## Expected Result

For `zebra_cls.jpg`, the Top-5 output should include `zebra` with a reasonable
confidence score and no NaN or all-zero distribution.
