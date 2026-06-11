English | [简体中文](./README_cn.md)

# ResNet152 Evaluation

This sample keeps lightweight functional evaluation material for the migrated runtime. Full ImageNet accuracy evaluation is not included in this sample directory.

## Functional Check

Run the default demo:

```bash
cd ../runtime/python
bash run.sh
```

Run the entry script directly:

```bash
python3 main.py \
  --model-path ../../model/s100/resnet152_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

For `test_data/zebra_cls.jpg`, the Top-K output should contain a zebra-related ImageNet class with a high confidence score. The output should be finite, non-zero, and stable across repeated runs with the same input.

## Original Records

The reference record lists the following reference values:

| Item | Value |
| --- | --- |
| Frame total latency, thread 1 | `426.180 ms` |
| Average latency, thread 1 | `2.131 ms` |
| Frame rate, thread 1 | `463.021 FPS` |
| Frame total latency, thread 3 | `1100.839 ms` |
| Average latency, thread 3 | `5.504 ms` |
| Frame rate, thread 3 | `539.012 FPS` |

The reference result image is `test_data/result.png`.
