English | [简体中文](./README_cn.md)

# MobileNetV4 Evaluation

This directory records how to validate the MobileNetV4 sample.
No standalone accuracy evaluator is provided in the original S100 sample.

## Functional Validation

Use the Python runtime:

```bash
cd ../runtime/python
bash run.sh
bash run.sh medium
```

Direct entry examples (substitute `<soc>` with `s100` or `s600`):

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

The result is considered valid only when Top-1 or Top-5 matches the semantic
content of the test image. For `zebra_cls.jpg`, `zebra` should appear with a
finite non-zero confidence score.

## Accuracy Evaluation

For full ImageNet evaluation, use the validation set and the same preprocessing
recorded in `../conversion/`: BGR input, NCHW training layout, ImageNet mean and
scale, and NV12 runtime input.

## Reference Records

The original conversion README records quantization cosine and toolchain
performance. See `../conversion/README.md` for details.
