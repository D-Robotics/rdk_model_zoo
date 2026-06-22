English | [简体中文](./README_cn.md)

# Model Evaluation

This directory records the evaluation entry for MobileNetV1 classification.
The current sample focuses on functional validation with a single ImageNet test
image. Full ImageNet accuracy evaluation should reuse the same preprocessing
and postprocessing logic from `runtime/python/mobilenetv1.py`.

## Functional Check

```bash
cd ../runtime/python
python3 main.py \
  --model-path ../../model/s100/mobilenetv1_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names
```

Pass criteria:

- Top-5 includes `zebra`.
- Scores are finite and not all zero.
- The output is stable across repeated runs with the same model and image.
