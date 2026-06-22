English | [简体中文](./README_cn.md)

# ResNet50 Evaluation

This directory records how to validate the ResNet50 sample on RDK S100 and RDK
S600. The original sample provides a notebook and runtime script, but no
standalone accuracy evaluator.

## Functional Validation

```bash
cd ../runtime/python
bash run.sh
```

Direct Python entry:

```bash
python3 main.py \
  --model-path ../../model/s100/resnet50_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

On RDK S600, swap the model path to `../../model/s600/resnet50_224x224_nv12.hbm`.

The result is considered valid only when Top-1 or Top-5 matches the semantic
content of the test image. For `zebra_cls.jpg`, `zebra` should appear with a
finite non-zero confidence score.

## Accuracy Evaluation

For full ImageNet evaluation, use the validation set and keep the same model
preprocessing as the conversion reference: 224x224 input and NV12 runtime input.

The original inference screenshot is preserved as `../test_data/result.png`.
