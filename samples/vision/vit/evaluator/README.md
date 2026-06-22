English | [简体中文](./README_cn.md)

# ViT Evaluation

This directory documents the CIFAR-10 accuracy record and functional result
checks for the ViT runtime.

## Functional Check

Run the default demo:

```bash
cd ../runtime/python
bash run.sh int8
```

Run the entry script directly:

```bash
python3 main.py \
  --model-path ../../model/s100/vit_cifar10_batch1_int8.hbm \
  --test-img ../../test_data/airplane_0000.png \
  --label-file ../../test_data/cifar10_classes.names \
  --top-k 5
```

For `test_data/airplane_0000.png`, the Top-K output should contain the
`airplane` CIFAR-10 class with a high confidence score. The output should be
finite, non-zero, and stable across repeated runs with the same input.

## Test Images

`test_data/` contains one image for each CIFAR-10 class:

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

## Accuracy Data

| Model | Top-1 | Top-5 |
| --- | --- | --- |
| ONNX | `74.54%` | `98.36%` |
| HBM | `72.62%` | `98.03%` |

## Accuracy Test Notes

1. BPU models can lose accuracy when NCHW RGB888 inputs are quantized and
   converted to YUV420SP (NV12), because the color-space conversion introduces
   additional error. Training with this conversion in the loop can reduce the
   loss.
2. Python and C/C++ inference interfaces can produce slight accuracy
   differences because memory copies and floating-point conversions are handled
   differently.
3. Batch evaluation scripts are available in the RDK Model Zoo evaluation tools:
   <https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/tools/eval_pycocotools>
4. The accuracy table uses PTQ with 50 calibration images. It represents a
   normal first-pass compilation result without QAT or additional accuracy
   tuning, and does not indicate the upper bound of model accuracy.
