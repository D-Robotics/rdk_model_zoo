# Model Evaluation

This directory contains instructions for evaluating the accuracy and performance of the ConvNeXt model.

## Directory Structure

```text
.
├── README.md              # Documentation (English)
└── README_cn.md           # Documentation (Chinese)
```

## Evaluation Types

| Type | Description | Tool |
|------|-------------|------|
| Performance | Test inference speed and throughput | `hrt_model_exec perf` |
| Accuracy | Calculate Top-1 and Top-5 accuracy on ImageNet | Shared `eval_classify.py` |

## Accuracy Evaluation

Use the shared evaluation scripts in the tools directory:

```bash
python3 ../../../../tools/eval_batch_python.py \
    --model-path ../../model/ConvNeXt_nano.bin \
    --image-path /path/to/imagenet/val
```

For more details, refer to the [General Evaluation Guide](../../../../../docs/README.md).
