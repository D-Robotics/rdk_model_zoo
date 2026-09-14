# Model Evaluation

This directory contains instructions for evaluating the accuracy and performance of the MODNet model.

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
| Accuracy | Evaluate alpha matte quality on matting benchmarks | Shared evaluation scripts |

## Performance Evaluation

Use `hrt_model_exec` to measure inference latency and throughput:

```bash
hrt_model_exec perf \
    --model_file ../../model/modnet_512x512_rgb.bin \
    --thread_num 1
```

## Accuracy Evaluation

Use the shared evaluation scripts in the tools directory for accuracy assessment on matting benchmarks.

For more details, refer to the [General Evaluation Guide](../../../../../docs/README.md).
