# ConvNeXt Image Classification Python Sample

This sample demonstrates how to use the quantized ConvNeXt model on BPU for image classification tasks.

## Environment Dependencies

This sample has no special environment dependencies, ensure that the environment dependencies in pydev are installed.

```bash
pip install numpy opencv-python hbm-runtime scipy
```

## Directory Structure

```text
.
├── main.py                # Inference entry script
├── convnext.py            # ConvNeXt model wrapper
├── run.sh                 # One-click execution script
└── README.md              # Usage instructions
```

## Parameter Description

| Parameter      | Description                                              | Default Value                               |
|----------------|----------------------------------------------------------|---------------------------------------------|
| `--model-path` | Path to the model file (.bin format)                     | `../../model/ConvNeXt_atto_224x224_nv12.bin` |
| `--test-img`   | Path to the test input image                             | `../../test_data/cheetah.JPEG`              |
| `--label-file` | Path to the ImageNet label file                          | `../../../../datasets/imagenet/imagenet_classes.names` |
| `--topk`       | Display Top-K results                                    | `5`                                         |
| `--resize-type`| Scaling strategy: 0 for direct resize, 1 for padding     | `1`                                         |

## Quick Run

- **One-click Execution Script**
    ```bash
    chmod +x run.sh
    ./run.sh
    ```

- **Manual Execution**
    - Use default parameters
        ```bash
        python3 main.py
        ```
    - Run with specified parameters
        ```bash
        python3 main.py \
            --model-path ../../model/ConvNeXt_atto_224x224_nv12.bin \
            --test-img ../../test_data/cheetah.JPEG \
            --topk 5
        ```

## Interface Description

- **ConvNeXtConfig**: Encapsulates model path and inference parameters.
- **ConvNeXt**: Contains the complete inference pipeline (`pre_process`, `forward`, `post_process`, `predict`).

Refer to the [Source Reference Documentation](../../../../../docs/source_reference/README.md) for more details.
