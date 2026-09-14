# MODNet Portrait Matting Python Sample

This sample demonstrates how to use the MODNet model on BPU for portrait matting tasks.

## Directory Structure

```text
.
├── main.py         # Inference entry script
├── modnet.py       # MODNet model wrapper
├── run.sh          # One-click execution script
├── README.md       # Usage instructions (English)
└── README_cn.md    # Usage instructions (Chinese)
```

## Parameter Description

| Parameter           | Description                            | Default Value                      |
|---------------------|----------------------------------------|------------------------------------|
| `--model-path`      | Path to the model file (.bin format)   | `../../model/modnet_512x512_rgb.bin` |
| `--test-img`        | Path to the test input image           | `../../test_data/person.jpg`       |
| `--bg-img`          | Path to the background image for compositing | `../../test_data/bg.jpg`    |
| `--matte-save-path` | Path to save the alpha matte           | `../../test_data/matte.png`        |
| `--img-save-path`   | Path to save the composited result image | `../../test_data/result.png`     |
| `--priority`        | Model scheduling priority (0~255)      | `0`                                |
| `--bpu-cores`       | BPU core indexes for inference         | `[0]`                              |
| `--ref-size`        | Target input resolution (longer side)  | `512`                              |

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
            --test-img path/to/img.jpg \
            --bg-img path/to/bg.jpg
        ```

## Interface Description

- **MODNetConfig**: Encapsulates model path and inference parameters.
- **MODNet**: Contains the complete inference pipeline (`pre_process`, `forward`, `post_process`, `predict`).

Refer to the [Source Reference Documentation](../../../../../docs/source_reference/README.md) for more details.
