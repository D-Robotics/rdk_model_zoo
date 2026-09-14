# PP-LiteSeg-STDC1 Semantic Segmentation Python Sample

This sample demonstrates how to use the quantized PP-LiteSeg-STDC1 model on BPU for real-time semantic segmentation.

## Directory Structure

```text
.
├── main.py          # Inference entry script
├── pp_liteseg.py    # PP-LiteSeg model wrapper (PPLiteSegConfig + PPLiteSeg)
├── run.sh           # One-click execution script
├── README.md        # Usage instructions (English)
└── README_cn.md     # Usage instructions (Chinese)
```

## Requirements

- RDK X5 board with firmware >= 3.5.0
- `hbm_runtime` Python package (pre-installed on board)
- `opencv-python` (`pip3 install opencv-python-headless`)

## Parameter Description

| Parameter        | Description                                    | Default                                               |
|------------------|------------------------------------------------|-------------------------------------------------------|
| `--model-path`   | Path to the BPU `.bin` model file              | `../../model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin` |
| `--test-img`     | Path to the test input image                   | `../../test_data/street.jpg`                          |
| `--output`       | Path to save the result image                  | `../../test_data/result.jpg`                          |
| `--alpha`        | Overlay blending alpha (0~1)                   | `0.55`                                                |
| `--input-width`  | Model input width                              | `1024`                                                |
| `--input-height` | Model input height                             | `512`                                                 |

## Quick Run

- **One-click Execution Script**
    ```bash
    chmod +x run.sh
    ./run.sh
    ```
    The script downloads the model automatically if it is not present.

- **Manual Execution**
    - Use default parameters
        ```bash
        python3 main.py
        ```
    - Specify image and output
        ```bash
        python3 main.py \
            --model-path ../../model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin \
            --test-img ../../test_data/street.jpg \
            --output ../../test_data/result.jpg
        ```

## Output

The result image is a 3-panel visualization:

```text
| Original | Overlay (alpha blend) | Segmentation (pure color) |
```

A class legend is embedded in the top-right corner of the overlay panel.

## Interface Description

- **PPLiteSegConfig**: Encapsulates model path and inference parameters.
- **PPLiteSeg**: Contains the complete inference pipeline (`pre_process`, `forward`, `post_process`, `predict`, `visualize`).

Refer to the [Source Reference Documentation](../../../../../docs/source_reference/README.md) for more details.
