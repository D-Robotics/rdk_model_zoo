English | [简体中文](./README_cn.md)

# PaddleOCR Text Detection & Recognition Sample (Python)

This sample demonstrates how to run quantised **PP-OCRv6** models on the BPU for Chinese text detection and recognition. The two-stage OCR pipeline uses DB for text detection and CRNN+CTC for text recognition.

> `run.sh` / `main.py` reads `/sys/class/boardinfo/soc_name` and automatically selects the matching prebuilt model for the current board (S100 / S100P / S600).

## Dependencies

This sample requires: `numpy`, `opencv-python`, `pyclipper`, `Pillow`.

`run.sh` only installs a package when the import fails, preserving any newer pre-installed versions (e.g. Pillow ≥10 or newer pyclipper on S600 noble images). To manually install fallback versions:

```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86 pyclipper==1.3.0.post6 Pillow==9.0.1
```

## Directory Structure

```text
.
├── paddle_ocr.py   # PaddleOCR inference wrapper (detection & recognition + helpers)
├── main.py         # Inference entry (argument parsing, SOC-aware default model paths)
├── run.sh          # Run script (auto-download dependencies, models, and execute)
└── README.md       # This file
```

## Arguments

| Argument | Description | Default |
|---|---|---|
| `--det-model-path` | Detection model path (`.hbm`) | S100/S100P: `/opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm`<br>S600: `/opt/hobot/model/s600/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` |
| `--rec-model-path` | Recognition model path (`.hbm`) | S100/S100P: `/opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm`<br>S600: `/opt/hobot/model/s600/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` |
| `--test-img` | Test image path | `../../test_data/gt_2322.jpg` |
| `--label-file` | Character dictionary (one char per line) | `../../test_data/ppocrv6_dict.txt` |
| `--threshold` | Detection mask binarisation threshold (0.0–1.0) | `0.5` |
| `--ratio-prime` | Contour expansion ratio | `2.7` |
| `--img-save-path` | Result image save path | `result.jpg` |
| `--priority` | Model scheduling priority (0~255) | `0` |
| `--bpu-cores` | BPU core indexes | `[0]` |

> **Note**: The font file is fixed to `../../test_data/FangSong.ttf` and is not exposed as a CLI argument.

## Quick Run

### Option 1: One-click run (recommended)

```bash
cd runtime/python/
./run.sh
```

The script automates: SOC detection → environment check → model download → inference.

### Option 2: Manual run

- Use defaults (paths match the current SOC):

    ```bash
    python3 main.py
    ```

- Custom arguments (S100 example):

    ```bash
    python3 main.py \
        --det-model-path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
        --rec-model-path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
        --test-img ../../test_data/gt_2322.jpg \
        --label-file ../../test_data/ppocrv6_dict.txt \
        --img-save-path result.jpg \
        --threshold 0.5 \
        --ratio-prime 2.7
    ```

### Output

On success, the result is saved to the current directory:

```text
[0] Prediction: 示例文字
[1] Prediction: 另一行文字
...
[Saved] Result saved to: result.jpg
```

- `result.jpg`: left half shows the original image with detection boxes overlaid, right half shows recognised text rendered on a white canvas.

## API Reference

See the [source reference docs](../../../../../docs/source_reference/README.md) for detailed API documentation.

## Notes

- **Platform compatibility**: `run.sh` / `main.py` support RDK S100 / S100P / S600. S100P automatically reuses S100 prebuilt models; unknown/other SOCs fall back to S100 models.
- If model files do not exist, `run.sh` automatically downloads the matching platform variant from the D-Robotics download centre.
- `pyclipper` (polygon expansion) and `Pillow` (Chinese font rendering) are required.
- Font rendering uses FangSong (`FangSong.ttf`), path fixed to `../../test_data/FangSong.ttf`.
- Recognition uses a pure NumPy CTC greedy decoder without any PaddlePaddle dependency.