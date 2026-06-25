English | [简体中文](./README_cn.md)

# PaddleOCR Text Detection & Recognition Sample (C++)

This sample demonstrates how to run quantised **PP-OCRv6** models on the BPU for Chinese text detection and recognition. The two-stage OCR pipeline uses DB for text detection and CRNN+CTC for text recognition.

> `run.sh` reads `/sys/class/boardinfo/soc_name` and automatically selects the matching prebuilt model for the current board (S100 / S100P / S600). CMake also injects a `-DSOC_*` macro at build time so the default model path in `main.cpp` matches the board.

## Dependencies

This sample requires three sets of development files: gflags, polyclipping, and freetype. `run.sh` probes the corresponding headers / pkg-config first and only runs `apt install` when missing, handling the different package names between Ubuntu 22.04 (`libfreetype6-dev`) and 24.04/noble (`libfreetype-dev`).

Manual install:

```bash
# Ubuntu 22.04 / S100 image
sudo apt install libgflags-dev libpolyclipping-dev libfreetype6-dev

# Ubuntu 24.04 / S600 noble image
sudo apt install libgflags-dev libpolyclipping-dev libfreetype-dev
```

## Directory Structure

```text
.
├── inc/
│   └── paddle_ocr.hpp     # PaddleOCR wrapper interface & function declarations
├── src/
│   ├── paddle_ocr.cpp     # PaddleOCR inference & pre/post-processing
│   └── main.cpp           # Inference entry (argument parsing & flow control)
├── CMakeLists.txt         # CMake build configuration (SOC-aware)
├── run.sh                 # Run script (auto-install deps, download model, build, run)
└── README.md              # This file
```

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

The output binary is `build/paddle_ocr`. During configuration, CMake reads `soc_name` and passes `-DSOC_S100` or `-DSOC_S600` to the compiler (S100P also keeps `-DSOC_S100P`, but its default model path falls back to S100).

## Arguments

| Argument | Description | Default |
|---|---|---|
| `--det_model_path` | Detection model path (`.hbm`) | S100/S100P: `/opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm`<br>S600: `/opt/hobot/model/s600/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` |
| `--rec_model_path` | Recognition model path (`.hbm`) | S100/S100P: `/opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm`<br>S600: `/opt/hobot/model/s600/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` |
| `--test_image` | Test image path | `../../../test_data/gt_2322.jpg` |
| `--label_file` | Character dictionary (one char per line) | `../../../test_data/ppocrv6_dict.txt` |
| `--threshold` | Detection mask binarisation threshold (0.0–1.0) | `0.5` |
| `--ratio_prime` | Contour expansion ratio | `2.7` |
| `--img_save_path` | Result image save path | `result.jpg` |
| `--font_path` | TTF font path for rendering recognised text | `../../../test_data/FangSong.ttf` |

> **Note**: The default model path is determined at compile time by the SOC macro. To reuse the same binary across platforms, override with `--det_model_path` / `--rec_model_path` explicitly.

## Quick Run

### Option 1: One-click run (recommended)

```bash
cd runtime/cpp/
./run.sh
```

The script automates: SOC detection → dependency install → model download → build → inference.

### Option 2: Manual run

- Use defaults (paths match the SOC used at build time):

    ```bash
    cd build/
    ./paddle_ocr
    ```

- Custom arguments (S100 example):

    ```bash
    cd build/
    ./paddle_ocr \
        --det_model_path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
        --rec_model_path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
        --test_image ../../../test_data/gt_2322.jpg \
        --label_file ../../../test_data/ppocrv6_dict.txt \
        --font_path ../../../test_data/FangSong.ttf \
        --img_save_path result.jpg
    ```

### Output

On success, the result is saved to `build/`:

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

- **Platform compatibility**: `run.sh` supports RDK S100 / S100P / S600. S100P automatically reuses S100 prebuilt models; unknown/other SOCs fall back to S100 models.
- If model files do not exist, `run.sh` automatically downloads the matching platform variant from the D-Robotics download centre.
- Detection model input is NV12 (Y + UV dual input), size 640×640.
- Recognition model input is float32 RGB NCHW, size 48×320.
- Requires polyclipping (ClipperLib polygon expansion) and freetype (Chinese font rendering); see "Dependencies" above for the exact package names.