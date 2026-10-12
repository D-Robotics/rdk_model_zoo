English | [简体中文](README_cn.md)

# PaddleOCR model preparation

<a id="artifacts"></a>
## Models by board

Prepare the detector and recognizer as a pair. X5 and S100 models are downloadable; S600 C++ inference uses the matching models supplied with the board image or BSP model package.

| Board | Detector | Recognizer | Preparation |
| --- | --- | --- | --- |
| X5 | `en_PP-OCRv3_det_640x640_nv12.bin` | `en_PP-OCRv3_rec_48x320_rgb.bin` | Python `--prepare`, PP-OCRv3 English |
| S100 | `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | Python `--prepare`, PP-OCRv6 |
| S600 | `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | S600 BSP models under `/opt/hobot/model/s600/basic/`; C++ only |

The download URLs and checksums for X5/S100 are in `docs/release/x5/models.yaml` and `docs/release/s/models.yaml`. S600 models are obtained from the S600 board image or its model package; use the S600 build. S100P has no supported model pair.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md
└── README_cn.md
```

<a id="preparation"></a>
## Preparation

Run the command for your board from the repository root. Preparation works on a host with Python and PyYAML and prints the saved files and their SHA-256 values.

X5:

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --model-dir samples/vision/paddle_ocr/model
```

S100:

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target s100 \
  --det-asset-id s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-asset-id s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --model-dir samples/vision/paddle_ocr/model/s100
```

For S600, check the model pair on the S600 board:

```bash
test -s /opt/hobot/model/s600/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
test -s /opt/hobot/model/s600/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

If these files are absent, obtain the matching pair from the S600 image or BSP model package and place them at these paths. To keep them elsewhere, pass `--det-model-path` and `--rec-model-path` to the [C++ entry](../runtime/cpp/README.md#run). The Python preparation command accepts X5 and S100.

<a id="accompanying-files"></a>
## Accompanying files

| Board | Dictionary | Location |
| --- | --- | --- |
| X5 | 96-character English alphabet | `X5_ALPHABET` in `runtime/python/cli.py` |
| S100 / S600 | PP-OCRv6 UTF-8 dictionary | [`ppocrv6_dict.txt`](../test_data/s100/ppocrv6_dict.txt), 18,708 lines |

The decoder adds a blank class and, for PP-OCRv6, a trailing space. The Python `--vocabulary-path` requires the dictionary checksum below. C++ rendering uses `../test_data/FangSong.ttf`; choose a different font with `--font-path`.

<a id="local-paths"></a>
## Model paths

- X5 Python defaults to `model/<BIN filename>` under this sample.
- S100 Python defaults to `model/s100/<HBM filename>`; the commands above save to these locations.
- C++ defaults to `/opt/hobot/model/<board>/basic/<HBM filename>` for S100 or S600.
- With a custom Python destination, pass both model paths and their corresponding asset IDs. `--model-dir` sets the preparation directory and saves the pair using their base filenames.

<a id="formats-checksums"></a>
## Formats and checksums

| File | SHA-256 | Status |
| --- | --- | --- |
| `en_PP-OCRv3_det_640x640_nv12.bin` | null | no publisher digest recorded for the manifest row |
| `en_PP-OCRv3_rec_48x320_rgb.bin` | null | no publisher digest recorded |
| `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | null | no publisher digest recorded |
| `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | null | no publisher digest recorded |
| `test_data/s100/ppocrv6_dict.txt` | `b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d` | recorded, enforced for `--vocabulary-path` replacements |

The publisher has not supplied model checksums. The preparation command prints the downloaded file SHA-256.
