English | [简体中文](README_cn.md)

# PP-LiteSeg model preparation

<a id="artifacts"></a>
## Artifacts

| Target | File | Format / use |
| --- | --- | --- |
| x5 | `pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin` | BIN, STDC1 Cityscapes inference |

Exact asset-id: `x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`. Source: X5 model manifest. No S-series assets or alternate variants are published.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── download.py  # Prepare model files
└── download.sh  # Model preparation command
```

<a id="preparation"></a>
## Preparation

```bash
# cwd: repository root; network required only for preparation
bash samples/vision/pp_liteseg/model/download.sh --target x5
```

If downloading fails, obtain the same file from the [published URL](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin) and place it at the path below. Do not replace it with another model under the same name.

<a id="accompanying-files"></a>
## Accompanying files

No external label file is needed: cli.py contains the fixed Cityscapes 19-class names and palette. The two images in../test_data are examples, not an accuracy dataset.

<a id="local-paths"></a>
## Local paths

Default runtime path: `samples/vision/pp_liteseg/model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`, resolved relative to the sample regardless of cwd. Downloader `--output-dir /data/models` changes only the download destination. To run that copy, pass both `--model-path /data/models/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin` and `--asset-id x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`. Relative custom paths use the current working directory.

<a id="formats-checksums"></a>
## Formats and checksums

`sha256: null (unknown)`. The publisher manifest has no trusted digest. Download preparation prints an observed digest, which can track local copies but cannot independently authenticate the publisher. Runtime validates file availability and tensor metadata; the asset ID selects the requested input/output contract. BIN is for X5, not interchangeable with Nash HBM.

## S100 / S100P / S600 logits variant

The new `stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104` variant uses the official trained PaddleSeg STDC1 Cityscapes checkpoint. Its S input is normalized RGB float32 NCHW `(1,3,512,1024)`: RGB/255, mean .5, std .5. The HBM exposes 19-class float logits; the Python runtime applies one CPU argmax. Select the exact converted variant or manifest asset. The existing X5 NV12/int32 class-map contract remains the default.

```bash
python3 samples/vision/pp_liteseg/model/download.py --target s100 --variant stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104
python3 samples/vision/pp_liteseg/runtime/python/main.py --target s100 --variant stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104 --test-img /path/to/street.png
```

Use `--list-models` to obtain the exact asset reference before an external `--model-path` override. S100P and S600 require their own target and HBM. Board smoke, Cityscapes accuracy, Runtime 1/2 concurrency and C++ 1/2 streams remain not run; compiler estimates and source-host checks are conversion evidence. See `conversion/converted_targets_20261103.json` and `conversion/source_export_20261104.json`.
