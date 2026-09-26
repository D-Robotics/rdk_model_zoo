English | [简体中文](README_cn.md)

# PP-LiteSeg model preparation

<a id="artifacts"></a>
## Artifacts

| Target | File | Format / use |
| --- | --- | --- |
| x5 | `pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin` | BIN, STDC1 Cityscapes inference |

Exact asset-id: `x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`. Source: [X5 model manifest](../../../../platforms/x5/docs/release/models.yaml). No S-series assets or alternate variants are published.

<a id="preparation"></a>
## Preparation

```bash
# cwd: repository root; network required only for preparation
bash samples/vision/pp_liteseg/model/download.sh --target x5
```

If downloading fails, obtain the same file from the [published URL](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin) and place it at the path below. Do not replace it with another model under the same name.

<a id="accompanying-files"></a>
## Accompanying files

No external label file is needed: visualization.py contains the fixed Cityscapes 19-class names and palette. The two images in ../test_data are examples, not an accuracy dataset.

<a id="local-paths"></a>
## Local paths

Default runtime path: `samples/vision/pp_liteseg/model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`, resolved relative to the sample regardless of cwd. Downloader `--output-dir /data/models` changes only the download destination. To run that copy, pass both `--model-path /data/models/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin` and `--asset-id x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`. Relative custom paths use the current working directory.

<a id="formats-checksums"></a>
## Formats and checksums

`sha256: null (unknown)`. The publisher manifest has no trusted digest. Download preparation prints an observed digest, which can track local copies but cannot independently authenticate the publisher. Runtime validates file availability and tensor metadata; exact asset-id asserts the requested contract, not proof of custom file provenance. BIN is for X5, not interchangeable with Nash HBM.
