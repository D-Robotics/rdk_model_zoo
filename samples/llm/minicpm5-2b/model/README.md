[English](README.md) | [简体中文](README_cn.md)

# Model files

Run from this directory on the host or board. Allow 6 GB free space. The helper verifies the archive, a pinned SHA256SUMS manifest, and every extracted member. An existing directory is reused only if all checks pass; a different existing directory is left intact and reported as an error.

```bash
bash download_model.sh
# Optional destination; the checksum remains fixed.
MODEL_DIR=/data/minicpm5-s600 bash download_model.sh
```

Archive: `minicpm5-2b_s600_oellm2_w8_ctx4096_20260908.tar.gz`

SHA256: `8f2bef6fc7d2290f05055570e7dcfb1cf4e8c07c9a6bb0d0acc24dd202a83841`

Size: 2457817532 bytes.

[Download model](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/minicpm5-2b_s600_oellm2_w8_ctx4096_20260908.tar.gz)

The default destination is `model/s600`. It contains the HBM, FP16 embedding, tokenizer files, model metadata, LICENSE, NOTICE and MODEL_INFO.json. SDK libraries are not included. The historical `rdk_s100` server directory in this URL does not change the artifact target: this HBM is S600/Nash-p only.
