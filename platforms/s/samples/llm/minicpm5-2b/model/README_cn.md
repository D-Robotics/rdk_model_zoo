[English](README.md) | [简体中文](README_cn.md)

# 模型文件

在主机或板卡的本目录执行，需要 6 GB 可用空间。脚本校验压缩包、固定摘要的 SHA256SUMS 清单和每个解压文件。已有目录只有全部校验通过才会复用；不匹配的已有目录保持原样并报错。

```bash
bash download_model.sh
# Optional destination; the checksum remains fixed.
MODEL_DIR=/data/minicpm5-s600 bash download_model.sh
```

Archive: `minicpm5-2b_s600_oellm2_w8_ctx4096_20260908.tar.gz`

SHA256: `8f2bef6fc7d2290f05055570e7dcfb1cf4e8c07c9a6bb0d0acc24dd202a83841`

Size: 2457817532 bytes.

[Download model](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/minicpm5-2b_s600_oellm2_w8_ctx4096_20260908.tar.gz)

默认输出到 `model/s600`，包含 HBM、FP16 embedding、tokenizer、模型元数据、LICENSE、NOTICE 和 MODEL_INFO.json，不包含 SDK 运行库。本产物只适用于 S600/Nash-p。

## S100 / S100P

```bash
BOARD=s100 bash download_model.sh
BOARD=s100p bash download_model.sh
```

每个目标有独立 HBM 和 tokenizer。默认目录分别为 `model/s100`、`model/s100p`。使用 [旧版 runtime](../runtime/legacy/README_cn.md)，不能使用 S600 2.0 入口。压缩包不包含 SDK；SHA256 固定在脚本内，解压后再次逐文件校验。

- [S100 archive](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/minicpm5-2b_s100_oellm1_w8_ctx4096_20260909.tar.gz): 2275420219 bytes; SHA256 `dc130ae21dc1f1cc266c526e090b03b4d3be4af8cf4157c4b5841b7f47ab818d`.

- [S100P archive](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/minicpm5-2b_s100p_oellm1_w8_ctx4096_20260909.tar.gz): 2270216392 bytes; SHA256 `65c89ae7ead48711a392fa556f602358ffffe16438e94051d1f8cca46d4e096a`.

```text
model/<board>/
  minicpm5-2b_ctx4096_<board>.hbm
  tokenizer/
  LICENSE
  NOTICE
  MODEL_INFO.json
  SHA256SUMS
```
