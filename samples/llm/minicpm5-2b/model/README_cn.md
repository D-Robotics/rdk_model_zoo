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
