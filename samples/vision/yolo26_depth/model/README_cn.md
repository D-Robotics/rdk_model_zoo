[English](./README.md) | [简体中文](./README_cn.md)

# 模型文件

量化后的 RDK X5 模型下载到 `model/bayes-e/`。仓库中不直接保存模型二进制文件。

## 模型列表

| 规格 | 文件 | SHA256 |
| --- | --- | --- |
| N | `yolo26n_depth_bayese_768x768_nv12.bin` | `e55091eb594e20e37e6c36a36cce42a94ad80ec651ae893a2143cd2273ed9b0b` |
| S | `yolo26s_depth_bayese_768x768_nv12.bin` | `0e43958195f504d7a8ac48b1c99f4802cd9a4c3580321bfb251d0e0f892ccf4c` |
| M | `yolo26m_depth_bayese_768x768_nv12.bin` | `f4f2f1958dc16324932b4492490209c817cf7565c3c29240bcf4f0012f9c0be0` |
| L | `yolo26l_depth_bayese_768x768_nv12.bin` | `6a5fa40bda20ee56208ca6e594ecfd9781329385d0baf1b15c9eaa9625286d14` |
| X | `yolo26x_depth_bayese_768x768_nv12.bin` | `61798227fb7e0772a739b483ae5b5acd58a8e785dd7fd9aec5dcac7db0903d91` |

所有规格均接收 `1×768×768×3` NV12 pyramid 输入，并输出 `1×192×192×1` float32 calibrated log-depth。

## 下载方式

脚本使用正式 archive 下载地址。如需使用内部镜像，可覆盖 `MODEL_BASE_URL` 后执行：

```bash
bash download_model.sh
```

也可以只下载指定规格：

```bash
bash download_model.sh n s
```

脚本会使用上表中的 SHA256 校验每个下载文件。
