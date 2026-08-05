[English](./README.md) | 简体中文

# DiffusionDrive 模型文件

将 S600 模型放到：

```text
s600/diffusiondrive_r34_256x1024_s600.hbm
```

文件名遵循 `<model_name>_<input_resolution>_<chip_name>.hbm`。其中相机输入分辨率为 `256x1024`；模型还包含 LiDAR、ego status 和扩散噪声输入。

HBM 二进制由 Git 忽略。可使用内部或正式发布地址下载：

```bash
MODEL_URL=<accessible-hbm-url> bash download_model.sh
```

当前验证产物由 OpenExplorer v3.7.0 面向 `nash-p` / S600 生成，采用全 INT16 PTQ 和 max 校准。下载后可在本目录执行 `sha256sum -c SHA256SUMS` 校验模型。
