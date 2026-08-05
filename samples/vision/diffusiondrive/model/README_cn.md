[English](./README.md) | 简体中文

# DiffusionDrive 模型文件

不同平台的 HBM 放置路径如下：

| 平台 | March | 模型路径 |
| --- | --- | --- |
| RDK S100P | `nash-m` | `s100p/diffusiondrive_r34_256x1024_s100p.hbm` |
| RDK S600 | `nash-p` | `s600/diffusiondrive_r34_256x1024_s600.hbm` |

文件名遵循 `<model_name>_<input_resolution>_<chip_name>.hbm`。其中相机输入分辨率为 `256x1024`；模型还包含 LiDAR、ego status 和扩散噪声输入。

HBM 二进制由 Git 忽略。下载脚本会自动识别板卡并使用 Model Zoo 公共归档地址：

```bash
bash download_model.sh
CHIP=s100p bash download_model.sh
CHIP=s600 bash download_model.sh
```

两个模型均由 OpenExplorer v3.7.0 生成，采用 INT16 优先 PTQ 和 max 校准。GridSample 按工具链要求保持为 INT8，所有分段都在 BPU 上运行。下载脚本会根据 `SHA256SUMS` 自动校验所选模型。
