[English](README.md) | 简体中文

# UNetMobileNet 模型准备

<a id="artifacts"></a>
## 制品清单

| Target | model/ 下文件 | 精确 asset-id |
| --- | --- | --- |
| s100 | s100/unet_mobilenet_1024x2048_nv12.hbm | s:unetmobilenet:s100/unet_mobilenet_1024x2048_nv12.hbm |
| s600 | s600/unet_mobilenet_1024x2048_nv12.hbm | s:unetmobilenet:s600/unet_mobilenet_1024x2048_nv12.hbm |

两者均为发布清单中的 HBM 部署制品，文件名相同但字节互不通用；请按 target 对应取用。S100P/X5 暂无发布制品。

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── download.py  # 准备模型文件
├── download.sh  # 模型准备命令
└── download_model.sh  # Shell 脚本
```

<a id="preparation"></a>
## 准备步骤

```bash
# cwd: repository root; explicit network preparation
bash samples/vision/unetmobilenet/model/download.sh --target s100
bash samples/vision/unetmobilenet/model/download.sh --target s600
```

`download_model.sh` 为快捷入口，转发相同参数。下载失败可重试，或从 [S100](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/unetmobilenet/unet_mobilenet_1024x2048_nv12.hbm)／[S600](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/unetmobilenet/unet_mobilenet_1024x2048_nv12.hbm) 手动获取并放入对应子目录。

<a id="accompanying-files"></a>
## 伴随文件

19 类 ID 使用内置 rdk_colors 绘制，无需外部类别文件。segmentation.png 是示例输入，result.jpg 是参考输出；数据集评估需准备带标注的 Cityscapes 数据。

<a id="local-paths"></a>
## 本地路径

运行时默认按示例位置解析 model/<target>/unet_mobilenet_1024x2048_nv12.hbm，不依赖 cwd。下载器 --output-dir 修改根目录，但保留目标子目录。使用已有 /opt 副本时，同时指定 --model-path /opt/hobot/model/s100/basic/unet_mobilenet_1024x2048_nv12.hbm 与 --asset-id s:unetmobilenet:s100/unet_mobilenet_1024x2048_nv12.hbm，且目标须一致。

<a id="formats-checksums"></a>
## 格式与校验值

两行 HBM 均为 `sha256: null (unknown)`。下载器打印本地 SHA-256。请选择目标板卡对应的发布模型；加载时校验模型输入输出的 shape 和 dtype。
