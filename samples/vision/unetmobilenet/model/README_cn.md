[English](README.md) | 简体中文

# UNetMobileNet 模型准备

<a id="artifacts"></a>
## 制品清单

| Target | model/ 下文件 | 精确 asset-id |
| --- | --- | --- |
| s100 | s100/unet_mobilenet_1024x2048_nv12.hbm | s:unetmobilenet:s100/unet_mobilenet_1024x2048_nv12.hbm |
| s600 | s600/unet_mobilenet_1024x2048_nv12.hbm | s:unetmobilenet:s600/unet_mobilenet_1024x2048_nv12.hbm |

两者均为 HBM 部署制品。[发布清单](../../../../platforms/s/docs/release/models.yaml)。同名不代表模型字节可以互换，没有发布的 S100P/X5 制品。

<a id="preparation"></a>
## 准备步骤

```bash
# cwd: repository root; explicit network preparation
bash samples/vision/unetmobilenet/model/download.sh --target s100
bash samples/vision/unetmobilenet/model/download.sh --target s600
```

`download_model.sh` 为兼容入口，转发相同参数。下载失败可重试，或从 [S100](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/unetmobilenet/unet_mobilenet_1024x2048_nv12.hbm)／[S600](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/unetmobilenet/unet_mobilenet_1024x2048_nv12.hbm) 手动获取并放入对应子目录。准备成功不证明本机硬件身份。

<a id="accompanying-files"></a>
## 伴随文件

无需外部类别文件；固定的 19 类 ID 使用源 rdk_colors 显示。segmentation.png 是示例输入，result.jpg 是历史输出，都不构成带标签验证集。

<a id="local-paths"></a>
## 本地路径

运行时默认按示例位置解析 model/<target>/unet_mobilenet_1024x2048_nv12.hbm，不依赖 cwd；替代旧 /opt/hobot/model/<soc>/basic 默认值。下载器 --output-dir 修改根目录，但保留目标子目录。使用已有 /opt 副本时，同时指定 --model-path /opt/hobot/model/s100/basic/unet_mobilenet_1024x2048_nv12.hbm 与 --asset-id s:unetmobilenet:s100/unet_mobilenet_1024x2048_nv12.hbm，且目标须一致。

<a id="formats-checksums"></a>
## 格式与校验值

两行 HBM 均为 `sha256: null (unknown)`。下载器打印实测 SHA-256 以追踪副本，不构成独立认证。运行时元数据校验可拒绝错误 shape/dtype，但无法证明自定义文件就是发布制品。请保留自定义文件来源，不能通过改名声称 S100 制品支持 S600。
