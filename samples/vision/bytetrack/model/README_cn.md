[English](README.md) | 简体中文

# ByteTrack 模型制品

<a id="artifacts"></a>
## 制品清单

ByteTrack 使用 YOLOv5 检测模型，跟踪部分在 CPU 上执行。X5 复用已发布的 YOLOv5 640×640 BIN，默认 `n-v7.0`：

| 变体 | asset ID | 文件名 |
|---|---|---|
| `n-v7.0` | `x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` | `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` |
| `s-v2.0` | `x5:yolov5:yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` | `yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` |
| `m-v2.0` | `x5:yolov5:yolov5m_tag_v2.0_detect_640x640_bayese_nv12.bin` | `yolov5m_tag_v2.0_detect_640x640_bayese_nv12.bin` |
| `l-v2.0` | `x5:yolov5:yolov5l_tag_v2.0_detect_640x640_bayese_nv12.bin` | `yolov5l_tag_v2.0_detect_640x640_bayese_nv12.bin` |
| `x-v2.0` | `x5:yolov5:yolov5x_tag_v2.0_detect_640x640_bayese_nv12.bin` | `yolov5x_tag_v2.0_detect_640x640_bayese_nv12.bin` |
| `s-v7.0` | `x5:yolov5:yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin` | `yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin` |
| `m-v7.0` | `x5:yolov5:yolov5m_tag_v7.0_detect_640x640_bayese_nv12.bin` | `yolov5m_tag_v7.0_detect_640x640_bayese_nv12.bin` |
| `l-v7.0` | `x5:yolov5:yolov5l_tag_v7.0_detect_640x640_bayese_nv12.bin` | `yolov5l_tag_v7.0_detect_640x640_bayese_nv12.bin` |
| `x-v7.0` | `x5:yolov5:yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin` | `yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin` |

下载地址与校验信息见 [YOLOv5 模型清单](../../yolov5/model/README_cn.md)。S 系列使用以下 HBM：

| target | asset ID | filename | URL | SHA-256 |
|---|---|---|---|---|
| S100 | `s:bytetrack:s100/yolov5x_672x672_nv12.hbm` | `s100/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S100P | `s:bytetrack:s100p/yolov5x_672x672_nv12.hbm` | `s100p/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100p/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S600 | `s:bytetrack:s600/yolov5x_672x672_nv12.hbm` | `s600/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |

表中 `null` 表示发布方未提供 SHA-256。

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

在可联网的环境中，从仓库根目录运行下载命令：

```bash
python3 -m samples.vision.bytetrack.model.download --target x5 --variant n-v7.0
```

S100：

```bash
python3 -m samples.vision.bytetrack.model.download \
  --target s100 --output-dir samples/vision/bytetrack/model
```

下载完成后显示模型标识、保存路径和文件 SHA-256。上述 S100 命令将模型保存到 `samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm`。S100P 和 S600 分别使用 `--target s100p` 和 `--target s600`。

<a id="accompanying-files"></a>
## 伴随文件

- `../test_data/coco_classes.names`：检测和画框使用的 COCO 标签。
- `../test_data/bus.jpg`：示例静态图片。
- `../test_data/readme_img/`：跟踪效果 GIF/PNG。
- `../requirements-host.txt`：Python 跟踪依赖。

<a id="local-paths"></a>
## 本地路径

X5 模型保存在 `samples/vision/bytetrack/model/<BIN文件名>`，S 系列保存在 `samples/vision/bytetrack/model/<target>/yolov5x_672x672_nv12.hbm`。使用其他路径时，传入 `--model-path` 和表中对应的完整 `--asset-id`。视频路径通过 `--input` 指定。

<a id="formats-checksums"></a>
## 格式与校验值

X5 使用 640×640 packed-NV12 `.bin` 模型，S 系列使用 672×672 split-NV12 `.hbm` 模型。下载器显示本地文件的 SHA-256；有发布校验值时，可用它核对下载文件。

<a id="license"></a>
## 许可

下载脚本采用 Apache-2.0 许可。模型权重和 ByteTrack 跟踪源码遵循各自的上游许可。
