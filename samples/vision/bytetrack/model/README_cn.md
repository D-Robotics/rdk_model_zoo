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

发布校验值见模型清单；未知校验值不用于完整性认证。

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

在仓库根目录选择 target，于允许联网的环境显式运行下载器：

```bash
python3 -m samples.vision.bytetrack.model.download --target x5 --variant n-v7.0
```

S100：

```bash
python3 -m samples.vision.bytetrack.model.download \
  --target s100 --output-dir samples/vision/bytetrack/model
```

成功时会打印精确 asset ID、嵌套 `Saved:` 路径和观测 digest，并创建 `samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm`。其他制品使用 `s100p` 或 `s600`。下载器不会获取输入视频。

<a id="accompanying-files"></a>
## 伴随文件

- `../test_data/coco_classes.names`：detector/可视化使用的 COCO 标签。
- `../test_data/bus.jpg`：示例静态图片。
- `../test_data/readme_img/`：源参考 GIF/PNG。
- `../requirements-host.txt`：CPU tracker 包版本；板端 SDK 独立。

<a id="local-paths"></a>
## 本地路径

X5 模型保存在 `samples/vision/bytetrack/model/<BIN文件名>`；S 系列默认选择解析到 `samples/vision/bytetrack/model/<target>/yolov5x_672x672_nv12.hbm`。外部路径必须带同 target 的精确限定 asset ID。`track_test.mp4` 需另行放入 `test_data/` 或通过 `--input` 指定。

<a id="formats-checksums"></a>
## 格式与校验值

X5 制品为 640x640 packed-NV12 `.bin`；三个 S 制品都是 672x672、split-NV12 YOLOv5x 的 `.hbm`。manifest URL 和 `sha256: null (unknown)` 为权威记录；本地观测 digest 用于跨主机识别同一份字节，发布者认证以发布摘要为准。

<a id="license"></a>
## 许可

准备 helper 遵循 Apache-2.0；模型和上游 tracker 许可仍以各自来源为准。
