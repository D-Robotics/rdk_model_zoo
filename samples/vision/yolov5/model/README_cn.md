# YOLOv5 模型制品

<a id="artifacts"></a>
## 制品清单

active manifest 发布 9 个 X5 制品和 2 个 S 制品。下表的 `{size}` 只是表示 `s`、`m`、`l`、`x` 四种大小的文档简写，不能原样作为文件名。

| target | variant | filename | format | URL | SHA-256 |
|---|---|---|---|---|---|
| X5 | n-v7.0 | `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | s-v2.0 | `yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | m-v2.0 | `yolov5m_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5m_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | l-v2.0 | `yolov5l_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5l_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | x-v2.0 | `yolov5x_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5x_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | s-v7.0 | `yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | m-v7.0 | `yolov5m_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5m_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | l-v7.0 | `yolov5l_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5l_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | x-v7.0 | `yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| S100 | x-672 | `s100/yolov5x_672x672_nv12.hbm` | hbm | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S600 | x-672 | `s600/yolov5x_672x672_nv12.hbm` | hbm | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |

S100P 没有 YOLOv5 制品。所有 manifest 发布校验值均为 `sha256: null (unknown)`。

<a id="preparation"></a>
## 准备步骤

在允许显式联网的环境中从仓库根目录运行；本轮没有执行：

```bash
python3 -m samples.vision.yolov5.model.download \
  --target x5 --variant n-v7.0 \
  --output-dir samples/vision/yolov5/model
```

S100/S600 使用 `--target s100 --variant x-672` 或 `--target s600 --variant x-672`。脚本会创建目标子目录，并打印观测 digest 和发布值（`unknown`），只下载所选 manifest 行。`run.sh` 和 runtime 不会调用它。

<a id="accompanying-files"></a>
## 伴随文件

- `../test_data/coco_classes.names`：仅用于可视化的 80 行 COCO 标签。
- `../test_data/bus.jpg`：X5 默认图；`../test_data/kite.jpg`：S 默认图。
- `conversion/` YAML：X5 源转换配置参考，不是模型二进制。

<a id="local-paths"></a>
## 本地路径

执行上述命令后，X5 文件位于 `samples/vision/yolov5/model/`；S 文件位于 `samples/vision/yolov5/model/s100/` 或 `model/s600/`。自定义 `--model-path` 必须和精确匹配的 `--asset-id` 一起使用，不能仅凭文件名猜 target。

<a id="formats-checksums"></a>
## 格式与校验值

X5 制品是 640x640 Bayes-e/NV12 的 `.bin`；S 制品是 672x672、拆分 NV12 输入的目标子目录 `.hbm`。manifest 没有发布 SHA-256；下载器打印的本地观测 hash 只能证明本地字节。

<a id="license"></a>
## 许可

仓库 helper 遵循 Apache-2.0；模型来源和上游权重许可仍以 manifest/源发布为准。
