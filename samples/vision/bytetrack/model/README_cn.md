# ByteTrack 模型制品

<a id="artifacts"></a>
## 制品清单

ByteTrack 使用 S manifest 中的 YOLOv5x detector HBM，没有独立神经网络模型：

| target | asset ID | filename | URL | SHA-256 |
|---|---|---|---|---|
| S100 | `s:bytetrack:s100/yolov5x_672x672_nv12.hbm` | `s100/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S100P | `s:bytetrack:s100p/yolov5x_672x672_nv12.hbm` | `s100p/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100p/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S600 | `s:bytetrack:s600/yolov5x_672x672_nv12.hbm` | `s600/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |

tracker 代码和 CPU 依赖是独立 runtime 输入。发布校验值未知。

<a id="preparation"></a>
## 准备步骤

在仓库根目录选择 target，于允许联网的环境显式运行下载器；本轮未运行：

```bash
python3 -m samples.vision.bytetrack.model.download \
  --target s100 --output-dir samples/vision/bytetrack/model
```

成功时会打印精确 asset ID、嵌套 `Saved:` 路径和观测 digest，并创建 `samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm`。其他制品使用 `s100p` 或 `s600`。下载器不会获取输入视频。

<a id="accompanying-files"></a>
## 伴随文件

- `../test_data/coco_classes.names`：detector/可视化使用的 COCO 标签。
- `../test_data/bus.jpg`：静态图片 fixture，不是缺失的视频。
- `../test_data/readme_img/`：源参考 GIF/PNG。
- `../requirements-host.txt`：CPU tracker 包版本；板端 SDK 独立。

<a id="local-paths"></a>
## 本地路径

默认选择解析到 `samples/vision/bytetrack/model/<target>/yolov5x_672x672_nv12.hbm`。外部路径必须带同 target 的精确限定 asset ID。`track_test.mp4` 需另行放入 `test_data/` 或通过 `--input` 指定。

<a id="formats-checksums"></a>
## 格式与校验值

三个制品都是 672x672、split-NV12 YOLOv5x 的 `.hbm`。manifest URL 和 `sha256: null (unknown)` 是权威事实；本地观测 digest 不代表发布者认证。

<a id="license"></a>
## 许可

准备 helper 遵循 Apache-2.0；模型和上游 tracker 许可仍以各自来源为准。
