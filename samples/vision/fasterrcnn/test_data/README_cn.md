[English](README.md) | 简体中文

# Faster R-CNN input image

<a id="overview"></a>
## 输入

`bus.jpg` 是来自仓库 YOLOv5 Sample 的 810×1080 RGB JPEG，OpenCV 将其读取为 BGR uint8。

<a id="directory"></a>
## 文件

```text
bus.jpg  # 检测输入
```

<a id="usage"></a>
## 运行图片

准备模型组合后，在仓库根目录执行：
```bash
python3 samples/vision/fasterrcnn/runtime/python/main.py --target x5 \
  --parta-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_partb.bin
```
默认输出为本 Sample 的 `test_data/result.jpg`。行人视频可使用 `samples/vision/bytetrack/test_data/track_test.mp4`，通过 `--video` 指定。

