English | [简体中文](README_cn.md)

# Faster R-CNN input image

<a id="overview"></a>
## Input

`bus.jpg` is an 810×1080 RGB JPEG from the repository’s YOLOv5 Sample. OpenCV reads it as BGR uint8.

<a id="directory"></a>
## Files

```text
bus.jpg  # Detection input
```

<a id="usage"></a>
## Run the image

Prepare the paired models, then run from the repository root:
```bash
python3 samples/vision/fasterrcnn/runtime/python/main.py --target x5 \
  --parta-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_partb.bin
```
The default output is `test_data/result.jpg` in this Sample. For person video tracking, use `samples/vision/bytetrack/test_data/track_test.mp4` with `--video`.

