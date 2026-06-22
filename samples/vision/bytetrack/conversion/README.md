English | [简体中文](./README_cn.md)

# ByteTrack Model Conversion Guide

ByteTrack itself does not contain a neural network that must be compiled into HBM. The runtime depends on an upstream detector. This sample uses the YOLOv5x pedestrian detector `yolov5x_672x672_nv12.hbm`, while the tracking logic runs in Python.

## Detector Model

The runtime detector model is:

```text
samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm
```

Model download script:

```bash
cd samples/vision/bytetrack/model
bash download_model.sh s100
```

This detector comes from the RDK S100 `ultralytics_YOLO` model resources and uses NV12 two-input preprocessing:

- Y plane
- UV plane

ByteTrack consumes only person boxes, scores, and class IDs from the detector output. The current sample tracks only the COCO `person` class (class id 0).

## Conversion Reference

For YOLO detector ONNX export, PTQ configuration generation, and model compilation, refer to the Ultralytics YOLO sample conversion guide and the examples provided in the OE package.

Applicable detector families include YOLOv5u, YOLOv8, YOLO11, YOLO12, and related detection models. When connecting another detector to ByteTrack, make sure the runtime output can be converted to `(x1, y1, x2, y2, score, class_id)` or an equivalent wrapper format.

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore environment. Model conversion is not intended to run on the board.

- OE resource entry point (Docker + OE development package): <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
