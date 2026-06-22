English | [简体中文](./README_cn.md)

# ByteTrack Model Files

This directory downloads and stores the YOLOv5x HBM detector used by the ByteTrack sample.

## Download Command

```bash
cd samples/vision/bytetrack/model
bash download_model.sh s100
```

After a successful download, the model file is:

```text
model/s100/yolov5x_672x672_nv12.hbm
```

## Model Description

| File | Description |
| --- | --- |
| `s100/yolov5x_672x672_nv12.hbm` | YOLOv5x COCO detector. The ByteTrack sample only uses `person` detections. |

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
