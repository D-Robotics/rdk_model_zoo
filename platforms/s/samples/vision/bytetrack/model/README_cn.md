[English](./README.md) | 简体中文

# ByteTrack 模型文件

本目录用于下载和存放 ByteTrack sample 使用的 YOLOv5x HBM 检测模型。

## 下载命令

```bash
cd samples/vision/bytetrack/model
bash download_model.sh s100
```

下载成功后生成：

```text
model/s100/yolov5x_672x672_nv12.hbm
```

## 模型说明

| 文件 | 说明 |
| --- | --- |
| `s100/yolov5x_672x672_nv12.hbm` | YOLOv5x COCO 检测模型，ByteTrack sample 只使用 `person` 类检测结果 |

## License

本目录遵循 [Apache 2.0 License](../../../../LICENSE)。
