# ByteTrack 多目标跟踪示例

## 简介 (Introduction)
ByteTrack 是一种简单、快速且强大的多目标跟踪算法。本示例演示了如何在 RDK 平台上结合 YOLOv5 检测模型和 ByteTrack 算法进行实时多目标跟踪。

## 算法介绍 (Algorithm Overview)
ByteTrack 的核心创新在于关联方法 BYTE。它通过关联每个检测框（不仅仅是高分框）来处理遮挡和低分检测。本示例使用 YOLOv5x 作为检测器，从视频流中提取目标（如行人），然后使用 ByteTrack 进行轨迹关联。

## 目录结构 (Directory Structure)
```bash
.
|-- model             # 模型文件与下载脚本
|-- runtime           # 推理示例
|   `-- python        # Python 推理实现
|-- test_data         # 示例输入视频
|-- conversion        # 模型转换说明
|-- evaluator         # 模型评估说明
`-- README.md         # 当前模型总览说明
```

## 快速体验 (QuickStart)

### Python
进入 `runtime/python` 目录并运行脚本：
```bash
cd runtime/python
bash run.sh
```
该脚本会自动下载 YOLOv5x 模型和测试视频，并在视频上运行跟踪。

## 模型转换 (Model Conversion)
ByteTrack 依赖于目标检测模型的输出。本示例默认使用 YOLOv5x 作为检测器。

## 模型推理 (Runtime)
本仓库提供 ByteTrack 的 Python 推理示例。详细的接口说明和运行方式请参考：
- [Python Runtime](runtime/python/README.md)

## License
遵循 ModelZoo 顶层 License。