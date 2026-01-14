# YOLO26 RDK Model Zoo 示例

本目录包含了在 RDK X5 硬件上导出、转换和运行 YOLO26 模型推理的脚本。

## 文件说明
- `export_yolo26_bpu.py`: 将 YOLO26 `.pt` 模型导出为针对 BPU 优化的 `.onnx` 格式。
- `mapper.py`: 使用 `hb_mapper` 将 `.onnx` 模型转换为 `.bin` 模型。
- `YOLO26_Inference.py`: 在 RDK 板端使用 `.bin` 模型进行推理。

## 快速上手

### 1. 导出 ONNX (PC/工作站)
确保已安装 `ultralytics`。
```bash
python3 export_yolo26_bpu.py
```

### 2. 模型转换 (OpenExplore Docker)
使用 `hb_mapper` 工具转换 ONNX 模型。
```bash
python3 mapper.py --onnx yolo26n_bpu.onnx --cal-images /路径/到/校准数据集
```

### 3. 板端推理 (RDK 板端)
将生成的 `.bin` 模型拷贝到 RDK 板端并运行：
```bash
python3 YOLO26_Inference.py --model-path yolo26n_bayese_640x640_nv12.bin --test-img bus.jpg
```

## 后处理说明
`YOLO26_Inference.py` 脚本包含了专门针对 BPU 导出模型优化的后处理实现，能够高效处理解码后的检测框和分类输出。
