# Conversion

Run the exporter in the Ultralytics training environment and the mapper in the matching OpenExplorer toolchain container. Dependencies must be installed explicitly. These are separate environments from the board runtime.

```bash
python export_monkey_patch.py --pt yolo11n.pt --opset 11
python mapper.py --platform x5 --onnx yolo11n.onnx --cal-images ./cal_images
python export_monkey_patch.py --pt yolo11n.pt --opset 19
python mapper.py --platform s600 --onnx yolo11n.onnx --cal-images ./cal_images
```

`--optse` remains an alias for `--opset`; the selected value now reaches export. X5 and S calibration, quantization, input configuration and compiler invocations stay separate. An explicit conflicting `--march` fails. Use `--toolchain-help` in the configured container for the original compiler-specific arguments. Compilation has not been executed in this local merge.
