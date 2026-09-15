# Conversion

Run the exporter in the Ultralytics training environment and the mapper in the matching OpenExplorer toolchain container. Dependencies must be installed explicitly. These are separate environments from the board runtime.

```bash
python export_monkey_patch.py --pt yolo11n.pt --opset 11
python mapper.py --platform x5 --onnx yolo11n.onnx --cal-images ./cal_images
python export_monkey_patch.py --pt yolo11n.pt --opset 19
python mapper.py --platform s600 --onnx yolo11n.onnx --cal-images ./cal_images
```

`--optse` remains an alias for `--opset`; the selected value now reaches export. X5 and S calibration, quantization, input configuration and compiler invocations stay separate. An explicit conflicting `--march` fails. Use `--toolchain-help` in the configured container for the original compiler-specific arguments. Compilation has not been executed in this local merge.

## YOLO26

```bash
python export_monkey_patch.py --family yolo26 --task detect --platform x5 --pt yolo26n.pt --output yolo26n_detect.onnx
python mapper.py --family yolo26 --platform x5 --onnx yolo26n_detect.onnx --cal-images ./cal_images
python export_monkey_patch.py --family yolo26 --task cls --platform s600 --pt yolo26n-cls.pt --output yolo26n_cls.onnx
python mapper.py --family yolo26 --platform s600 --toolchain-help
```

All five tasks share the export entry. Defaults: X5 opset=11/simplify=1; S opset=19/simplify=0, with explicit overrides. Classifier imgsz defaults to 224, others to 640. Export errors fail nonzero; dependencies are never installed implicitly. Original calibration/compiler workflows remain in `yolo26/mapper_x5.py` and `mapper_s.py`. Real ONNX export and toolchain compilation have not been executed.
