# Python 运行时

`main.py` 是受限 OCR 试点的原生入口，支持已审计的 X5 PP-OCRv3 与 S100
PP-OCRv6 检测/识别资产。它解析限定 manifest 引用、校验执行目标，通过
`model_runner.py` 惰性加载两个模型，并输出有序框和文字。

## 命令和参数

```bash
python samples/vision/paddle_ocr/runtime/python/main.py --help
python samples/vision/paddle_ocr/runtime/python/main.py --list-models --target auto
python samples/vision/paddle_ocr/runtime/python/main.py --dry-run --target x5
bash samples/vision/paddle_ocr/runtime/python/run.sh --dry-run --target s100
```

`--target` 支持 `auto/x5/s100/s100p/s600`，正式执行要求检测到精确目标板。
`--det-asset-id` 与 `--rec-asset-id` 是 `--list-models` 打印的限定引用；若
提供 `--det-model-path` 和 `--rec-model-path`，两者必须同时提供并且文件已
存在，推理不会隐式下载。`--test-img` 指定 BGR 图像，`--output-format
json` 输出 `target`、`image_shape`、资产引用、`boxes` 和 `texts`，还可用
`--json-output` 写入本地文件。`--priority` 默认 0，`--bpu-cores` 默认 0。

`--list-models`、`--dry-run` 不加载 SDK、OpenCV、pyclipper 或模型。`--prepare`
是唯一显式准备入口，会通过共享 manifest 读取器获取 URL；普通推理和
`predict` 不调用它。S100P/S600 没有已审计 OCR 资产，会被拒绝。`hbm_runtime`
和 `pyclipper` 只在实际阶段惰性导入。完整契约、旧版映射和验收状态见
[`上级试点说明`](../../README_cn.md)，旧版命令见 [X5 runtime](../../../../../platforms/x5/samples/vision/paddleocr/runtime/python/README.md)
和 [S100 runtime](../../../../../platforms/s/samples/vision/paddle_ocr/runtime/python/README.md)。
