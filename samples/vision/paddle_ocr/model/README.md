# Model preparation

The pilot does not copy compiled model files into this directory. Use the
native entrypoint's explicit `--prepare` operation with both qualified asset
references and a local `--model-dir`:

```bash
python samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --model-dir /tmp/rdk-models
```

Preparation is explicit and uses the existing manifest asset reader. Normal
inference has no download side effect. The four audited rows have no recorded
publisher SHA-256, so the command reports observed digests without claiming
publisher provenance. See the [pilot README](../README.md) and the [legacy X5
model notes](../../../../platforms/x5/samples/vision/paddleocr/model/README.md)
and [legacy S model notes](../../../../platforms/s/samples/vision/paddle_ocr/model/README.md).
