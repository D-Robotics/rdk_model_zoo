# Pinned classification conversion

These host tools export local, explicitly versioned timm checkpoints. Inference
continues to use each Sample's native download and runtime commands.

The tested export environment uses PyTorch 2.8.0, timm 1.0.20, ONNX 1.17.0,
ONNX Runtime 1.19.2, onnxsim 0.4.36 and safetensors. No checkpoint is downloaded
by the exporter. Obtain the selected model's checkpoint from its recorded
official URL/revision and verify its complete publisher SHA-256 first.

```bash
# Repository root, an isolated new conversion run.
python3 utils/tools/classification_conversion/prepare_calibration.py \
  --images /path/to/frozen/images --count 200 --output /path/to/run/calibration_rgb_f32
python3 utils/tools/classification_conversion/export.py \
  --model edgenext_xx_small.in1k --checkpoint /path/to/pinned/model.safetensors \
  --calibration-sample /path/to/run/calibration_rgb_f32/0000.rgb \
  --output /path/to/run/model.onnx
```

Calibration uses raw RGB/NCHW float32 bytes, Pillow bicubic shorter-edge 256
and rounded center crop 224. Normalization is omitted from these tensors;
the X5 recipe fuses the checkpoint's actual mean and standard deviation.
This explicit 224/256 deployment geometry may differ from an upstream model's
evaluation configuration, which the export receipt records separately. No
upstream accuracy is claimed for this deployment profile.

Export strictly loads the checkpoint, checks any structural reparameterization,
simplifies/checks ONNX and compares PyTorch with ONNX Runtime CPU on both a
deterministic random input and one real calibration image. Output must be finite
1,000-class logits. The receipt records hashes, actual normalization, model
parameters, package versions and the errors. This is a host graph check; board
smoke, accuracy and speed remain `not-run` until independently measured.

Use the Sample conversion instructions and the Workbench Guide's converted
publication flow for checker/compiler, OE extraction, immutable Archive upload
and Web registration. Each weight, configuration or exporter change needs a
fresh attempt directory. Preserve all historical Sample recipes and artifacts.
