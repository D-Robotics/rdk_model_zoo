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

X5 calibration uses raw RGB/NCHW float32 bytes, Pillow bicubic shorter-edge 256
and rounded center crop 224. The X5 recipe fuses the checkpoint's actual mean
and standard deviation. S SDK 3.7.0 reads calibration in the ONNX input domain:
normalize the same photographs before writing `.npy` tensors. The S runtime
still uses NV12 and its recipe's fused mean/scale. Do not use raw X5 tensors
for S PTQ or normalize the runtime crop twice.
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

## S100 / S100P / S600 calibration and staged compilation

```bash
# Repository root; reuse the exact export receipt of the selected checkpoint.
python3 utils/tools/classification_conversion/prepare_calibration.py \
  --images /path/to/frozen/images --count 200 --target s100 \
  --normalization-json /path/to/export/model.export.json \
  --output /path/to/run/calibration/normalized
```

The tensors can be shared by targets with this exact source graph and
normalization. Each march requires its own PTQ, BC, NodeInfo and final HBM.
The new profiles explicitly use INT16 activations and O1. Inspect each
finished HBM's actual float32 output; the internal precision setting does
not determine its public dtype. NodeInfo cosine is a conversion diagnostic,
not held-out classification accuracy.

For the frozen S SDK 3.7.0 (hb_compile 3.5.3, HMCT 2.6.5, HBDK 4.7.5),
`prepare_bc.py` calls the SDK's native stages through NodeInfo and stops before
backend compilation. Run in its GPU image with CUDA mapped, `HBTL_THREAD_NUM=2`,
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`,
`OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=0`, `NVIDIA_TF32_OVERRIDE=0`, and
`HORIZON_TC_UI_DEBUG=1`. CPU/CUDA ONNX output parity must pass on three real
inputs and seed-0 data at atol 1e-4 / rtol 1e-3 before PTQ.

```bash
# In the frozen GPU SDK; config has working_dir /work/ptq-output.
python3 /repo/utils/tools/classification_conversion/prepare_bc.py \
  --config /work/ptq.yaml --manifest /work/calibration/normalized.manifest.json \
  --output /work/prepared.json
# Matching cached CPU SDK, same target and numerical configuration. Only the
# working_dir changes to /work/compiled; retain the original full NodeInfo CSV.
hb_compile -c /work/compile.yaml -m /work/ptq-output/MODEL_quantized_model.bc
hb_model_info /work/compiled/MODEL.hbm
```

Freeze all hashes in `prepared.json`, the original PTQ/optimized ONNX and
quantization JSON, the BC, both configurations, both logs and the native CSV.
The resumed CLI does not repeat PTQ or NodeInfo. Its final HBM/OE must be
collected separately and tied to the same BC. The helper is specific to this
verified SDK API; other SDK versions require verification of their own stages.
