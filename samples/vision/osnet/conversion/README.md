English | [简体中文](README_cn.md)

# OSNet X5 conversion

<a id="source-model"></a>
## Source model

Use the pinned PyTorch [source](../third_party/README.md) and the [official x1.0 Market1501 checkpoint](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view). Download that checkpoint to /inputs/osnet_x1_0_market1501.pth. SHA-256 is 2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154. Source SHA-256 is c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae; the original upstream Git commit is unknown.

<a id="directory"></a>
## Directory

prepare.py exports the spatial head, prepares calibration and base configs, and derives the late-INT8 config. All generated weights/data/runs are written to the explicit workspace outside Git.

<a id="toolchain-targets"></a>
## Toolchain and targets

Run in an existing openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310 environment: hb_mapper 1.24.3, HBDK 3.49.15, PyTorch 1.13.0+cpu, ONNX 1.12.0 and ONNX Runtime 1.16.2, with NumPy/Pillow/PyYAML. Target: X5 bayes-e. /repo must map the complete repository, /inputs the prepared files, /work a new writable workspace. These scripts do not create a container. No nash-e/nash-p recipe or S-series binary is provided.

<a id="export"></a>
## Export

```bash
python3 /repo/samples/vision/osnet/conversion/prepare.py prepare \
  --source-file /repo/samples/vision/osnet/third_party/osnet.py \
  --weights /inputs/osnet_x1_0_market1501.pth \
  --dataset /inputs/Market-1501-v15.09.15 --workspace /work
```
Strictly load the 751-class checkpoint. Replace inference Linear/BN1d with equivalent 1×1 Conv/BN2d, export opset 11, and compare 16 images against the source float model. ONNX input is [1,3,256,128] float32 and output is [1,512,1,1].

<a id="calibration"></a>
## Calibration

Prepare Market-1501-v15.09.15 with bounding_box_train/query/bounding_box_test as described in the evaluator guide. The same prepare command selects 128 training images with seed 20261029; train/query identities must not overlap. It checks 12936/3368/15913 split sizes. Calibration tensors are planar RGB float32 [0,255]. Fused mean is 123.675,116.28,103.53; scale is 0.0171247538316637,0.0175070028011204,0.0174291938997821. Runtime uses different signed RGB-minus-128 bytes.

<a id="compile"></a>
## Compile

```bash
hb_mapper checker --model-type onnx \
  --model /work/weights/osnet_x1_0_spatial.onnx --march bayes-e
hb_mapper makertbin --config /work/runs/spatial-rgb-int16/config.yaml --model-type onnx
python3 /repo/samples/vision/osnet/conversion/prepare.py late8 --workspace /work
hb_mapper makertbin --config /work/runs/spatial-rgb-late8/config.yaml --model-type onnx
# Optional fast INT8 comparison
hb_mapper makertbin --config /work/runs/spatial-rgb/config.yaml --model-type onnx
```
Configs use latency/O3/jobs=8. The accuracy-first config requests set_all_nodes_int16, with compiler fallbacks retained. late8 selects conv4/conv5/projection nodes from the optimized float graph and assigns INT8 input/output. Inspect actual placement; the high-precision model is not all-INT16.

<a id="validation"></a>
## Validate artifacts

Inspect compiler logs and quant-info. On X5, use the native model-info command below, then collect full Market1501 features using evaluator/main.py after registering the newly verified artifact identity. A new build does not inherit original BIN hashes or measured scores. The library matching.OSNet accepts matching custom tensor contracts; the normal CLI selects only registered artifacts.

```bash
hrt_model_exec model_info --model_file=/work/runs/spatial-rgb-late8/compiled/osnet_x1_0.bin
```

<a id="artifacts"></a>
## Artifacts

/work/weights/osnet_x1_0_spatial.onnx is the source graph; /work/runs/<variant>/config.yaml is the compile config; compiled/osnet_x1_0.bin is the X5 deployment model. Keep export.json, calibration_manifest.json, full compiler report and subsequent board results with each source/weight pair.

<a id="known-gaps"></a>
## Additional preparation

Provide the exact checkpoint, complete separately licensed Market1501 dataset, and matching OE environment yourself. The recovery TAR contains deployed models/fixtures/results, not the original conversion workspace. Preparation refuses occupied calibration/output directories; use a new workspace. Smaller widths, S100 and S600 require separate adaptation and accuracy evaluation.

### Reuse an existing BIN with a new FP32 baseline

`export` uses the pinned source and checkpoint, verifies the spatial head, and exports FP32 ONNX without creating calibration data or rebuilding the BIN. It requires the complete query/gallery splits. Its receipt explicitly says `float_baseline_only`; this does not recover the original conversion archive. The `prepare` command continues to require the independent full training split.

```bash
python3 samples/vision/osnet/conversion/prepare.py export --weights /absolute/market1501.pth --dataset /absolute/Market1501 --workspace /absolute/new/export-run
```

S-series HBMs come from target-specific private candidate bundles. Verify SHA256SUMS and artifacts.json, then pass --model-path; the X5 recovery ZIP/TAR does not contain these HBMs. See [S-series Workbench workflow](../evaluator/S_SERIES.md) for commands, calibration and fixed-input requirements.
