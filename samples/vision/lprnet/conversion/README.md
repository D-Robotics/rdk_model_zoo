English | [简体中文](README_cn.md)

# LPRNet conversion

## Rebuilt X5 default PTQ / X5 独立重转

The new X5 profile `sirius-68class-bgr-defaultptq-o2-20261104` uses the same pinned trained Sirius checkpoint, 68-class order, full `[1,68,18,1]` output and BGR `(pixel-127.5)/128` input as the S profiles described below. It is independent of the old `lpr.bin`. Select its exact `x5:lprnet:lprnet_sirius_68class_24x94_bgr_featuremap_defaultptq_o2_20261104_bayese.bin` asset ID. The old X5 default remains unchanged.

```bash
# Run the source export above into a fresh dev-server directory, then convert its 200 NPY tensors.
python3 -m samples.vision.lprnet.conversion.prepare_x5_calibration --calibration /path/to/export/calibration --output /path/to/export/calibration-bin
cp samples/vision/lprnet/conversion/recipes/sirius-68class-x5.yaml /path/to/export/compile-x5.yaml
# In the cached X5 v1.2.8 toolchain, cwd /path/to/export:
hb_mapper checker --model-type onnx --march bayes-e --model model.onnx --input-shape images 1x3x24x94
hb_mapper makertbin --model-type onnx --config compile-x5.yaml
hb_model_info output/lprnet_sirius_68class_24x94_bgr_featuremap_defaultptq_o2_20261104_bayese.bin
```

Observed hb_mapper1.24.3 / HBDK3.49.15 / HBRT3.15.55.0; CPU execution inside the exact cached image recorded by digest. SDK **default PTQ**, O2, one core, DDR FEATUREMAP F32 NCHW, no fused preprocessing. The raw calibration files are binary F32 with no NPY header and pass byte-exact round trips. Actual graph is mixed INT8/INT16 BPU plus float CPU, with39 quantized/CPU table rows and two original BPU subgraph reports retained. This is not a pure INT16 conversion. All new board measurements remain `not-run`; native simulation cosines and compiler estimates are diagnostics.


## Reproducible Sirius S conversion / 可复现的 S 转换

The 2026-11-04 S profiles use source commit `7c976664b3f3879efabeaff59c7a117e49d5f29e` from [sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch) (Apache-2.0), trained checkpoint `weights/Final_LPRNet_model.pth` SHA-256 `d1ca22ca7a251a90702a166c8ad8ce7f05acb39f5ef8055139a33d462716d1ad`. Keep all weights, ONNX and tensors on dev-server.

```bash
# Repository root, pinned source already present on dev-server; fresh output only.
# torch 2.6.0+cpu, numpy 1.26.4, onnx 1.17.0, onnxruntime 1.20.1, opencv 4.11
python3 -m samples.vision.lprnet.conversion.export --source /path/to/pinned/LPRNet_Pytorch --output /path/to/fresh/export
cp samples/vision/lprnet/conversion/recipes/sirius-68class-s100.yaml /path/to/fresh/export/compile.yaml
# Cached S SDK 3.7.0 container, cwd the fresh export directory:
hb_compile -c compile.yaml
hb_model_info output/lprnet_sirius_68class_24x94_bgr_featuremap_int16_cpuptq_o1_20261104_nashe.hbm
```

Use the target's own YAML: S100 nash-e, S100P nash-m, S600 nash-p. Actual tools: hb_compile 3.5.3 / HMCT 2.6.5 / HBDK 4.7.5. CPU-only PTQ, FEATUREMAP float32 NCHW, no fused preprocessing, INT16 max percentile 0.9999, O1, one core, two compiler jobs. Each march has an independent calibration/quantization run. Do not relabel an old BIN/HBM or share another march's BC.

The exporter verifies the trained checkpoint, source definition and 68-class order; exactly rewrites 4D-unbatched MaxPool3d to channel slicing plus spatial pooling, retaining bit-exact source outputs. It exports all `(1,68,18,1)` logits and checks three real inputs plus random against source PyTorch at atol=1e-4/rtol=1e-3. The cached CPU SDK's full-output parity gate also passed, with zero observed difference.

Calibration uses 200 distinct real plate images, seed42, selected from the pinned source's 1,000-image test directory. This is explicitly a PTQ partition; the other 800 are held out for future accuracy evaluation. BGR linear resize94×24, `(pixel-127.5)/128`, float32 NCHW `.npy`; no duplicated or random calibration. Dataset accuracy is not measured.

`converted_targets_20261104.json` freezes actual source/checkpoint/ONNX/effective-config/recipe hashes, CPU reference gates and compiler receipts. The original Workbench producer and effective configuration remain in the run evidence; this portable exporter is a subsequent repository entry. Compiler ABI and complete originalNodeInfo/OE reports are collected for every artifact. Board smoke, accuracy, Runtime1/2 and C++1/2 are all `not-run`.

The following sections preserve the historical X5 delivery; they do not describe the new S checkpoints.

<a id="source-model"></a>
## Historical X5 source model

The source delivery provides the X5 deployment artifact and protocol notes, but no in-repository ONNX export script, checkpoint, calibration set, or PTQ YAML. The source conversion README uses an external OE package and a user-supplied `your_lprnet_config.yaml`; that placeholder is not a reproducible file in this repository.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

The source names an RDK X5 OpenExplorer environment with `hb_mapper` and `hrt_model_exec`. That historical delivery supports X5 only; the S conversions above use independent source weights.

<a id="export"></a>
## Export

Prepare an ONNX/checkpoint export pipeline with `input float32 NCHW (1,3,24,94)` and `output float32 (1,68,18,1)`. These are the deployment contracts for `lpr.bin`; CTC decoding consumes the `(68,18)` logits after singleton removal.

<a id="calibration"></a>
## Calibration

No calibration dataset or source calibration producer is included. The checked-in `test_input.dat` is a runtime input fixture, not a calibration set. Prepare a separate set of representative calibration inputs.

<a id="compile"></a>
## Compile

After the user supplies the missing ONNX and YAML in this conversion directory, the source documents these OE commands; they are conditional templates, not a runnable recipe here:

```bash
# cwd: samples/vision/lprnet/conversion; input: external your_lprnet_config.yaml
hb_mapper checker --model-type onnx --config ./your_lprnet_config.yaml
hb_mapper makertbin --model-type onnx --config ./your_lprnet_config.yaml
# success requires the OE tools to report success and create an X5 .bin artifact
```

The exact output filename, calibration options, quantization settings, and source checkpoint remain unknown.

<a id="validation"></a>
## Post-conversion validation

Inspect the generated model with `hrt_model_exec model_info --model_file ./lpr.bin`, then compare its metadata to the runtime binding. After conversion, run the [evaluator](../evaluator/README.md) comparator on the board to check two-implementation logits parity; conversion reproducibility follows the prerequisites listed in this directory.

<a id="artifacts"></a>
## Artifacts

The historical X5 deployment artifact is `x5:lprnet:lpr.bin`; the runtime expects it at `../model/lpr.bin`. ONNX, checkpoint, calibration, and compiler logs are external and are not checked in.

<a id="known-gaps"></a>
## Additional preparation

- No source export script, checkpoint version, calibration producer, PTQ YAML, or reproducible OE package is provided.
- `your_lprnet_config.yaml` is a source placeholder, not a repository file.
- Publisher SHA-256 is unknown. Board parity of the published artifact is recorded in the evaluator README and does not extend to rebuilding.
