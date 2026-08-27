English | [简体中文](README_cn.md)

# YOLO26 Depth (RDK-S)

YOLO26 Depth estimates monocular depth from a single RGB image. This sample
deploys the five YOLO26 Depth variants (`n`, `s`, `m`, `l`, `x`) for the
RDK-S series (S100 / S100P / S600) using a **mixed release profile** selected
per variant for the best verified accuracy:

- **`n` / `s` / `m` — NV12 profile.** The ONNX graph keeps the calibrated
  `clip → scale/bias → exp → resize4x` postprocess in-graph; the runtime feeds
  a letterboxed NV12 image. Board-measured raw cosine vs FP32: 0.9984–0.9996.
- **`l` / `x` — lite profile.** The ONNX boundary is the raw 192×192 depth
  logit; the runtime feeds a scale-filled float32 featuremap and applies
  `clip → scale/bias → exp` plus the final resize on the CPU. Board-measured
  raw cosine vs FP32: 0.9997, zero saturated pixels.

The split exists because the NV12 compile of `l`/`x` clips its depth output
(quantization max pinned at the calibration max), while the lite compile of
`n`/`s`/`m` does not reach the 0.999 cosine bar. The mixed set is the only
combination where **all five variants pass** on every board.

## Platform compatibility

| Board | SoC | march | `n` / `s` / `m` (NV12) | `l` / `x` (lite) |
|---|---|---|---|---|
| S100 | s100 | nash-e | `model/nash-e/…_nv12.hbm` | `model/nash-e/…_lite_….hbm` |
| S100P | s100p | nash-m | `model/nash-m/…_nv12.hbm` | `model/nash-m/…_lite_….hbm` |
| S600 | s600 | nash-p | `model/nash-p/…_nv12.hbm` | `model/nash-p/…_lite_….hbm` |

## Directory structure

```
yolo26_depth/
├── conversion/          # ONNX export + hb_compile quantization
│   ├── ptq_yamls/       # committed YAMLs: NV12 n/s/m + lite l/x per march
│   ├── scripts/         # quantize.py, export.py, prepare_calibration.py, ...
│   ├── export.py  extract_sunrgbd_subset.py  prepare_calibration.py
├── evaluator/           # SUNRGBD numeric evaluation
├── model/               # download_model.sh + per-march .hbm (downloaded)
├── runtime/python/      # hbm_runtime inference: main.py, yolo26_depth.py, run.sh
└── test_data/           # bus.jpg
```

## Quick start

On the board:

```bash
cd samples/vision/yolo26_depth/runtime/python
bash run.sh            # default variant `n` (NV12 profile)
bash run.sh l          # variant `l` (lite profile)
# -> writes output/depth.png, output/overlay.png, output/log_depth.npy, ...
```

The runtime picks the input contract automatically from the variant: NV12 for
`n`/`s`/`m`, featuremap for `l`/`x`. Outputs are identical across profiles.

## Conversion

See [`conversion/README.md`](./conversion/README.md). Both ONNX boundaries come
from the same upstream weights: export either the calibrated log-depth ONNX
(NV12 profile, `n`/`s`/`m`) or the raw-logit lite ONNX (`l`/`x`), prepare the
matching calibration tensors, then run `hb_compile --config <yaml>`. Copy the
`.hbm` into `model/<march>/`.

## Evaluation

See [`evaluator/README.md`](./evaluator/README.md) for the SUNRGBD numeric
evaluation scripts.

## Validation

Keep raw-logit and postprocessed-depth validation separate when evaluating a
regenerated HBM. The release bar is raw-domain cosine ≥ 0.999 vs FP32 with
zero saturated pixels on `bus.jpg`, measured on board per (variant, march).

## License

This sample follows the RDK Model Zoo license. Upstream YOLO26 weights retain
their original license.
