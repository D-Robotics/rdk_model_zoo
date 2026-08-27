# Conversion (YOLO26 Depth, RDK-S)

Quantize the five YOLO26 Depth variants into `.hbm` models with the RDK-S
OpenExplorer toolchain (`hb_compile`). Run inside the OE Docker image, from this
`conversion/` folder.

## Mixed release profile

The release uses **two compile profiles selected per variant**:

| Variants | Profile | ONNX boundary | Input contract | Calibration |
| --- | --- | --- | --- | --- |
| `n` / `s` / `m` | **NV12** | calibrated log-depth (in-graph `clip → scale/bias → exp → resize4x`) | letterboxed NV12, `data_scale=1/255` | `max`, percentile `0.9999` |
| `l` / `x` | **lite** | raw 192×192 depth logit | scale-filled float32 NCHW featuremap `/255` | `default` (KL) |

The split exists because the NV12 compile of `l`/`x` clips its depth output
(quantization max pinned at the calibration max), while the lite compile of
`n`/`s`/`m` does not reach the 0.999 raw-cosine bar. Do not mix calibration
assets between profiles: the letterboxed RGB tensors for NV12 and the
scale-filled featuremap tensors for lite are different input contracts and
cannot be used interchangeably.

## Prerequisites

- OE Docker image: `ai_toolchain_ubuntu_22_s100_s600_gpu` (provides `hb_compile`).
- Ultralytics YOLO26 checkpoint for each variant (`yolo26{n,s,m,l,x}-depth-log.pt`).
- SUNRGBD calibration images (use `extract_sunrgbd_subset.py` to prepare a subset).

## Steps

### 1. Export per-variant ONNX

```bash
# lite boundary (l / x): raw logit output
python3 export.py --weights /path/to/yolo26l-depth.pt --variant l --output-dir ./onnx
# -> ./onnx/yolo26l-depth_op11_lite.onnx

# NV12 boundary (n / s / m): calibrated log-depth output
python3 export.py --weights /path/to/yolo26n-depth.pt --variant n --output-dir ./onnx
# -> ./onnx/yolo26n-depth-log.onnx
```

`export.py` produces the boundary matching the `--boundary` selection (default
follows the mixed profile: lite for `l`/`x`, log-depth for `n`/`s`/`m`).

### 2. Prepare calibration data

```bash
python3 extract_sunrgbd_subset.py --src /path/to/sunrgbd --out ./sunrgbd_subset
python3 prepare_calibration.py --images ./sunrgbd_subset --output ./calibration \
  --manifest ./calibration_manifest.json --report ./calibration_report.md
# -> ./calibration/*.npy (float32 tensors matching each profile's contract)
```

### 3. Quantize with hb_compile

24 committed YAMLs live under `ptq_yamls/` (9 NV12 for n/s/m + 15 lite for
all five variants; the lite n/s/m YAMLs are retained for experiments).

```bash
python3 scripts/quantize.py --variant n --march nash-e   # NV12 profile selected automatically
python3 scripts/quantize.py --variant l                  # lite profile
python3 scripts/quantize.py                               # release set: 5 variants × 3 marches
```

### 4. Copy the .hbm into the model directory

```bash
cp bpu_model_output_yolo26n_nv12_nashe/yolo26n_depth_nashe_768x768_nv12.hbm ../model/nash-e/
cp bpu_model_output_yolo26l_lite_nashe/yolo26l_depth_lite_nashe_768x768.hbm ../model/nash-e/
```

The output filename matches `runtime/python/main.py`.

## Notes

- int8 quantization (CNN-friendly).
- The NV12 profile applies `/255` via `data_scale` inside the graph; the lite
  profile applies `/255` in the calibration generator and board runtime.
- Each config is committed (paths relative to this `conversion/` folder) so the
  compile is reproducible without runtime-generated YAML.

## Precision tuning (optional)

Keep the mixed profile as the release default. Precision changes are
controlled experiments, not interchangeable flags:

- The NV12 compile of `l`/`x` is **not shippable**: the depth output clips at
  the calibration max (bus.jpg shows ~17% saturated pixels, cosine 0.9938/0.9944
  across boards). Fixing it requires a calibration set with higher-logit
  coverage, not a YAML change — `max + 0.9999` was already in use.
- The lite compile of `n`/`s`/`m` stays below the 0.999 raw-cosine bar
  (0.9903/0.9854/0.9529 on S100 bus.jpg). `max + 0.9999` calibration improves
  `n` to 0.9975 but does not clear the bar.
- A nash-e/S100 test of `node_info` with
  `/model.23/head/head.3/Conv: OutputType: int16` produced the same raw cosine
  as the lite int8 baseline (0.985449 on `bus.jpg`).
- A nash-e/S100 test of
  `calibration_parameters.optimization: set_all_nodes_int16` increased the HBM
  size from about 13 MB to 25 MB, increased measured inference from about
  5.8 ms to 23.0 ms, and reduced raw cosine to 0.982804. It is therefore **not
  a recommended default** for this model.

If a future model or dataset fails its acceptance metric, first preserve the
same ONNX, calibration images, preprocessing, board, and test input; then
change one quantization option at a time and compare raw-logit cosine and the
postprocessed depth map. Record the selected profile and board evidence before
replacing a published HBM.
