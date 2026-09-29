[English](README.md) | [简体中文](README_cn.md)

# YOLO26 Depth conversion

This directory combines the X5 Mapper and S-series `hb_compile` recipes while
keeping their input representations separate. It prepares ONNX, deterministic
calibration tensors, resolved compiler configurations, logs and artifact hashes.
Host preparation has tests; actual Torch export, OpenExplorer compilation and
board validation have **not** been run during this migration.

<a id="source-model"></a>
## Source model and graph boundary

Supply a trained `yolo26{n,s,m,l,x}-depth.pt` checkpoint compatible with the
Ultralytics `Depth` head. No checkpoint is bundled or downloaded by these tools.
The source branches pin `ultralytics==8.4.105`; X5 also records `torch==1.13.0`.
Their requirements are preserved in `requirements-x5-source.txt` and
`requirements-s-source.txt`. These historical pins are not a claim that a new
installation or arbitrary Python version has been verified.

| Target / variants | ONNX output | Runtime preparation | CPU decode |
|---|---|---|---|
| X5 n/s/m/l/x | calibrated log-depth `[1,192,192,1]` | 768-square linear letterbox, padding 114, NV12 | exp, resize to 768, remove padding, restore source size |
| S100/S100P/S600 n/s/m | same calibrated log-depth | same letterbox and packed NV12 | same decode |
| S100/S100P/S600 l/x | raw logit `[1,192,192,1]` | 768-square linear scale-fill, RGB float32 NCHW `/255` | clip `[-4,5]`, checkpoint scale/bias, exp, direct source-size resize |

`export.py` retains the source attention/area-attention patches and Depth-head
refinement. The log boundary ends after clip and scale/bias; **exp and final
resizing are on CPU**. Earlier S prose placed them inside the graph, which
conflicts with its actual exporter and runtime.

Export reports record checkpoint calibration. The published S l/x runtime uses
`(cal_a, cal_b)=(1,-0.2498779296875)` / `(1,-0.316650390625)`; retrained weights
with different coefficients need a corresponding explicit runtime binding.
These maps are relative depth, not independently calibrated distances in metres.

<a id="toolchain-targets"></a>
## Toolchains and target recipes

Run conversion on an x86 Linux host in the corresponding OE environment.
The source X5 record is OpenExplorer 1.2.8 / Mapper 1.24.3, O3 latency and an
int16 tail-convolution output. The S record names
`ai_toolchain_ubuntu_22_s100_s600_gpu` but does not pin an image digest or compiler
version. Record the actual tool versions in any new conversion evidence.

The source X5 image URL is
`https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz`.
After obtaining and loading that image, the original container setup was:

```bash
# From repository root; replace /path/to/work with an external working directory.
docker run -it --rm --network host --shm-size=15g \
  -v "$PWD":/workspace -v /path/to/work:/work --workdir /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

Image availability, installation and compatibility are not reverified here.
Use an isolated export environment for the source requirements; `requirements.txt`
in this directory contains only host calibration/configuration dependencies.
Nothing installs dependencies implicitly. Python `--help` on the exporter does
not import Torch or Ultralytics.

All 29 source YAMLs are retained byte-for-byte under `ptq_yamls/{x5,s}`:

| Target | march | Published recipes | Retained experiments |
|---|---|---|---|
| X5 | bayes-e | 5 NV12, n/s/m/l/x | none |
| S100 | nash-e | 3 NV12 n/s/m + 2 lite l/x | 3 lite n/s/m |
| S100P | nash-m | 3 NV12 n/s/m + 2 lite l/x | 3 lite n/s/m |
| S600 | nash-p | 3 NV12 n/s/m + 2 lite l/x | 3 lite n/s/m |

`compile.py` reads the correct template and writes an external `config.yaml`
with absolute ONNX, calibration and working paths. The template's historical
relative names are not assumed to match generated files.

<a id="export"></a>
## Export

All following Python commands run from this `conversion/` directory. Use a new
output directory for each export; existing paths are refused.

```bash
python export.py --target x5 --variant n \
  --weights /work/weights/yolo26n-depth.pt --output-dir /work/depth/export_x5_n
python export.py --target s100 --variant n \
  --weights /work/weights/yolo26n-depth.pt --output-dir /work/depth/export_s_n
python export.py --target s100 --variant l \
  --weights /work/weights/yolo26l-depth.pt --output-dir /work/depth/export_s_l
```

With default opset 11 these produce `yolo26n-depth_op11_log.onnx` or
`yolo26l-depth_op11_lite.onnx`, the copied checkpoint and `export-report.json`.
Geometry is fixed at 768; `--opset` is configurable but only the source default
has historical provenance. `--boundary` can explicitly select an alternate S
export for experiments. X5 lite is refused because it has no source recipe.

<a id="calibration"></a>
## Calibration

The common archive extractor is preserved from both branches. Supply a local
SUNRGBD ZIP and respect the dataset's own terms; no dataset is included.

```bash
python extract_sunrgbd_subset.py --archive /work/datasets/SUNRGBD.zip \
  --split train --count 100 --seed 20260725 --output /work/depth/train100
python prepare_calibration.py --target x5 --variant n \
  --images /work/depth/train100/images --output /work/depth/cal_x5 --count 100
python prepare_calibration.py --target s100 --variant n \
  --images /work/depth/train100/images --output /work/depth/cal_s_nv12 --count 100
python prepare_calibration.py --target s100 --variant l \
  --images /work/depth/train100/images --output /work/depth/cal_s_lite --count 100
```

X5 emits RGB CHW uint8 `.bin` files; Mapper applies `/255`. S emits RGB float32
NCHW `.npy` files already divided by 255, preserving the source preparation.
S NV12 and lite differ in letterbox versus scale-fill geometry. Direct RGB
calibration is not byte-identical to RGB reconstructed from subsampled NV12.

Each directory has sibling `.json` and `.md` records (for example
`cal_s_nv12.json`). `--manifest` / `--report` may override those paths but must
remain distinct and outside the tensor directory. Records bind source and tensor
SHA-256, shape, dtype and geometry. Default seed is 20260725, count 100. Invalid
images, zero counts and existing destinations fail; use a new directory after a
failed partial preparation. Never mix the two S calibration directories.

<a id="compile"></a>
## Compile

First check preparation without invoking OE. Use a different new output path
for the real compilation, because work directories are not overwritten:

```bash
python compile.py --target s100 --variant n \
  --onnx /work/depth/export_s_n/yolo26n-depth_op11_log.onnx \
  --calibration-manifest /work/depth/cal_s_nv12.json \
  --output /work/depth/preflight_s_n --prepare-only
python compile.py --target s100 --variant n \
  --onnx /work/depth/export_s_n/yolo26n-depth_op11_log.onnx \
  --calibration-manifest /work/depth/cal_s_nv12.json \
  --output /work/depth/compile_s_n
python compile.py --target x5 --variant n \
  --onnx /work/depth/export_x5_n/yolo26n-depth_op11_log.onnx \
  --calibration-manifest /work/depth/cal_x5.json \
  --output /work/depth/compile_x5_n
```

For S l/x select the matching variant, lite ONNX and lite manifest. Use
`--target s100p` / `s600` for nash-m / nash-p; the S calibration representation
is shared across those marches. `--experimental-lite` selects retained S n/s/m
lite recipes; it does not turn them into published runtime profiles.

The wrapper checks all tensor digests, shape/dtype/range and directory contents
before writing a config. X5 runs checker, makertbin and model-info; S runs
`hb_compile`. Both require the expected compiled file to exist even when the
compiler returned zero. X5 also preserves quantized ONNX and compiler cosine,
latency/FPS and DDR estimates. `mapper.py` retains the original direct X5 flow;
it takes `--onnx`, `--variant`, `--calibration`, `--output`, `--size 768`,
`--jobs` and `--optimize-level`, without the newer manifest validation.

<a id="validation"></a>
## Validation and interpretation

`--prepare-only` validates local files, not the ONNX graph, OE compatibility,
model semantics or numerical accuracy. A real compiler run still needs inspection
of the input contract, output layout/type and raw/calibrated boundary, followed
by floating-point comparison and board tests. Retain the input/checkpoint hashes,
calibration manifest, compiler version and logs for that comparison.

```bash
hb_model_info /work/depth/compile_x5_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin
hrt_model_exec model_info --model_file /work/depth/compile_x5_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin
```

The second command requires the matching target environment. Do not equate
compiler estimates, one image's cosine, dataset accuracy and measured latency.
Offline depth evaluation belongs in `../evaluator/`. Recompiled artifacts do not
inherit a published artifact's SHA-256 or its historical performance evidence.

<a id="artifacts"></a>
## Output artifacts

- `config.yaml`: resolved exact source recipe and external input paths.
- `preparation.json`: ONNX/calibration hashes, identity and initially not-run status.
- `reports/`: real command stdout/stderr, kept even when a compiler fails.
- `working/`: compiler intermediates.
- `artifacts/`: final BIN/HBM; X5 also copies quantized ONNX.
- `compile-report.json`: written only after required artifacts/checks complete.

Failed compilation has no successful final report. Partial work remains available
for diagnosis; start another attempt in a new output directory.

<a id="known-gaps"></a>
## Known gaps and source evidence

The migration has not executed Torch export, either OE toolchain, a downloaded
model or SUNRGBD evaluation. Checkpoint hashes, S image/compiler versions and
publisher hashes for S HBMs remain absent from the source record. The exporter
requires the expected Depth module and fails if it cannot patch exactly one head.

Historical S experiments reported clipping for NV12 l/x, weaker cosine for lite
n/s/m, and no benefit from the tested int16 changes. Those observations explain
the mixed release but are not new validation. The source's “all variants pass
0.999” claim conflicts with its 0.9984 table entry; see the
[source audit](../../../../docs/releases/unified-migration/2026-09-26-b8-yolo26-depth-source-review.md).
Do not conceal that conflict by dropping the table or lowering its threshold.

The original optional tuning record gives these concrete comparison points:
NV12 l/x had about 17% saturated pixels and cosine 0.9938/0.9944; lite n/s/m
on S100 measured 0.9903/0.9854/0.9529. Max/0.9999 calibration raised lite n to
0.9975, still below the stated bar. Tail-convolution int16 gave 0.985449 on the
reported S100 test; all-nodes-int16 changed about 13 MB to 25 MB and 5.8 ms to
23.0 ms while reducing cosine to 0.982804. These are source observations with
incomplete reproduction evidence. For a new experiment hold ONNX, calibration,
image, board and preprocessing fixed, change one option, and record raw-domain
and restored-depth comparisons separately.

Repository code follows the root license; Ultralytics weights and SUNRGBD retain
their respective upstream terms. No export or dataset license is granted by this
sample's presence.
