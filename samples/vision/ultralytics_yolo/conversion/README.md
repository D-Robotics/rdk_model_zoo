# Ultralytics YOLO conversion

[简体中文](README_cn.md)

This directory turns a float Ultralytics checkpoint into the ONNX graph and
then into the target BPU artifact used by the shared Python sample. Export and
compilation run on a host; they are not board runtime operations. The shortest
supported scope for this migration is representative YOLOv8/YOLO11 DFL
detection and YOLO26 direct-LTRB detection. Existing classification, segmentation, pose and OBB export entries are also listed below. Having an exporter is not evidence that every task has completed end-to-end conversion validation.

<a id="source-model"></a>
## Source model and reproducibility

Input is a local Ultralytics PyTorch `.pt` checkpoint matching the selected task. `/models/*.pt` paths below must be prepared by the user; these weights are not bundled. Record the checkpoint SHA-256, training/export package versions, classes and input geometry. This repository does not pin one Ultralytics/PyTorch/ONNX version combination or publisher digest for every checkpoint family. Preserve training configuration and class order for custom weights. A compiled file in the [published model inventory](../model/README.md) does not prove that a similarly named `.pt` is its exact source checkpoint.

<a id="toolchain-targets"></a>
## Prepare the two host environments

Use an Ultralytics training/export environment for the first step. It must
already contain the `ultralytics` package, PyTorch, and the ONNX export
dependencies appropriate for the checkpoint. Pass a local checkpoint path and
verify it before invoking the script. The script calls `ultralytics.YOLO`
directly; Ultralytics can download a known bare model name, so a bare name is
not evidence that a local checkpoint was used. This repository does not claim
or record such an implicit download. Run the exporter with an existing
absolute `.pt` path when reproducibility matters.

Use the matching D-Robotics OpenExplore environment for the second step:

| Target | Compiler check | Artifact | Calibration file | Input convention in the compiler config |
| --- | --- | --- | --- | --- |
| X5 | `hb_mapper --version` | `.bin` | raw float32 `.rgbchw` | `bayes-e`, runtime `nv12` |
| S100 | `hb_compile --help` | `.hbm` | float32 `.npy`, values divided by 255 | `nash-e`, runtime `nv12` |
| S100P | `hb_compile --help` | `.hbm` | float32 `.npy`, values divided by 255 | `nash-m`, runtime `nv12` |
| S600 | `hb_compile --help` | `.hbm` | float32 `.npy`, values divided by 255 | `nash-p`, runtime `nv12` |

The mapper checks that the compiler is sourced, the ONNX model has exactly one
static `tensor(float)` rank-4 input, and the calibration directory contains
JPG, JPEG, or PNG files. It does not silently install a missing dependency.
The compiler environment and the board runtime image are separate; do not
copy `hb_mapper` or `hb_compile` installation steps into a board setup.

For X5, the repository's previously documented OE 1.2.8 CPU image can be
loaded and started on an x86 Linux host as follows. Use the D-Robotics image
and tag supplied with your release if they differ from this example; the
source download is the [X5 toolchain package](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz).

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker images
docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

For S, select the image matching S100/S100P or S600 from the [RDK S OE
toolchain documentation](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)
or the [toolchain download page](https://toolchain.d-robotics.cc/), then load
and mount it in the same way:

```bash
docker load -i ai_toolchain_ubuntu_22_s100_<release>.tar
docker images
docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace \
  <loaded-s100-or-s600-image>:<tag> /bin/bash
```

The S release pages distinguish [S100/S100P](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/overview?v=4.0.5&p=RDK+S100)
from [S600](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/overview?v=5.1.0&p=RDK+S600).
The `-v /workspace` mount exposes the checkpoint, ONNX file, calibration
images, and output directory to the container. Run the mapper commands below
inside that container from `/workspace`.

<a id="export"></a>
## Export the ONNX graph

For a generic Ultralytics detector, choose the target so the shared exporter
uses the target's reviewed opset default. X5 uses opset 11; the S targets use
opset 19. The `--opset` spelling and the historical `--optse` spelling are
equivalent and an explicit value wins.

```bash
# YOLOv8/YOLO11-style DFL detector for X5.
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --platform x5 --pt /models/yolo11n.pt --require-local --opset 11

# The same graph family for S600.
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --platform s600 --pt /models/yolo11n.pt --require-local --opset 19
```

The generic exporter loads `ultralytics.YOLO`, applies the sample's BPU
monkey-patch to the model head, and calls `model.export(format='onnx',
simplify=False, opset=...)`. Ultralytics normally writes the ONNX result next
to the checkpoint. Confirm the actual path before starting the mapper; the
mapper does not guess a similar file.

YOLO26 detection has a separate graph exporter because its head emits three
NHWC classification tensors and three direct four-channel LTRB tensors. The
platform defaults are X5 `opset=11, simplify=1` and S `opset=19, simplify=0`.

```bash
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task detect --platform x5 \
  --pt /models/yolo26n.pt --require-local \
  --output /models/yolo26n_det_bpu.onnx

python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task detect --platform s600 \
  --pt /models/yolo26n.pt --require-local \
  --output /models/yolo26n_det_bpu.onnx
```

For direct use, the equivalent script is
`yolo26/export_yolo26_detect_bpu.py` with `--weights`, `--output`, `--imgsz`,
`--platform`, `--opset`, `--simplify`, and optional `--require-local`. A failed exporter returns an error;
it does not leave a claimed artifact path in the documentation.

Run all Python examples from the repository root. The generic exporter patches the head type found in the checkpoint; `--task` does not turn detection weights into classification/segmentation weights. YOLO26 dispatches through `--family yolo26 --task ...` to dedicated scripts, with default image size 224 for cls and 640 for other tasks. These are existing export recipes, not claims of export execution in this round:

```bash
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task cls --platform x5 \
  --pt /models/yolo26n-cls.pt --imgsz 224 --output /models/yolo26n_cls_bpu.onnx
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task seg --platform x5 \
  --pt /models/yolo26n-seg.pt --imgsz 640 --output /models/yolo26n_seg_bpu.onnx
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task pose --platform x5 \
  --pt /models/yolo26n-pose.pt --imgsz 640 --output /models/yolo26n_pose_bpu.onnx
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task obb --platform x5 \
  --pt /models/yolo26n-obb.pt --imgsz 640 --output /models/yolo26n_obb_bpu.onnx
```

YOLO26 cls/seg/pose/obb exporters do not currently accept `--require-local`. Check that the absolute checkpoint paths exist before running; do not pass that flag to these four scripts.

Pass the generated ONNX to the mapper below, adding `--family yolo26` for YOLO26. Expected input is static batch-one float32 NCHW. Detection DFL and direct-LTRB outputs are not interchangeable; segmentation includes mask coefficients/prototypes, pose keypoints, OBB angles, and classification logits with runtime Softmax. Check each exporter's output description and runtime binding rather than inferring compatibility from tensor count alone.

<a id="calibration"></a>
## Calibration data preparation

Put 20–50 representative input images in a calibration directory. The shared
workflow converts each image from BGR to RGB, resizes it to the static ONNX
width and height, transposes HWC to NCHW, and writes float32 data. X5 writes
the byte-for-byte RGB tensor to `*.rgbchw`; S writes a NumPy `*.npy` tensor
after dividing by 255. This is a compiler calibration representation. Runtime
NV12 packing still follows the selected board input binding.

<a id="compile"></a>
## Compile

Run the canonical dispatcher from the repository root (or use the historical
platform wrapper, which supplies the same platform argument):

```bash
# X5 -> hb_mapper makertbin, bayes-e, .bin
python samples/vision/ultralytics_yolo/conversion/mapper.py \
  --platform x5 \
  --onnx /models/yolo11n.onnx \
  --cal-images /datasets/calibration \
  --output-dir /models/compiled

# S100 -> hb_compile, nash-e, .hbm
python samples/vision/ultralytics_yolo/conversion/mapper.py \
  --platform s100 --march nash-e \
  --onnx /models/yolo11n.onnx \
  --cal-images /datasets/calibration \
  --output-dir /models/compiled

# S100P/S600 use nash-m/nash-p respectively.
python samples/vision/ultralytics_yolo/conversion/mapper.py \
  --platform s600 --march nash-p \
  --onnx /models/yolo11n.onnx \
  --cal-images /datasets/calibration \
  --output-dir /models/compiled
```

<a id="validation"></a>
## Post-conversion validation

After the mapper succeeds, copy the target artifact to the matching board and
run the shared runtime with that exact path. For example, the S600 output from
the command above is `/models/compiled/yolo11n_nashp_640x640_nv12.hbm`:

```bash
# On an S600 board, from the repository root:
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo11 --task detect \
  --model-path /models/compiled/yolo11n_nashp_640x640_nv12.hbm \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/yolo11n-s600.jpg
```

Use `--platform x5` and the `.bin` name for X5, or `--platform s100`/`s100p`
and the corresponding `.hbm` name for the other Nash targets. A successful
compile does not bypass runtime input/output binding; the board still checks
the artifact metadata and the DFL or direct-LTRB contract before inference.

The generic exporter and YOLO26 detection exporter accept `--require-local` when the checkpoint must already exist; without it the
script preserves Ultralytics' ability to resolve a known bare model name.

The generic and YOLO26 mapper paths now call the same
`conversion/workflow.py` functions. `mapper_x5.py` and `yolo26/mapper_x5.py`
select the X5 profile. `mapper_s.py` and `yolo26/mapper_s.py` select the Nash
profile and retain `--march`. The workflow still keeps the target protocol
differences in the generated YAML:

* X5 invokes `hb_mapper makertbin --config config.yaml --model-type onnx`,
  uses `bayes-e`, raw RGB/NCHW calibration, and adds the historical Softmax
  int8 optimization (plus `set_all_nodes_int16` for `--quantized int16`).
* S invokes `hb_compile --config config.yaml`, uses the requested Nash march,
  normalized NumPy calibration, and adds `input_no_padding` and
  `output_no_padding`. `--quantized int16` adds the S `quant_config` model
  and output type settings.

The generated config is written inside a unique child of `--ws` (the option
is a workspace parent). Successful runs remove that child unless
`--save-cache` is set. A failed compiler run keeps the child and its config,
calibration files, and logs for diagnosis. The workflow refuses to overwrite
an existing artifact unless `--overwrite` is explicit, and it never removes a
user-supplied workspace parent, input model, calibration pool, or output
directory.

<a id="artifacts"></a>
## Output names and paths

With an ONNX stem of `yolo11n` and an input of 640×640, the default output
directory is the ONNX directory and the names are:

```text
X5:    yolo11n_bayese_640x640_nv12.bin
S100:  yolo11n_nashe_640x640_nv12.hbm
S100P: yolo11n_nashm_640x640_nv12.hbm
S600:  yolo11n_nashp_640x640_nv12.hbm
```

`--output-dir` changes only the final artifact and compiler log location. A
successful X5 run can leave `hb_mapper_makertbin.log`; an S run can leave
`hb_compile.log`. With `--save-cache`, inspect `config.yaml`, the calibration
directory, and `bpu_model_output/` under the reported unique workspace child.

## Code flow and compatibility symbols

The maintained path is intentionally small:

```text
export_monkey_patch.py
  ├─ generic detector -> Ultralytics YOLO export with shared patch
  └─ YOLO26 detect    -> yolo26/export_yolo26_detect_bpu.py

mapper.py -> mapper_x5.py / mapper_s.py
          -> yolo26/mapper_x5.py / mapper_s.py when --family yolo26
          -> workflow.py
             inspect_onnx -> calibration_images/select_calibration_images
             -> prepare_calibration -> render_config -> compiler -> artifact/log move
```

The historical platform paths under `platforms/x5/...` and `platforms/s/...`
remain thin forwarding entries. Their old class names and command defaults
are kept for callers; they do not contain another calibration or compiler
implementation. The mapper dispatcher continues to accept the existing
`--family` values so older segmentation/pose/classification commands are not
redirected to the detector protocol.

YOLOv8/YOLO11 detection uses the reviewed three-level DFL output contract.
YOLO26 detection uses direct LTRB and is not interchangeable with DFL. Export
and compile success alone does not prove that a custom graph has the runtime
contract; runtime binding checks input/output shapes and dtypes before board
inference. See [`DETECTION_CONTRACT.md`](../DETECTION_CONTRACT.md) for the
finite runtime detection protocol and
[`CONVERSION_CONTRACT.md`](CONVERSION_CONTRACT.md) for the old-to-new
conversion symbol map and target adapter boundary.

## Troubleshooting

* **`hb_mapper` or `hb_compile` is unavailable:** source the matching
  OpenExplore environment and rerun its version/help command. The board image
  is not a compiler environment.
* **ONNX input is dynamic, not float32, or has multiple inputs:** export a
  static batch-one NCHW graph. The mapper intentionally stops before writing
  calibration data when the input contract is unknown.
* **No calibration images:** use JPG/JPEG/PNG files directly in
  `--cal-images`; files with other extensions are ignored. Use
  `--cal-sample false` to retain the full pool, or change
  `--cal-sample-num` for sampling.
* **`--platform` and `--march` disagree:** use `x5` without a Nash march, or
  pair S100/S100P/S600 with `nash-e`/`nash-m`/`nash-p`.
* **Existing artifact:** choose a new `--output-dir` or pass `--overwrite`
  after checking that replacing the artifact is intended.
* **Compiler failed:** rerun with `--save-cache`; inspect the unique workspace
  `config.yaml`, calibration files, and compiler log. A failed workspace is
  retained by default for this purpose.

This checkout has host tests for planning, configuration, entry points, and
runtime contracts. Real ONNX export and OpenExplore compilation are not
executed in this local merge; board validation records refer to the runtime
artifacts listed by the release evidence, not to an unrun local conversion.

<a id="known-gaps"></a>
## Known gaps

- No real training environment, checkpoint export or OpenExplore compilation was exercised in this round. Historical board tests of published artifacts do not validate a fresh conversion.
- No pinned calibration image set/dataset version or complete source-checkpoint hash set is bundled. The 20–50 image guidance is a script recommendation, not a reproduced experiment. Defaults are `--cal-sample true --cal-sample-num 20`; `--cal-sample false` uses the whole eligible image pool. Save selected filenames and digests for reproducibility.
- `--quantized` defaults to int8, with int16 available; `--jobs` defaults to 16 and `--save-cache` to false. Inspect target-specific optimization choices using `mapper.py --platform x5 --toolchain-help` or the appropriate S target. Dispatch selection does not establish compiler compatibility.
- Container links are retained source-branch environment examples, not a verified latest-version recommendation or coverage of every target. Record the actual image identity and version output.
- Fresh artifacts require binding checks, a matching-board smoke run and applicable dataset/reference comparisons before being called validated. New conversion board validation remains not-run.
