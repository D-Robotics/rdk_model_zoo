# Ultralytics YOLO conversion

[简体中文](README_cn.md)

This directory turns a float Ultralytics checkpoint into the ONNX graph and
then into the target BPU artifact used by the shared Python sample. Export and
compilation run on a host; they are not board runtime operations. Export covers
YOLOv8/YOLO11 DFL detection and YOLO26 direct-LTRB detection; classification,
segmentation, pose and OBB export entries are also listed below. An exporter
entry documents the recipe — record each produced artifact's own validation.

<a id="source-model"></a>

## Source model and reproducibility

Input is a local Ultralytics PyTorch `.pt` checkpoint matching the selected task. `/models/*.pt` paths below are prepared by the operator; custom weights are not bundled. Record the checkpoint SHA-256, training/export package versions, classes, input geometry and training configuration. The [published model inventory](../model/README.md) lists compiled `.bin`/`.hbm` runtime artifacts separately from conversion checkpoints.

<a id="toolchain-targets"></a>
## Prepare the two host environments

Use an Ultralytics training/export environment for the first step. It must
already contain the `ultralytics` package, PyTorch, and the ONNX export
dependencies appropriate for the checkpoint. Pass a local checkpoint path and
verify it before invoking the script. The script calls `ultralytics.YOLO`
directly; a known bare model name may resolve to an online checkpoint. Pass an
existing absolute `.pt` path to select the local checkpoint used for export.

Use the matching D-Robotics OpenExplore environment for the second step.
For the training/export environment, Ubuntu 22.04 with Python 3.10 is
recommended (a CUDA-capable GPU for training; verify with
`torch.cuda.is_available()`). Source `.pt` weights should be trained with the
`ultralytics/ultralytics` repository, or use officially released Ultralytics
pretrained weights; no program changes are required during training, and the
model `forward` method must not be modified. Per-target compiler entry:

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

For X5, the mapper is also installable as a pip package (hb_mapper is
expected to be 1.24.3 or newer):

```bash
conda create -n rdk_env python=3.10 -y
conda activate rdk_env
pip install rdkx5-yolo-mapper
hb_mapper --version
# If PyPI is slow, use a mirror:
pip install rdkx5-yolo-mapper -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

If a download is interrupted and leaves an incomplete package, retry the
install.

For X5, the OE 1.2.8 CPU image can be
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

The CPU image covers model conversion; the GPU image from the same OE 1.2.8
release is optional for extended environments with GPU dependencies:

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_gpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_gpu_v1.2.8.tar.gz
```

Offline
images are also available from the D-Robotics developer community:
<https://forum.d-robotics.cc/t/topic/35229>. Verify a fresh Docker install
with `docker --version` and `docker run hello-world` (install:
[Docker documentation](https://docs.docker.com/engine/install/)).

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

Set up the Ultralytics training/export environment first: clone
[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics.git) and
follow the official [Quick Start](https://docs.ultralytics.com/quickstart/)
and [training](https://docs.ultralytics.com/modes/train/) documentation
(see [Docker install](https://docs.docker.com/engine/install/) for a
containerized setup). Officially released pretrained weights can be fetched
directly, e.g.
`wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`.
For host pip installs, the
[Aliyun PyPI mirror](https://mirrors.aliyun.com/pypi/simple/) is available.

For a generic Ultralytics detector, choose the target so the shared exporter
uses the target's opset default. X5 uses opset 11; the S targets use
opset 19. The `--opset` and `--optse` spellings are equivalent and an
explicit value wins.

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
`--platform`, `--opset`, `--simplify`, and optional `--require-local`. A failed
exporter returns an error.

Run all Python examples from the repository root. The generic exporter patches the head type found in the checkpoint; `--task` does not turn detection weights into classification/segmentation weights. YOLO26 dispatches through `--family yolo26 --task...` to dedicated scripts, with default image size 224 for cls and 640 for other tasks. Export recipes:

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

YOLO26 cls/seg/pose/obb exporters do not accept `--require-local`. Check that the absolute checkpoint paths exist before running; do not pass that flag to these four scripts.

Pass the generated ONNX to the mapper below, adding `--family yolo26` for YOLO26. Expected input is static batch-one float32 NCHW. Detection DFL and direct-LTRB outputs are not interchangeable; segmentation includes mask coefficients/prototypes, pose keypoints, OBB angles, and classification logits with runtime Softmax. Check each exporter's output description and runtime binding rather than inferring compatibility from tensor count alone.

<a id="dataflow"></a>
## DFL-family dataflow: graph outputs and runtime decode

The illustrations below explain the DFL-family deployment pipeline. They
describe the DFL detection protocol (YOLOv5u/v8/v9/v10/11/12/13) and its
segmentation/pose extensions. YOLO26 detection has no diagram here; its
protocol difference is scoped at the end of this section.

### Object detection (DFL)

![](./imgs/ultralytics_yolo_detect_dataflow.png)

In the standard processing flow, the scores, categories, and xyxy coordinates
of all 8400 bounding boxes (bbox) are fully computed so that loss can be
calculated together with the ground truth during training. During deployment,
only bbox results that meet the score threshold need to be preserved, so there
is no need to fully compute all 8400 bbox results.

The optimization mainly uses the monotonicity of the Sigmoid function to
filter candidates before computing them. The same "filter first, then
compute" idea is also used in the DFL and feature decoding stages, saving a
large amount of computation and reducing inference latency.

- **Classification branch: ReduceMax operation**

ReduceMax obtains the maximum value on a specified dimension. In the YOLO
detection head, it finds the maximum among the 80 class scores of each of the
8400 grid cells. The operation is performed on the C dimension and outputs
the maximum value, not the index of that maximum.

The Sigmoid function is monotonic, so the relative ordering of the 80 scores
does not change before and after Sigmoid:

$$Sigmoid(x)=\frac{1}{1+e^{-x}}$$

$$Sigmoid(x_1) > Sigmoid(x_2) \Leftrightarrow x_1 > x_2$$

The position of the maximum value output by the model is therefore the
position of the final maximum score, and applying Sigmoid to that output
value yields the maximum class score of the float model:
$\operatorname{Sigmoid}(\max \mathrm{logits}) = \max \operatorname{Sigmoid}(\mathrm{logits})$.
The argmax ordering agrees before and after Sigmoid; the raw output value
itself is a logit, not yet a probability.

- **Classification branch: Threshold(TopK) operation**

Threshold(TopK) filters the grid cells that meet the threshold requirement.
It operates on the 8400 grid cells along the H/W dimensions; implementations
may flatten H/W for convenience, which does not change the semantics. Assume
the raw score of one class on a grid cell is $x$, the value after Sigmoid is
$y$, and the threshold is $C$. The necessary and sufficient condition for
this score to qualify is:

$$y=Sigmoid(x)=\frac{1}{1+e^{-x}}>C$$

which can be transformed into:

$$x > -ln\left(\frac{1}{C}-1\right)$$

This operation obtains the indices of the qualifying grid cells and their
corresponding maximum values. After Sigmoid, each maximum value is the class
score of that grid cell.

- **Classification branch: GatherElements and ArgMax operations**

Using the indices produced by Threshold(TopK), GatherElements extracts the
qualifying grid cells, and ArgMax determines which of the classes carries the
maximum, producing the class id of each selected grid cell.

- **Bounding box branch: GatherElements operation**

Using the same grid-cell indices, GatherElements extracts the corresponding
bbox information and obtains bbox features with shape `1×64×k×1`.

- **Bounding box branch: DFL (SoftMax + Conv)**

Each grid cell uses 4 numbers to describe the bbox position. The DFL
structure provides 16 estimates for the offset of one side relative to the
cell (anchor) position. These 16 estimates are passed through SoftMax, and
the expected value is calculated by convolution. This is a core Anchor-Free
design: each grid cell predicts exactly one bounding box. For one side
offset, assume the 16 estimates are $l_p$, where $p=0,1,...,15$. The offset
is calculated as:

$$\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{S}}, S =\sum_{p=0}^{15}{e^{l_p}}$$

- **Bounding box branch: Decode (dist2bbox / ltrb2xyxy)**

This operation decodes the ltrb description of each bounding box into an
xyxy description. ltrb represents the distances of the left, top, right, and
bottom sides from the grid-cell center:

![](./imgs/ltrb2xyxy.jpg)

For an input size $Size=640$ and feature level $i$ ($i=1, 2, 3$) with
downsampling factor $Stride(i)$, YOLOv8-Detect uses $Stride(1)=8$,
$Stride(2)=16$, $Stride(3)=32$, corresponding to feature map sizes
$n_i = Size/Stride(i)$, i.e. $n_1 = 80$, $n_2 = 40$, $n_3 = 20$, and
$n_1^2+n_2^2+n_3^2=8400$ grid cells in total. For the cell at column $x$,
row $y$ of level $i$ ($x$ counts along the horizontal axis and $y$ along the
vertical axis; $x,y \in [0, n_i)\cap Z$, with $Z$ the integers), the
ltrb-to-xyxy conversion is:

$$x_1 = (x+0.5-l)\times{Stride(i)},\quad y_1 = (y+0.5-t)\times{Stride(i)}$$

$$x_2 = (x+0.5+r)\times{Stride(i)},\quad y_2 = (y+0.5+b)\times{Stride(i)}$$

The final detection results are the class (id), score, and position (xyxy).

**Where these stages run in this sample.** The exported graph stops at the
per-stride messages drawn at the top of the figure — NHWC classification
logits (`1×80×80×80` at stride 8, `1×40×40×80` at 16, `1×20×20×80` at 32
for an 80-class model; a custom class count changes the 80) and DFL box
logits (`...×64`) — and the runtime performs the ReduceMax /
threshold-filter / gather / ArgMax, DFL SoftMax-plus-expected-bin, and
dist2bbox stages, followed by class-wise NMS where the selected binding
requires one, in its Python post-processing (`decode_dfl` in
`runtime/python/decode.py`; protocol in
[`DETECTION_CONTRACT.md`](../DETECTION_CONTRACT.md)). The S-series YOLOv10
binding is the NMS-free exception: it reuses the same decode stages with
`nms='none'` fixed (see the [runtime
README](../runtime/python/README.md)). The runtime consumes
already-dequantized floating outputs: integer tensors or SCALE quantization
metadata fail at binding, and no manual output dequantization step exists.

### Instance segmentation (DFL families)

![](./imgs/ultralytics_yolo_seg_dataflow.png)

Instance segmentation extends the object detection flow. After bbox results
that meet the requirement are selected from the detection branch, two
GatherElements operations extract that grid cell's 32 mask coefficients,
which are linearly combined with the prototype branch output (a weighted
sum, drawn as MatMul; prototypes are `1×160×160×32`, i.e. stride 4) to
generate the instance masks. The ReduceMax, Threshold(TopK), GatherElements,
DFL, and Decode optimizations of the detection branch therefore still apply.
The runtime performs the same coefficient–prototype combination
in post-processing on floating heads (`segmentation_decode`).

### Pose estimation (DFL families)

![](./imgs/ultralytics_yolo_pose_dataflow.png)

> **Pose head channels.** The pose head exports one class channel
> (single-class person models) and `3 × 17 = 51` keypoint channels per grid
> cell for the 17 COCO keypoints.

Ultralytics YOLO Pose keypoints are based on the object detection result.
The COCO keypoint definitions:

```python
COCO_keypoint_indexes = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}
```

The object detection part of the Pose model is the same as the Detect model,
and the pose head adds one per-cell feature map. Under the published
17-keypoint COCO contract, the binding requires one class channel plus
`3 × 17 = 51` keypoint channels per cell: each keypoint has an x/y
coordinate relative to that feature level's downsampling factor, plus a
visibility score (`pose_decode.py` binds `cls` with 1 channel and `kpts`
with `3 × nkpt`; the published binding fixes `nkpt = 17`, and changing the
shape declaration alone does not produce a supported variant). After the
detection branch selects a cell, the DFL decode computes each keypoint's
model-input position as `(raw_xy × 2 + anchor − 0.5) × stride`, where
`anchor` is the cell's half-integer center (`decode_kpts` in
`runtime/python/rdk_yolo_utils/postprocess.py`); `inverse_points`, together
with `inverse_boxes`, then restores original-image geometry from the
model-input letterbox, and Sigmoid converts the keypoint visibility logits
into scores (`pose_decode.py`). For comparison, the YOLO26 direct-LTRB pose
branch uses `(raw_xy + anchor) × stride`, without the DFL ×2 form. The
runtime decodes these heads in post-processing (`pose_decode`).

### YOLO26 direct-LTRB difference

The two box diagrams above describe the **DFL protocol only**; they do not
describe YOLO26. YOLO26 detection exports three NHWC classification tensors
plus three **direct four-channel LTRB** tensors (`[1,Hs,Ws,4]`, not
`[1,Hs,Ws,64]`): the four channels are already the left/top/right/bottom
distances in cell units, so the DFL SoftMax + 16-bin expectation stage does
not exist in that protocol. The `ltrb2xyxy.jpg` illustration belongs to the
DFL protocol's decode. The YOLO26 decoder takes the maximum class in
raw-logit space, applies Sigmoid to the selected class only, converts the
four distances using the same cell-center grid geometry, and then applies
the shared class-wise NMS (`decode_ltrb`). DFL and direct-LTRB artifacts are
not interchangeable, as stated in the export section above and in
[`DETECTION_CONTRACT.md`](../DETECTION_CONTRACT.md).

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

Run the dispatcher from the repository root (the platform wrapper entry
supplies the same platform argument):

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

Quick artifact inspection without running the full runtime:

```bash
# X5 (in the OE environment)
hb_model_info yolo11n_bayese_640x640_nv12.bin
# S-series (on the board)
hrt_model_exec model_info --model_file yolo11n_detect_nashm_640x640_nv12.hbm
hrt_model_exec perf --model_file yolo11n_detect_nashm_640x640_nv12.hbm --thread_num 1
```

If copied files have unexpected ownership on
the host, check file owners or run `sudo chown -R`; use optimization level
`O0`, `O1` or `O2` — `O3` is not supported on Nash.

The generic exporter and YOLO26 detection exporter accept `--require-local` when the checkpoint must already exist; without it the
script preserves Ultralytics' ability to resolve a known bare model name.

The generic and YOLO26 mapper paths call the same
`conversion/workflow.py` functions. `mapper_x5.py` and `yolo26/mapper_x5.py`
select the X5 profile. `mapper_s.py` and `yolo26/mapper_s.py` select the Nash
profile and retain `--march`. The workflow still keeps the target protocol
differences in the generated YAML:

* X5 invokes `hb_mapper makertbin --config config.yaml --model-type onnx`,
  uses `bayes-e`, raw RGB/NCHW calibration, and adds the Softmax int8
  optimization (plus `set_all_nodes_int16` for `--quantized int16`).
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

The conversion code path is intentionally small:

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

The mapper dispatcher accepts the published `--family` values for detection,
segmentation, pose, classification and OBB, and selects each task's workflow.

YOLOv8/YOLO11 detection uses the three-level DFL output contract.
YOLO26 detection uses direct LTRB and is not interchangeable with DFL. Export
and compile the custom graph, then run the runtime binding check to inspect its
input/output shapes and dtypes before board inference. See
[`DETECTION_CONTRACT.md`](../DETECTION_CONTRACT.md) for the runtime detection
protocol.

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

<a id="known-gaps"></a>
## Additional preparation

- Each produced artifact requires its own binding check, matching-board smoke run and applicable dataset/reference comparisons; published-artifact board records do not transfer to a fresh conversion.
- No pinned calibration image set/dataset version or complete source-checkpoint hash set is bundled. The 20–50 image guidance is a script recommendation. Defaults are `--cal-sample true --cal-sample-num 20`; `--cal-sample false` uses the whole eligible image pool. Save selected filenames and digests for reproducibility.
- `--quantized` defaults to int8, with int16 available; `--jobs` defaults to 16 and `--save-cache` to false. Inspect target-specific optimization choices using `mapper.py --platform x5 --toolchain-help` or the appropriate S target. Dispatch selection does not establish compiler compatibility.
- The container images linked in this guide are the documented environment examples for the targets they cover; select an image matching your target's toolchain requirements and record the actual image identity and version output.
- Validate fresh artifacts with the binding check, a matching-board smoke run and applicable dataset/reference comparisons (see [Validation](#validation)).
