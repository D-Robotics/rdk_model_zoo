English | [简体中文](README_cn.md)

# YOLOv5 evaluator

<a id="dataset"></a>
## Dataset

The source supplies `test_data/bus.jpg` for X5 and `test_data/kite.jpg` for S, plus `coco_classes.names`; there is no labeled benchmark harness in this sample. The evaluator compares one complete two-implementation run on the same image, target, artifact, and thresholds. It is a consistency evidence tool, not an mAP evaluator.

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── native/  # Files for native
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── compare.py  # Python script
└── source_reference.py  # Python script
```

<a id="environment"></a>
## Environment

Run on the target board with Python, NumPy, OpenCV, and its `hbm_runtime`. Prepare the model and image before execution. The evaluator loads the reference runtime from its pinned Git revision.

<a id="command"></a>
## Evaluation command

From the repository root, after preparing the exact model and using a recognized board, choose a new empty evidence directory:

```bash
python3 samples/vision/yolov5/evaluator/compare.py \
  --target x5 --variant n-v7.0 \
  --asset-id x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --model-path samples/vision/yolov5/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --test-img samples/vision/yolov5/test_data/bus.jpg \
  --output-dir /tmp/yolov5-evidence-unique
```

The utility runs the pinned original implementation and this sample's implementation, stores complete native input/output/result arrays as `.npy`, records metadata, code/model/image hashes, thresholds and board identity in `comparison.json`, and returns `0` only when every declared comparison passes. The output directory must not already exist. Run it on a prepared board for the variants you need — the X5 commands cover all nine variants and the S100/S600 commands cover the `x-672` model with `kite.jpg`; a run validates the tree it executes.

### Native C++ implementation comparison (board)

Use `evaluator/native/` to compare the C++ implementations on the same board, model, image and thresholds. The reference uses X5 `main.cc` at `ac115717197920355fc390bb04299b20e6436864` or S `src/yolov5.cpp`/`src/main.cpp` at `380e1a2bf42041af54be6f34935e50197cfadff9`. It captures each implementation’s preprocessing, raw outputs and decoded detections.

On the target board, with the repo checked out and the model downloaded
(X5 default: `yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` with
`test_data/bus.jpg`; S default: the x-672 asset with `test_data/kite.jpg`;
identical thresholds on both sides):

```bash
# 0) The pinned snapshots must exist locally. A shallow clone (the usual board
#    checkout) does not carry them; fetch them first (instrument.py fails
#    closed with this same hint otherwise):
git fetch --depth=1 origin ac115717197920355fc390bb04299b20e6436864   # x5
git fetch --depth=1 origin 380e1a2bf42041af54be6f34935e50197cfadff9   # s100/s600

# 1) Generate the instrumented fixed-source build (fail-closed on SHA/anchor
#    mismatch; audit in instrumentation-audit.json — the S build closure
#    (yolov5.hpp + utils/c_utils) is pinned and copied into the work dir, so
#    no mutable working-tree file enters the build). X5 rebinding example:
python3 samples/vision/yolov5/evaluator/native/instrument.py \
  --target x5 --repo-root . --work-dir /tmp/yolov5-fixed-src \
  --model-path samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
  --image-path samples/vision/yolov5/test_data/bus.jpg
#    (S100/S600 use --target s100 here to select the shared S source group;
#    actual unified build and comparison must select the real board target.)

# 2) Build the instrumented fixed source, then run it THROUGH the external
#    runner, which records the REAL process evidence (true argv including
#    gflags-stripped arguments, separate stdout/stderr, the actual exit code,
#    start/end UTC, cwd, board identity, pre/post hashes of the binary,
#    model and image, and the instrumentation-audit verification). The C++
#    observer only records what cannot be seen from outside the process
#    (tensor payloads and stage metadata) — it no longer tees stdout/stderr
#    or pretends to know the process exit code. The capture directory must
#    be EMPTY; an in-progress marker survives any crash and the comparison
#    rejects it.
cmake -S /tmp/yolov5-fixed-src -B /tmp/yolov5-fixed-src/build && \
  cmake --build /tmp/yolov5-fixed-src/build -j2
python3 samples/vision/yolov5/evaluator/native/run_capture.py \
  --binary /tmp/yolov5-fixed-src/build/yolov5_fixed_capture \
  --capture-dir /tmp/yolov5-source-capture \
  --model <model file> --image <image file> \
  --audit /tmp/yolov5-fixed-src/instrumentation-audit.json \
  -- [--model_path <m> --test_img <i> --label_file <l>]   # S flags, kept verbatim

# 3) Run the unified binary THROUGH THE SAME RUNNER (--role unified: full
#    process evidence for the unified side too — real argv/exit code and its
#    own binary/model/image hashes; no instrumentation audit on this role).
#    Build with -DYOLOV5_TARGET=s100/s100p/s600 matching the board.
python3 samples/vision/yolov5/evaluator/native/run_capture.py --role unified \
  --binary <build>/yolov5_cpp --capture-dir /tmp/yolov5-unified-run \
  --model samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
  --image samples/vision/yolov5/test_data/bus.jpg \
  -- --target x5 --test-img samples/vision/yolov5/test_data/bus.jpg \
     --model-path samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
     --dump-dir /tmp/yolov5-unified-dump

# 4) Compare the two same-board runs (the --target must equal the unified
#    build_target exactly — an s100 comparison rejects s100p/s600 builds;
#    compare an s600 build with --target s600; BOTH sides need their runner
#    records):
python3 samples/vision/yolov5/evaluator/native/compare_native.py \
  --target x5 --repo-root . \
  --source-capture /tmp/yolov5-source-capture --unified-dump /tmp/yolov5-unified-dump \
  --source-binary /tmp/yolov5-fixed-src/build/yolov5_fixed_capture \
  --unified-binary <build>/yolov5_cpp \
  --unified-run-record /tmp/yolov5-unified-run/run-record.json \
  --output /tmp/yolov5-native-comparison
```

The comparison requires matching target identities, tensor counts and shapes, dtypes, quantization parameters, thresholds and original-image detections. Input/output payloads, binaries, models and images are checked against their SHA-256 records. Supported layouts are uniform-pitch strided and exact-size compact arrays. Thresholds and scale descriptors are compared as float32 values; malformed layouts or missing captures return an error. Keep both process-record directories, payloads and `instrumentation-audit.json` together when copying results.

The comparison restores logical arrays from each side's recorded physical
layout (strides/dtype), so padded layouts compare correctly while
uninitialized padding bytes are preserved in the evidence (`originals/`) but
never claimed byte-equal. Fixed criteria, never adjusted per run: inputs
exact; raw outputs `allclose(atol=1e-5, rtol=0)`; scale/zero-point
descriptors exact; boxes `atol=1e-4`, scores `atol=1e-5`, class ids exact,
after a declared order normalization (both sides sorted by class_id, score,
x1..y2). Detections are compared in MODEL input space AND, mandatorily, in final
original-image coordinates (`detections_original` from both sides — the
capture emits it and the current dump); a missing or
one-sided final-coordinate capture fails the whole comparison rather than
passing with a disclaimer. Any missing material, nonzero run, model/image
hash or threshold mismatch fails nonzero with the gathered evidence
preserved; a native failure can never pass as empty arrays. These native
comparison steps execute on the real boards (build guidance:
`runtime/cpp/README.md`).

<a id="metrics"></a>
## Metrics

Inputs are exact; raw output arrays use shape/dtype checks and `rtol=0, atol=1e-5`; result boxes use `atol=1e-4`, scores `1e-5`, and class IDs exact. X5 and S must be compared using their own source protocol; The performance listed below is the source-published record with its original conditions.

<a id="outputs"></a>
## Outputs

Each run directory contains `legacy_*` and `unified_*` `.npy` arrays plus `comparison.json`. Failed preload or mismatch runs retain an error/failed record and return nonzero; mismatches return a nonzero status. The arrays preserve all captured input/output tensors and decoded result fields.

<a id="reference-results"></a>
## Reference results

The complete source X5 reference table:

| Model | Size | Params | BPU throughput | Python post-process |
|---|---|---:|---:|---:|
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |


### Board performance

The following COCO detector measurements separate BPU execution from Python postprocessing. Thread counts are stated for each measurement. X3 values are historical reference data; the runtime support matrix above applies to current board usage.

### RDK X5 & RDK X5 Module
Object Detection (COCO)
| Model | size (pixels) | number of classes | number of parameters (M) | float-point precision <br/>(mAP:50-95) | quantization precision <br/>(mAP:50-95) | BPU latency /BPU throughput (threads) | post-processing time <br/>(Python) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5  | - | - | 14.3 ms / 70.0 FPS(1 thread) <br/> 18.7 ms / 106.8 FPS(2 threads) | 12 ms |
| YOLOv5m_v2.0 | 640×640 | 80 | 21.8 | - | - | 27.0 ms / 37.0 FPS(1 thread) <br/> 44.1 ms / 45.2 FPS(2 threads) | 12 ms |
| YOLOv5l_v2.0 | 640×640 | 80 | 47.8 | - | - | 50.8 ms / 19.7 FPS(1 thread) <br/> 91.5 ms / 21.8 FPS(2 threads) | 12 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 | - | - | 86.3 ms / 11.6 FPS(1 thread) <br/> 162.1 ms / 12.3 FPS(2 threads) | 12 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 | 28.0 | - | 8.5 ms / 117.4 FPS(1 thread) <br/> 8.9 ms / 223.0 FPS(2 threads) <br/> 10.7 ms / 277.2 FPS(3 threads) | 12 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 | 37.4 | - | 13.0 ms / 76.6 FPS(1 thread) <br/> 16.0 ms / 124.2 FPS(2 threads) | 12 ms |
| YOLOv5m_v7.0 | 640×640 | 80 | 21.2 | 45.4 | - | 25.7 ms / 38.8 FPS(1 thread) <br/> 41.2 ms / 48.4 FPS(2 threads) | 12 ms |
| YOLOv5l_v7.0 | 640×640 | 80 | 46.5 | 49.0 | - | 47.9 ms / 20.9 FPS(1 thread) <br/> 85.7 ms / 23.3 FPS(2 threads) | 12 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 | 50.7 | - | 81.1 ms / 12.3 FPS(1 thread) <br/> 151.9 ms / 13.1 FPS(2 threads) | 12 ms |

### RDK X3 & RDK X3 Module (historical benchmark)
Object Detection (COCO)
| Model | size (pixels) | number of classes | number of parameters (M) | float-point precision <br/>(mAP:50-95) | quantization precision <br/>(mAP:50-95) | BPU latency /BPU throughput (threads) | post-processing time <br/>(Python) |

|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | - | - | 55.7 ms / 17.9 FPS(1 thread) <br/> 61.1 ms / 32.7 FPS(2 threads) <br/> 78.1 ms / 38.2 FPS(3 threads)| 13 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | - | - | 512.4 ms / 2.0 FPS(1 thread) <br/> 519.7 ms / 3.8 FPS(2 threads) <br/> 762.1 ms / 3.9 FPS(3 threads) | 13 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 28.0 | - | 85.4 ms / 11.7 FPS(1 thread) <br/> 88.9 ms / 22.4 FPS(2 threads) <br/> 121.9 ms / 32.7 FPS(4 threads) <br/> 213.0 ms / 37.2 FPS(8 threads) | 13 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 37.4 | - | 175.4 ms / 5.7 FPS(1 thread) <br/> 182.3 ms / 11.0 FPS(2 threads) <br/> 217.9 ms / 18.2 FPS(4 threads) <br/> 378.0 ms / 20.9 FPS(8 threads) | 13 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 50.7 | - | 1021.5 ms / 1.0 FPS(1 thread) <br/> 1024.3 ms / 2.0 FPS(2 threads) <br/> 1238.0 ms / 3.1 FPS(4 threads)<br/> 2070.0 ms / 3.6 FPS(8 threads) | 13 ms |

Note:

1. BPU latency vs. BPU throughput.
- Single thread latency is the latency of a single frame, single thread, single BPU core,BPU reasoning about a task.
- Multi-threaded frame rate means that multiple threads can simultaneously jam tasks to the BPU, each BPU core can handle the tasks of multiple threads. In general, 4 threads can control the single frame latency to be small, and eat all Bpus to 100% at the same time, getting a good balance between throughput (FPS) and frame latency. X5 BPU as a whole is quite good, generally 2 threads can eat up the BPU, frame latency and throughput are very good.
- The table generally records data where the throughput no longer increases significantly with the number of threads.
-BPU latency and BPU throughput are tested at the board side using the following commands
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
2. The test board is in the best condition.
The state of -X5 is the best state: 8 × A55@1.8G for CPU, full core Performance scheduling, and 1 × Bayes-e@10TOPS for BPU.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost" # 1.8 Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
-X3 is in the best state: 4 × A53@1.8G for CPU, full core Performance scheduling, and 2 × Bernoulli2@5TOPS for BPU.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost" # 1.8 Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
Floating-point/fixed-point mAP: 50-95 accuracy calculated using pycocotools, from the COCO dataset, refer to the Microsoft paper, here used to evaluate the accuracy degradation of on-board deployments.
4. On post-processing: At present, the post-processing of Python reconstruction on X5 only takes about 12ms from a single core and a single thread in serial, that is to say, it only takes 2 CPU cores (200% CPU occupancy, and the maximum CPU occupancy is 800%), and 166 frames of image post-processing can be completed every minute, and post-processing will not constitute a bottleneck.


## Additional reference measurements

Each table retains its published model, board and measurement conditions. Measurements from different configurations are separate reference sets.

### RDK X5 & RDK X5 Module

| 模型 | 尺寸(像素) | 类别数 | 参数量(M) | BPU延迟/BPU吞吐量(线程) |  后处理时间 |
|-----|----------|-------|----------|------------------------|----------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5  | 13.0 ms / 76.6 FPS (1 thread  ) <br/> 16.0 ms / 124.8 FPS (2 threads) | 2.3 ms |
| YOLOv5m_v2.0 | 640×640 | 80 | 21.8 | 23.9 ms / 41.7 FPS (1 thread  ) <br/> 37.7 ms / 52.9 FPS (2 threads) | 2.3 ms |
| YOLOv5l_v2.0 | 640×640 | 80 | 47.8 | 44.0 ms / 22.7 FPS (1 thread  ) <br/> 78.2 ms / 25.5 FPS (2 threads) | 2.3 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 | 74.1 ms / 13.5 FPS (1 thread  ) <br/> 137.6 ms / 14.5 FPS (2 threads) | 2.3 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 | 8.1 ms / 122.7 FPS (1 thread  ) <br/> 8.6 ms / 232.3 FPS (2 threads) <br/> 9.7 ms / 307.9 FPS (3 threads) | 2.3 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 | 11.9 ms / 83.8 FPS (1 thread  ) <br/> 13.7 ms / 145.3 FPS (2 threads) | 2.3 ms |
| YOLOv5m_v7.0 | 640×640 | 80 | 21.2 | 22.7 ms / 44.0 FPS (1 thread  ) <br/> 35.3 ms / 56.6 FPS (2 threads) | 2.3 ms |
| YOLOv5l_v7.0 | 640×640 | 80 | 46.5 | 41.6 ms / 24.0 FPS (1 thread  ) <br/> 73.1 ms / 27.3 FPS (2 threads) | 2.3 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 | 69.7 ms / 14.4 FPS (1 thread  ) <br/> 129.0 ms / 15.5 FPS (2 threads) | 2.3 ms |

<a id="boundaries"></a>
## Boundaries

Prepare the model and input images before comparison. X5 uses OpenCV XYXY-to-NMSBoxes; S uses class-wise XYXY NMS. Compare each target with its matching reference implementation.

Keep both capture directories, `instrumentation-audit.json` and stdout/stderr logs together. The comparison checks these files and copies them into `originals/` in its output directory.


### X3 and X3 Module COCO measurement set (Chinese source)

[Source measurement table](https://github.com/D-Robotics/rdk_model_zoo/blob/cb86079ae5befcef9ca50fb46c8a6d8980106dec/samples/vision/yolov5/README_cn.md).

This source reports 3 ms postprocessing for all five rows; the English source table above reports 13 ms. Each full record has its own source attribution. Conditions: X3 / X3 Module, COCO detection, 4 × A53 at 1.8 GHz with all cores in performance mode, 2 × Bernoulli2 at 1.0 GHz. Single-thread latency measures one task on one BPU core; multithread FPS measures queued throughput. These X3 measurements do not indicate current runtime support.

| 模型 | 尺寸(像素) | 类别数 | 参数量(M) | 浮点精度<br/>(mAP:50-95) | 量化精度<br/>(mAP:50-95) | BPU延迟/BPU吞吐量(线程) |  后处理时间 |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | - | - | 55.7 ms / 17.9 FPS(1 thread) <br/> 61.1 ms / 32.7 FPS(2 threads) <br/> 78.1 ms / 38.2 FPS(3 threads)| 3 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | - | - | 512.4 ms / 2.0 FPS(1 thread) <br/> 519.7 ms / 3.8 FPS(2 threads) <br/> 762.1 ms / 3.9 FPS(3 threads) | 3 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 28.0 | - | 85.4 ms / 11.7 FPS(1 thread) <br/> 88.9 ms / 22.4 FPS(2 threads) <br/> 121.9 ms / 32.7 FPS(4 threads) <br/> 213.0 ms / 37.2 FPS(8 threads) | 3 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 37.4 | - | 175.4 ms / 5.7 FPS(1 thread) <br/> 182.3 ms / 11.0 FPS(2 threads) <br/> 217.9 ms / 18.2 FPS(4 threads) <br/> 378.0 ms / 20.9 FPS(8 threads) | 3 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 50.7 | - | 1021.5 ms / 1.0 FPS(1 thread) <br/> 1024.3 ms / 2.0 FPS(2 threads) <br/> 1238.0 ms / 3.1 FPS(4 threads)<br/> 2070.0 ms / 3.6 FPS(8 threads) | 3 ms |
