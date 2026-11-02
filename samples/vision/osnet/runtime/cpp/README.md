English | [简体中文](README_cn.md)

# Native OSNet verification

<a id="overview"></a>
## Overview

Verify fixed input bytes and raw embeddings, then measure native forward, preprocessing and end-to-end latency. Input is already-decoded RGB, not JPEG.

<a id="directory"></a>
## Directory

main.cpp calls osnet_main in osnet.cpp; run.sh checks board/artifact identity and builds the executable. build/ holds the ignored executable and result.json.

<a id="supported-boards"></a>
## Supported boards

X5 uses the registered RGB BINs. S100/S100P/S600 use their own accepted F32 HBM variants and the UCP backend. Prepare the exact model and matching fixture references.

<a id="dependencies"></a>
## Dependencies

AArch64 compiler supporting C++17, dnn/hb_dnn.h and libdnn (recorded 1.24.5, HBRT 3.15.55.0). The implementation uses arm_neon.h. Python is used only by the launcher for identity checks. No OpenCV, JPEG decoder or CMake is required.

<a id="build"></a>
## Build

Run at repository root on X5; no installation or download is performed. run.sh invokes g++ -O3 -std=c++17 -Wall -Wextra and links -ldnn. The host/board guard runs before compilation.

```bash
bash samples/vision/osnet/runtime/cpp/run.sh
```

<a id="run"></a>
## Run

Default execution uses spatial-rgb-late8, prepared test_data/fixture files, 3000 iterations and build/result.json. Customize with environment variables:

```bash
VARIANT=spatial-rgb-int16 RUNS=3000 OUTPUT_JSON=/tmp/osnet-int16.json \
  bash samples/vision/osnet/runtime/cpp/run.sh
```

<a id="parameters"></a>
## Parameters

Executable positional arguments are MODEL, images.dat, reference.f32, result.json, optional runs (binary default 1000; launcher default 3000). Launcher overrides: VARIANT=spatial-rgb-late8; MODEL_PATH=model/<variant>.bin; FIXTURE_DIR=test_data/fixture; BUILD_DIR=runtime/cpp/build; OUTPUT_JSON=<build>/result.json; RUNS=3000; CXX=g++; PYTHON_BIN=python3. Paths without explicit overrides are resolved from the script directory.

<a id="interface-lifecycle"></a>
## Interface and lifecycle

In osnet.cpp, Resize.run converts decoded RGB to the signed-planar input buffer; OSNet loads one model, allocates cached tensors, and infer waits synchronously, copies float output and optionally normalizes it. Each task is released after waiting; the destructor frees tensors and the model. Instances and resize buffers are single-threaded. The CLI owns file IO and statistics. Numerical code is preserved from the measured source; there is no separate public header.

<a id="results-interpretation"></a>
## Results

result.json contains preprocess_exact, feature_max_abs_error, l2_norm_max_error, native_forward, pipeline and preprocess (mean/median/p95/p99/min/max in ms). It refuses timing if input/feature checks fail. Exit 0 indicates completion, not full retrieval accuracy. Original 2026-10-29 measurements used 3,368 query + 15,913 gallery images, L2-normalized squared Euclidean distance and no reranking. C++ used 50 warmups, 3,000 iterations, batch 1, one stream, BPU 996 MHz and CPU schedutil. E2E covers decoded 64×128 RGB crop → Pillow-equivalent NEON resize to 128×256 → synchronous runtime → L2. Decoding, detection and matching are excluded; other crop sizes cannot inherit these timings.

### Workbench 1/2-stream benchmark

The new benchmark measures each decoded RGB crop through resize/packing, synchronous Runtime (cache operations and output copy included), and L2 normalization. Each worker owns its model and buffers. Every round warms each worker before timed frames. Aggregate FPS uses the common wall time, including loop overhead. Image decoding, file I/O and retrieval are outside timing. CPU policy is unchanged; this implementation uses native NEON and no OpenCV.

```bash
BENCHMARK=1 STREAMS=1 WARMUP=20 RUNS=200 ROUNDS=3 OUTPUT_JSON=/absolute/new/e2e-1.json bash samples/vision/osnet/runtime/cpp/run.sh
BENCHMARK=1 STREAMS=2 WARMUP=20 RUNS=200 ROUNDS=3 OUTPUT_JSON=/absolute/new/e2e-2.json bash samples/vision/osnet/runtime/cpp/run.sh
```

The executable also accepts `osnet --benchmark MODEL images.dat reference.f32 result.json STREAMS WARMUP RUNS ROUNDS`. The JSON contains per-frame stages, common wall time, per-round walls, 32-fixture correctness checks for every worker and native worker counts. With 200 runs and 3 rounds, total timed frames are 600/1200 for 1/2 streams. Record environmental snapshots separately; do not relabel historical timing as this benchmark.

For S-series set TARGET=s100 (or s100p/s600), VARIANT=<target>-f32-int16 and MODEL_PATH. run.sh enables UCP and links libdnn/libhbucp; tested SDK is 3.13.6 / HBRT 4.7.5. FIXTURE_DIR must contain images.dat and <variant>.f32, independently captured by the Python SDK from that HBM. See [S-series workflow](../../evaluator/S_SERIES.md).
