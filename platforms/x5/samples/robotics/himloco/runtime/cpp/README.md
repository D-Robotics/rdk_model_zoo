[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco C++ Runtime

This sample loads the fused RDK X5 model through the BPU SDK, runs
source-indexed offline inputs, and writes float32 actions plus a JSON report.

## Requirements

- RDK X5 with DNN Runtime headers and libraries.
- CMake 3.10 or later and a C++17 compiler.
- gflags headers and library.

On an RDK OS image with apt access, install the build dependencies with:

```bash
sudo apt install cmake g++ libgflags-dev
```

The BSP provides `dnn/hb_dnn.h`, `dnn/hb_sys.h`, `libdnn.so`, and
`libhbrt_bayes_aarch64.so`.

## Directory Structure

```text
cpp/
├── inc/himloco.hpp   # Config, metadata, and reusable public interface
├── src/himloco.cc    # X5 model ownership and standard inference pipeline
├── src/main.cpp      # gflags command-line inference and JSON report
├── CMakeLists.txt    # C++17 build configuration
├── run.sh            # Model download, Release build, and one-command run
├── README.md         # English instructions
└── README_cn.md      # Chinese instructions
```

Build products and generated files are ignored under `build/` and `runs/`.

## Model Contract

| Tensor | Name | Type | Runtime shape |
| --- | --- | --- | --- |
| Input | `obs_history` | float32 | `[1,270,1,1]` |
| Output | `actions` | float32 | `[1,12,1,1]` |

The implementation queries tensor properties, allocates `alignedByteSize`,
cleans the input cache, invalidates the output cache, and extracts valid output
values according to the aligned shape.

## Build

```bash
cd runtime/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

`run.sh` performs these steps automatically.

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--model_path` | RDK X5 Bayes-e model | `../../model/bayes-e/himloco_go2_bayese_1x270.bin` |
| `--input_path` | One input file or input directory | `../../test_data/obs_history` |
| `--output_dir` | New action dump directory | `runs/<timestamp>/action_dumps` |
| `--report` | New JSON evidence report | `runs/<timestamp>/cpp-report.json` |
| `--warmup` | Unmeasured Runtime calls | `10` |
| `--priority` | DNN priority in [0,255], or -1 | `-1` |

## Quick Run

From the HIMLoco sample directory, download the model when missing, build, and
run all 21 bundled inputs:

```bash
bash runtime/cpp/run.sh
```

Specify parameters when using external data:

```bash
bash runtime/cpp/run.sh \
  --model_path /root/work/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin \
  --input_path /root/work/himloco/test_data/obs_history \
  --output_dir /root/work/himloco/evaluation/cpp_actions \
  --report /root/work/himloco/evaluation/cpp-report.json \
  --warmup 10
```

## Outputs

- One 48-byte float32 action file per source-indexed input.
- A JSON report containing board/DNN versions, tensor metadata, per-sample
  latency, p50/p95, and sequential throughput.

Compare actions with the JIT reference using
[`../../evaluator/compare_action_dumps.py`](../../evaluator/compare_action_dumps.py).

## Performance

On RDK X5 with DNN Runtime 1.24.5, 100 inputs and 10 warm-up calls produced:

| Min | Mean | P50 | P95 | Max | Sequential FPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.332 ms | 0.350 ms | 0.349 ms | 0.364 ms | 0.396 ms | 2853.09 |

Latency covers `hbDNNInfer` plus `hbDNNWaitTaskDone`; file I/O and cache
maintenance are excluded.

## Code Interface

`HimLoco` construction is lightweight. Call `init`, then use
`pre_process`, `infer`, `post_process`, or the combined `predict`.
Every public pipeline method returns 0 on success and -1 on failure; inspect
`last_error()` after a failure.

Follow the [source-reference documentation guide](../../../../../docs/source_reference/README.md)
to generate API documentation.

## Notes

- The sample performs offline tensor inference only; it does not command a robot.
- One `HimLoco` instance is not thread-safe.
- X5 has one BPU core, so the C++ sample exposes priority but not core binding.
