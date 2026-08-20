[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco C++ Runtime

This Runtime uses the RDK X5 BPU SDK C API to load the fused `.bin`, validate
its tensor contract, and run source-indexed offline policy inputs. It does not
construct live robot observations or command actuators.

## Requirements

- RDK X5 with DNN Runtime headers and libraries.
- CMake 3.10 or later.
- A C++17 compiler.

The BSP normally provides `dnn/hb_dnn.h`, `dnn/hb_sys.h`, `libdnn.so`, and
`libhbrt_bayes_aarch64.so` under `/usr/include` and `/usr/lib`.

## Model Contract

| Tensor | Name | Type | Runtime shape |
| --- | --- | --- | --- |
| Input | `obs_history` | float32 | `[1,270,1,1]` |
| Output | `actions` | float32 | `[1,12,1,1]` |

The implementation queries all tensor properties, allocates
`alignedByteSize`, cleans the cached input after CPU writes, invalidates the
output after BPU writes, and extracts valid output values according to the
reported aligned shape.

## Prepare Inputs

On the host, generate source-indexed inputs from a held-out rollout:

```bash
python3 samples/robotics/himloco/evaluator/prepare_runtime_inputs.py \
  --data /work/himloco/rollout_evaluation.pt \
  --output /work/himloco/evaluation/runtime_inputs \
  --num-samples 0
```

Each `.bin` input contains exactly 270 float32 values (1080 bytes). Copy the
model and input directory to the board.

## Build and Run

```bash
cd /root/work/himloco/runtime/cpp
bash run.sh \
  --model-path /root/work/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin \
  --input-path /root/work/himloco/test_data/obs_history \
  --output-dir /root/work/himloco/evaluation/cpp_actions \
  --report /root/work/himloco/evaluation/cpp-report.json \
  --warmup 10
```

`run.sh` configures a Release build under `build/`, builds `himloco_cpp`, and
runs it. Add `--priority 0..255` only for a controlled scheduling experiment.
X5 has one BPU core, so no multi-core binding option is exposed.

## Outputs

- One 48-byte float32 action file per source-indexed input.
- A JSON report containing board/DNN versions, valid and aligned shapes,
  per-sample latency, p50/p95, and sequential throughput.

Outputs are temporary evaluator evidence and must remain outside the repository.
Compare them with the JIT reference using
[`../../evaluator/compare_action_dumps.py`](../../evaluator/compare_action_dumps.py).

## Performance

On RDK X5 with DNN Runtime 1.24.5, 100 inputs and 10 warm-up calls produced:

| Min | Mean | P50 | P95 | Max | Sequential FPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.332 ms | 0.350 ms | 0.349 ms | 0.364 ms | 0.396 ms | 2853.09 |

Latency covers `hbDNNInfer` plus `hbDNNWaitTaskDone`; file I/O and cache
maintenance are excluded. This scope differs from the Python Runtime timing.

## Code Interface

`inc/himloco.hpp` declares the reusable `HimLoco` interface.
`src/himloco.cpp` implements model ownership, tensor memory, cache
synchronization, inference, and aligned output extraction. `src/main.cpp`
implements the source-indexed CLI and evidence report.

Follow the [source-reference documentation guide](../../../../../docs/source_reference/README.md)
to generate and browse API documentation.
