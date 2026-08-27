[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco Model Guide

This sample provides fused-model export, PTQ conversion, numerical evaluation,
and Python/C++ offline inference for a Unitree Go2 HIMLoco policy on RDK X5.

## Algorithm Overview

[HIMLoco](https://github.com/OpenRobotLab/HIMLoco) is a learned legged-locomotion
controller based on a hybrid internal model. The estimator consumes six stacked
45-value observations, predicts velocity and a normalized latent vector, and
feeds them with the current observation to the actor.

The policy used for this sample was trained with
[himloco_lab](https://github.com/IsaacZH/himloco_lab), an Isaac Lab port of the
original Isaac Gym implementation. The two projects implement the same policy
concept, but independently trained checkpoints do not share weights. This
sample is tied to the interface and artifacts of the exported himloco_lab Go2
policy.

## Capabilities

- Exports one fused static-shape ONNX from `policy.pt` and checks it against
  TorchScript.
- Generates representative Mapper calibration inputs from native rollout `.pt`
  data.
- Builds an RDK X5 Bayes-e `.bin` with MIX PTQ and records compilation evidence.
- Evaluates JIT/ONNX and X5/JIT numerical consistency on source-indexed samples.
- Provides Python and C++ BPU Runtime implementations with the same I/O contract.

## Platform Compatibility

| Platform | Runtime model | Python | C++ |
| --- | --- | --- | --- |
| RDK X5 | `.bin` | Supported | Supported |

Board inference was validated with RDK OS 3.5.0-beta, DNN Runtime 1.24.5, and
HBRT 3.15.55. Python must use the BSP-provided `hbm_runtime` matching the
installed `libdnn`.

## Model Contract

| Field | Value |
| --- | --- |
| Robot | Unitree Go2 |
| Source | Fused TorchScript `policy.pt` |
| Input | `obs_history`, float32 `[1,270]` |
| Output | `actions`, float32 `[1,12]` |
| History | Current 45-value observation followed by five previous observations |
| Target | RDK X5, `march: bayes-e` |
| Runtime input | DDR featuremap, no additional preprocessing |
| Action application | `default_joint_position + 0.25 * actions` |

The fused graph contains the estimator, velocity/latent processing, and actor.
It replaces the earlier two-model `encoder.onnx` plus `policy.onnx` deployment
boundary with one opset-11 ONNX model.

## Directory Structure

```text
himloco/
├── conversion/           # JIT export, calibration preparation, and X5 PTQ
├── evaluator/            # Floating-point and X5 numerical evaluation
├── model/                # X5 model download script and documentation
├── runtime/
│   ├── cpp/              # X5 BPU SDK C++ offline inference
│   └── python/           # X5 Python offline inference
├── test_data/            # 21 source-indexed offline Runtime inputs
├── README.md
└── README_cn.md
```

Model weights, ONNX files, calibration data, compiler workspaces, rollouts,
action dumps, and reports must be stored outside the repository. The downloaded
`.bin` is ignored under `model/bayes-e/`.

## Quick Start

Both scripts download the default model when missing, use the bundled test
inputs, and write results to an ignored timestamped `runs/` directory.

### Python

```bash
bash runtime/python/run.sh
```

See [Python Runtime](runtime/python/README.md) for parameters and integration.

### C++

```bash
bash runtime/cpp/run.sh
```

The script configures a Release build before inference. See
[C++ Runtime](runtime/cpp/README.md) for build and parameter details.

The included `test_data/obs_history` contains source indices 0 through 20 from
a held-out native rollout. Additional Runtime inputs can be generated with
`evaluator/prepare_runtime_inputs.py`. See
[evaluator/README.md](evaluator/README.md) before interpreting board outputs as
accuracy evidence.

## Model Conversion

The converted RDK X5 model is already available through
`model/download_model.sh`; ordinary Runtime users can skip conversion. To
reproduce it, follow [conversion/README.md](conversion/README.md). The supported
path is:

```text
policy.pt -> fused ONNX -> representative calibration -> MIX PTQ -> Bayes-e BIN
```

`conversion/mapper.py` generates the complete effective Mapper YAML for every
run; no separately maintained PTQ template is required.

## Runtime Inference

- [Python Runtime](runtime/python/README.md)
- [C++ Runtime](runtime/cpp/README.md)

Both implementations validate the single-input/single-output model contract and
write 12-value float32 action files. These action dumps are temporary numerical
evidence consumed by the evaluator; they are not model assets and must not be
committed.

## Inference Result

A successful default run processes the 21 bundled inputs and creates:

```text
runs/<timestamp>/
├── action_dumps/000000.bin ... 000020.bin  # 12 float32 actions per input
└── python-report.json or cpp-report.json   # Model, Runtime, and latency evidence
```

Use `evaluator/compare_action_dumps.py` to compare these actions with the
TorchScript reference before treating them as accuracy evidence.

## Evaluation and Performance

MIX PTQ compilation reported output cosine similarity `0.999606`. The following
board measurements used the same model, 100 inputs, and 10 warm-up runs:

| Runtime | Timing scope | Mean latency | Sequential throughput |
| --- | --- | ---: | ---: |
| Python | synchronous `HB_HBMRuntime.run` | 0.885 ms | 1129.37 FPS |
| C++ | `hbDNNInfer` + `hbDNNWaitTaskDone` | 0.350 ms | 2853.09 FPS |

The timing scopes differ, so the table does not claim that C++ changes BPU graph
execution speed. The compiler estimate was 0.063 ms; Runtime submission,
scheduling, and synchronization dominate this small model. Held-out board
accuracy remains a separate gate; see [evaluator/README.md](evaluator/README.md).

## Source Reference

Follow the [source-reference documentation guide](../../../docs/source_reference/README.md)
to generate and browse API documentation.

## Notes

- Calibration rollout and held-out evaluation rollout must not overlap.
- Offline tensor agreement does not validate observation construction, control
  period, joint mapping, actuator limits, or closed-loop stability.
- Validate in simulation first. Early robot tests require physical support,
  an emergency stop, and explicit command/limit checks.

## License

Follows the Model Zoo top-level License.
