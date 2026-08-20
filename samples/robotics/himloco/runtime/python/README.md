[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco Python Runtime

This Runtime loads the fused RDK X5 `.bin`, validates its tensor contract, and
runs source-indexed offline policy inputs on BPU. It does not construct live
robot observations or command actuators.

## Requirements

- RDK X5 with RDK OS 3.5.0 or later.
- BSP-provided `hbm_runtime` matching the installed `libdnn`.
- Python 3 and NumPy.

```bash
cat /etc/version
python3 -c "from hbm_runtime import HB_HBMRuntime; print(HB_HBMRuntime.version)"
```

Do not install the unrelated PyPI package named `hbm_runtime`.

## Model Contract

| Tensor | Name | Type | Logical shape |
| --- | --- | --- | --- |
| Input | `obs_history` | float32 | `[1,270]` |
| Output | `actions` | float32 | `[1,12]` |

Equivalent batch-one Runtime layouts are accepted when their element counts
match. Unexpected names, types, shapes, NaN/Inf, or non-`.bin` models are
rejected before inference.

## Prepare Inputs

On the host, generate source-indexed inputs from a held-out rollout:

```bash
python3 samples/robotics/himloco/evaluator/prepare_runtime_inputs.py \
  --data /work/himloco/rollout_evaluation.pt \
  --output /work/himloco/evaluation/runtime_inputs \
  --num-samples 0
```

Copy the model and `runtime_inputs` directory to the board.

## Run

```bash
cd /root/work/himloco/runtime/python
bash run.sh \
  --model-path /root/work/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin \
  --input-path /root/work/himloco/test_data/obs_history \
  --output-dir /root/work/himloco/evaluation/python_actions \
  --report /root/work/himloco/evaluation/python-report.json \
  --warmup 10
```

Runtime scheduling is automatic by default. Use `--priority` or `--bpu-cores`
only for a controlled scheduling experiment. Run `python3 main.py --help` for
the complete command-line interface.

## Outputs

- One 48-byte float32 action file per source-indexed input.
- A JSON report containing model/input/output hashes, Runtime metadata,
  scheduling state, per-sample latency, and latency summary.

The action files are temporary evaluator inputs, not repository assets. They
must be copied back to the host and checked with
[`../../evaluator/compare_action_dumps.py`](../../evaluator/compare_action_dumps.py).

## Performance

On RDK X5 with DNN Runtime 1.24.5, 100 inputs and 10 warm-up calls produced
0.885 ms mean synchronous `HB_HBMRuntime.run` latency (1129.37 sequential FPS).
File I/O, validation, and dump writes are excluded.

## Code Interface

`himloco.py` provides the reusable `HimLoco` class. `main.py` implements input
discovery, manifest validation, batch execution, action serialization, and the
JSON evidence report.

Follow the [source-reference documentation guide](../../../../../docs/source_reference/README.md)
to generate and browse API documentation.
