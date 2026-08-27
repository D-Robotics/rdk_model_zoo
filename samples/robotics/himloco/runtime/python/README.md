[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco Python Runtime

This sample loads the fused RDK X5 model, runs source-indexed offline policy
inputs on BPU, and writes float32 actions plus a JSON evidence report.

## Requirements

- RDK X5 with RDK OS 3.5.0 or later.
- BSP-provided `hbm_runtime` matching the installed `libdnn`.
- Python 3.10 or later and NumPy.

The required Runtime is normally provided by the BSP. Check it with:

```bash
cat /etc/version
python3 -c "from hbm_runtime import HB_HBMRuntime; print(HB_HBMRuntime.version)"
```

Do not install the unrelated PyPI package named `hbm_runtime`.

## Directory Structure

```text
python/
├── himloco.py       # HimLocoConfig and reusable Model Zoo inference pipeline
├── main.py          # Source-indexed command-line inference and report writer
├── run.sh           # Model download and one-command execution
├── README.md        # English instructions
└── README_cn.md     # Chinese instructions
```

Generated files are written under the ignored `runs/` directory.

## Model Contract

| Tensor | Name | Type | Logical shape |
| --- | --- | --- | --- |
| Input | `obs_history` | float32 | `[1,270]` |
| Output | `actions` | float32 | `[1,12]` |

Equivalent batch-one Runtime layouts are accepted when their element counts
match. Unexpected names, types, shapes, NaN/Inf, or non-`.bin` models are
rejected before inference.

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--model-path` | RDK X5 Bayes-e model | `../../model/bayes-e/himloco_go2_bayese_1x270.bin` |
| `--input-path` | One input file or input directory | `../../test_data/obs_history` |
| `--output-dir` | New action dump directory | `runs/<timestamp>/action_dumps` |
| `--report` | New JSON evidence report | `runs/<timestamp>/python-report.json` |
| `--warmup` | Unmeasured Runtime calls | `10` |
| `--priority` | Optional DNN priority in [0,255] | Runtime default |
| `--bpu-cores` | Optional BPU core indexes | Runtime default |

## Quick Run

From the HIMLoco sample directory, download the model when missing and run all
21 bundled inputs with defaults:

```bash
bash runtime/python/run.sh
```

Specify parameters when using external data:

```bash
bash runtime/python/run.sh \
  --model-path /root/work/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin \
  --input-path /root/work/himloco/test_data/obs_history \
  --output-dir /root/work/himloco/evaluation/python_actions \
  --report /root/work/himloco/evaluation/python-report.json \
  --warmup 10
```

## Outputs

- One 48-byte float32 action file per source-indexed input.
- A JSON report containing model/input/output hashes, Runtime metadata,
  scheduling state, per-sample latency, and latency summary.

The action files are evaluator inputs, not robot commands. Compare them with
[`../../evaluator/compare_action_dumps.py`](../../evaluator/compare_action_dumps.py).

## Performance

On RDK X5 with DNN Runtime 1.24.5, 100 inputs and 10 warm-up calls produced
0.885 ms mean synchronous `HB_HBMRuntime.run` latency (1129.37 sequential
FPS). File I/O, validation, and action writes are excluded.

## Code Interface

`himloco.py` exposes:

- `HimLocoConfig` for model path and scheduling defaults;
- `HimLoco.pre_process`, `forward`, `post_process`, and `predict`;
- `HimLoco.__call__` as the callable equivalent of `predict`.

Follow the [source-reference documentation guide](../../../../../docs/source_reference/README.md)
to generate API documentation.

## Notes

- The sample performs offline tensor inference only; it does not command a robot.
- X5 requires the BSP Runtime package that matches the board `libdnn`.
- Output directories and report paths must not already contain run artifacts.
