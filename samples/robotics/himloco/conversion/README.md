English | [简体中文](README_cn.md)

# HIMLoco Model Conversion

<a id="source-model"></a>
## Source model

Use the fused Go2 estimator/actor TorchScript `policy.pt` from [himloco_lab](https://github.com/IsaacZH/himloco_lab). Supply your training checkpoint and rollout; independently trained weights are not the same published artifact. Use the steps below to export and compile the fused policy. See [model identity](../model/README.md) for the published BIN.

This directory exports a fused HIMLoco TorchScript policy to static ONNX and
compiles it into an RDK X5 Bayes-e `.bin` model.

## Directory Structure

```text
conversion/
├── export_onnx.py          # Fused TorchScript -> static opset-11 ONNX
├── prepare_calibration.py  # Native rollout .pt -> Mapper float32 inputs
├── mapper.py               # Generate YAML, run Mapper, and write evidence
├── README.md
└── README_cn.md
```

`mapper.py` generates the complete YAML from validated CLI arguments. No static
YAML template is maintained. The generated YAML is stored under each run's
`config/` directory as the effective build record.

## Model Protocol

| Field | Value |
| --- | --- |
| Input | `obs_history`, float32 `[1,270]` |
| Output | `actions`, float32 `[1,12]` |
| ONNX batch/opset | fixed batch 1 / opset 11 |
| Runtime input | DDR featuremap, NHWC, no preprocessing |
| PTQ | MIX, float32 calibration data |
| Compiler | latency mode, one core, O3 by default |

One 45-value observation is ordered as follows:

```text
velocity_commands(3)
base_ang_vel(3, scale=0.25)
projected_gravity(3)
joint_pos_rel(12)
joint_vel_rel(12, scale=0.05)
last_action(12)
```

The 270-value input contains the current observation followed by five previous
observations. Clipping, scaling, joint order, and history order must match the
training policy boundary exactly.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── export_onnx.py  # Python script
├── mapper.py  # Python script
└── prepare_calibration.py  # Python script
```

<a id="toolchain-targets"></a>
## Environments

ONNX export runs in the training/export environment that can load `policy.pt`.
The source-recorded versions were PyTorch 2.7.0, ONNX 1.22.0, and NumPy 1.26.0.

PTQ compilation runs on an x86 Linux host in the RDK X5 OpenExplorer
environment. The source-recorded build used OpenExplorer v1.2.8 / Mapper 1.24.3.
Verify the required modules and tools before starting:

```bash
python3 -c "import torch, onnx, numpy, yaml"
hb_mapper --version
hb_model_info --help
```

Toolchain references:

- <https://developer.d-robotics.cc/rdk_doc/Advanced_development/toolchain_development/overview>
- <https://toolchain.d-robotics.cc/>

Run every command below from repository root.

<a id="export"></a>
## 1. Export Fused ONNX

Run in the environment that produced or can load the fused JIT model:

```bash
python3 samples/robotics/himloco/conversion/export_onnx.py \
  --jit /work/himloco/policy.pt \
  --output /work/himloco/export/himloco_go2_op11.onnx
```

The exporter checks the `[1,270] -> [1,12]` contract, runs `onnx.checker`, and
compares deterministic TorchScript and ONNX ReferenceEvaluator outputs. It also
writes an adjacent `.export.json` receipt with hashes, versions, operators, and
numerical metrics. Existing outputs are never overwritten.

<a id="calibration"></a>
## 2. Prepare Calibration Data

Capture the exact policy-boundary tensor during representative simulation
rollouts. The native file should contain:

```python
{"obs_history": torch.Tensor}  # [N,270]
```

Generate 100 deterministic calibration samples:

```bash
python3 samples/robotics/himloco/conversion/prepare_calibration.py \
  --input /work/himloco/rollout_calibration.pt \
  --tensor-key obs_history \
  --num-samples 100 \
  --seed 20260820 \
  --output /work/himloco/calibration
```

Each generated `.bin` contains 270 contiguous float32 values (1080 bytes).
`calibration-manifest.json` records the source hash, selected source indexes,
statistics, and preprocessing contract. Do not rename the serialized `.pt`
file to `.bin`.

<a id="compile"></a>
## 3. Run Checker and MIX PTQ

Run in the X5 OpenExplorer environment:

```bash
python3 samples/robotics/himloco/conversion/mapper.py \
  --onnx /work/himloco/export/himloco_go2_op11.onnx \
  --calibration /work/himloco/calibration/obs_history \
  --calibration-type mix \
  --optimize-level O3 \
  --output /work/himloco/compile_mix_o3
```

The script validates the calibration manifest, generates the effective YAML,
and executes `hb_mapper checker`, `hb_mapper makertbin`, and `hb_model_info`.
The output path must not already exist.

```text
compile_mix_o3/
├── artifacts/
│   ├── himloco_go2_bayese_1x270.bin
│   └── himloco_go2_bayese_1x270_quantized_model.onnx
├── config/himloco_go2_bayese_1x270.yaml
├── reports/
│   ├── checker.log
│   ├── makertbin.log
│   ├── hb_model_info.log
│   └── compile-report.json
└── working/
```

<a id="validation"></a>
## Validation Results

The source MIX/O3 build recorded the following:

| Metric | Result |
| --- | ---: |
| Output cosine similarity | 0.999606 |
| Compiler-estimated latency | 63.0 us |
| Compiler-estimated throughput | 15870.5 FPS |
| Compiler-estimated DDR traffic | 261776 bytes |

These are Mapper metrics, not held-out task accuracy or full Runtime latency.
Confirm the `.bin` march with `hb_model_info`, then follow
[`../evaluator/README.md`](../evaluator/README.md) for independent numerical and
board-output validation.

<a id="artifacts"></a>
## Artifacts and parameters

The compile directory contains `artifacts/himloco_go2_bayese_1x270.bin` for X5 and, when produced, a quantized ONNX for analysis. A newly trained/compiled BIN does not inherit the published digest; the published asset-id cannot bypass runtime identity checks.

| Script | Required | Defaults |
| --- | --- | --- |
| `export_onnx.py` | `--jit`, `--output` | `--report` = output stem + `.export.json`; `--opset 11`; `--validation-samples 8`; `--seed 20260820` |
| `prepare_calibration.py` | `--input`, `--output` | `--tensor-key obs_history`; `--num-samples 100`; `--seed 20260820` |
| `mapper.py` | `--onnx`, `--calibration`, `--output` | `--jobs 8`; `--optimize-level O3`; `--calibration-type mix`; `--max-percentile 1.0` |

<a id="known-gaps"></a>
## Boundaries

Training weights and a full rollout are not bundled. Follow the export, calibration, compilation, and runtime steps below.

- Use representative rollout states; random Gaussian calibration is not a
  release-quality substitute.
- Calibration and held-out evaluation rollouts must not overlap.
- Store all generated models, calibration data, logs, and reports outside the
  repository.
- Do not publish a build when Mapper does not report output similarity or
  `hb_model_info` does not confirm `bayes-e`.

## License

Tools in this directory follow the repository license. Training code, policies,
and redistributed artifacts remain subject to their upstream terms.
