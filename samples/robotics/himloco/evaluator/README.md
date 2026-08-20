[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco Model Evaluation

This directory compares the fused floating-point formats and validates X5
policy outputs against a held-out native rollout.

## Files

- `compare_jit_onnx.py`: TorchScript versus floating-point ONNX.
- `prepare_runtime_inputs.py`: native rollout to source-indexed X5 inputs.
- `compare_action_dumps.py`: X5 float32 actions versus JIT or recorded actions.
- `metrics.py`: shared loading, hashing, and numerical metrics.

Use the training/export environment with PyTorch, ONNX, and NumPy available:

```bash
python3 -c "import torch, onnx, numpy"
```

Generated inputs, action dumps, and reports must be written outside the
repository, for example under `/work/himloco/evaluation`.

## Data Contract

The held-out rollout must not overlap PTQ calibration data. Its preferred
native `.pt` structure is:

```python
{
    "obs_history": torch.Tensor,  # [N,270]
    "actions": torch.Tensor,      # optional [N,12]
}
```

Capture `obs_history` after deployment-equivalent clipping, scaling, joint
ordering, and history stacking. Recorded actions must be the unscaled policy
output.

## 1. Validate TorchScript and ONNX

```bash
python3 compare_jit_onnx.py \
  --jit /work/himloco/policy.pt \
  --onnx /work/himloco/export/himloco_go2_op11.onnx \
  --data /work/himloco/rollout_evaluation.pt \
  --report /work/himloco/evaluation/jit-vs-onnx.json
```

The default gates are minimum per-sample cosine similarity `>= 0.999` and
maximum action absolute error `<= 1e-4`. A failed gate still writes its report
and returns a non-zero exit code.

## 2. Prepare X5 Inputs

```bash
python3 prepare_runtime_inputs.py \
  --data /work/himloco/rollout_evaluation.pt \
  --output /work/himloco/evaluation/runtime_inputs \
  --num-samples 0
```

Each input is 270 contiguous float32 values. Its numeric stem retains the
source rollout row so the board output can be matched to the correct reference.

## 3. Validate X5 Action Dumps

Run the prepared inputs with either board Runtime, copy the generated actions
back to the host, and compare them with the fused JIT:

```bash
python3 compare_action_dumps.py \
  --jit /work/himloco/policy.pt \
  --data /work/himloco/rollout_evaluation.pt \
  --candidate-dir /work/himloco/evaluation/x5_actions \
  --report /work/himloco/evaluation/x5-vs-jit.json \
  --min-cosine 0.99
```

If `actions` was recorded by the exact exported policy, omit `--jit` and use
`--action-key actions`.

Action dumps are retained as a Runtime feature because this policy has no image
or human-readable output. The 48-byte, source-indexed files provide a portable
boundary between the X5-only Runtime and the host-side TorchScript evaluator,
allow independent hashing and Python/C++ cross-checks, and avoid installing
PyTorch on the board. They are generated validation evidence, not files to
commit.

Each report contains action MAE/RMSE/maximum error, global and minimum sample
cosine, and equivalent joint-target errors using the `0.25` rad action scale.

## RDK X5 Performance Data

The same MIX model was run with 100 inputs and 10 warm-up calls:

| Runtime | Timing scope | Min | Mean | P50 | P95 | Max | Sequential FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python | `HB_HBMRuntime.run` | 0.818 ms | 0.885 ms | 0.880 ms | 0.942 ms | 0.975 ms | 1129.37 |
| C++ | `hbDNNInfer` + wait | 0.332 ms | 0.350 ms | 0.349 ms | 0.364 ms | 0.396 ms | 2853.09 |

Environment: RDK X5, RDK OS 3.5.0-beta, DNN Runtime 1.24.5, HBRT 3.15.55.
The timing scopes differ and exclude file I/O. Use `hrt_model_exec perf` for a
standardized pure-model benchmark before publishing a formal performance claim.

## Acceptance Order

```text
JIT versus ONNX
  -> Mapper output cosine
  -> X5 actions versus JIT
  -> closed-loop simulation
  -> guarded robot validation
```

Offline numerical agreement is necessary but does not establish control safety.

## License

Evaluator code follows the repository license. Policies and rollout data remain
subject to their respective terms.
