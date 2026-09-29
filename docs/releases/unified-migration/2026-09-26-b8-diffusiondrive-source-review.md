# B8 DiffusionDrive source audit and migration decisions

Status: source audit only; canonical sample pending. Board=not-run;
independent Review=not-run; Closed=no. No hardware or remote computer was used.

## Source and delivered capability

[Capture script](evidence/2026-09-26-b8-diffusiondrive-audit/capture.py) verifies
39 files against S `380e1a2bf42041af54be6f34935e50197cfadff9` and inspects twelve
NPZ archives without pickle or SDK execution. No X5 or native C++ source exists.
The manifest publishes S100P/nash-m and S600/nash-p HBMs with SHA-256:

- S100P: `5829a1a318116e2eba5c3ba313841fbd064fe2d48fe3cd37eb1ec7d9a0ffa8fa`
- S600: `78605b5aaa573fbaa2bf4788c6c8922bcab39d90c195e264aa3a8d3a78321c53`

The Python source eagerly imports hbm_runtime, selects its first model, quantizes
four NAVSIM inputs, runs inference, dequantizes four outputs, applies agent
sigmoid/threshold and BEV argmax, and renders camera/BEV/LiDAR/trajectory/agents.
Five-case execution and offline float-reference comparison must also survive
migration. There is no actuation interface; none is required or added.

## Tensor contract

All packaged inputs/references are finite float32. Inputs: camera
[1,3,256,1024], lidar [1,1,256,256], status [1,8], noise [1,20,8,2]. Outputs:
trajectory [1,8,3], agent_states [1,30,5], agent_labels [1,30],
bev_semantic_map [1,7,128,256]. Preserve fixed diffusion noise; new random noise
changes the comparison problem.

NPZ logical floats do not establish physical HBM dtypes. Source discovers
S8/U8/S16/U16/S32/U32/F16/F32 from actual metadata. Inspect shared U16/U32
normalization before preserving those contracts. Validate actual names, shapes,
dtypes and quantization rather than assuming all-node INT16 implies int16 public
IO. Real HBM metadata remains unobserved locally.

Input quantization is per-tensor rint(x/scale + zero), clipped to dtype limits.
Output dequantization is per-tensor/per-axis. Agent sigmoid clips logits to
[-60,60], threshold defaults0.5 with >=; BEV argmax uses axis1. Preserve this
arithmetic and keep raw physical outputs distinct from decoded results.

## Verified defects and design decisions

The capture demonstrates three counterexamples:

1. Per-axis scales with one scalar zero point raise a source reshape error.
   Broadcast the scalar across channels, reusing the existing shared correction.
2. Negative input scale silently maps 1 to -1. Reject nonpositive/nonfinite scales,
   invalid zero-point cardinality/range and invalid axes before execution.
3. The evaluator broadcasts [1,2,2] and [2,2] label maps and can report perfect
   agreement. Require exact shapes, finite values and valid class domains before
   metrics. Define zero-vector cosine behavior explicitly.

Source auto selection falls back to S600 on unknown hosts and accepts RDK_SOC.
Replace fallback with explicit target/asset selection and a physical identity gate
before SDK loading; inspection stays host-safe. Reuse shared NamedArrayRunner's
physical_inputs mapping for four-input transport. Keep the task constructor plus
pre_process/forward/post_process/predict only. Move dtype/quantization helpers,
rendering, scheduling, data IO and evidence outside the task.

## Conversion, data and README preservation

Preserve both YAMLs bytewise. Source OE image:
`registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`.
Settings: graph-wide INT16/max, GridSample INT8 exception, O2/latency, core1,
jobs32, cache, no input/output padding. The clean ONNX requires ScatterND rewrites
and static depthwise pooling, but no exporter, checkpoint revision or original
>=100-sample calibration set is provided. Configs alone do not reproduce the chain.

Keep all six input/reference pairs and six historical result images byte-identical.
Add a Chinese peer to test_data README, retaining five-case scene descriptions,
rare-class macro-IoU caveat, fixed noise, x-forward/y-left coordinates,0.25m pixels,
seven semantic classes and palette. Never regenerate references to pass tests.

Preserve all source accuracy/performance tables and INT8 versus INT16-first
comparisons, explicitly historical. Accuracy uses case_000 while profiling uses
case_017. Single-thread latency and two-thread aggregate throughput must remain
distinct; BPU-only/0ms CPU claims are source records, not new evidence. Full NAVSIM
PDM Score requires absent scene logs, sensor blobs, maps and metric cache.

## Next work

Implement source-parity tests on bundled arrays, strict binding/quantization,
shared transport reuse, pure stages, separate rendering, CLI and five-case runs
with provenance, strict offline comparison, explicit model downloads, preserved
configs and honest conversion preparation. Complete bilingual root/model/runtime/
conversion/evaluator/test_data README files, then regressions, catalog, contract
checks and host evidence. B8 and H0–H9 remain open.
