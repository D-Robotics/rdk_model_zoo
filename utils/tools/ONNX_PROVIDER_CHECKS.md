# ONNX provider checks before S PTQ

Use the already provisioned source-export environment and the matching cached
S SDK. These host checks do not load a board model or measure board performance.
They preserve every source ONNX output at `atol=1e-4`, `rtol=1e-3`.

The input case JSON records the actual ONNX identity (`path`, `sha256`,
`size_bytes`), the frozen `calibration_manifest` identity, the exact
`toolchain_image_id`, and three `real_inputs`. Each real-input entry maps every
source ONNX input name to its actual tensor file identity. Tensors must be
float32 `.npy` files in the ONNX input domain with the exact static input shape.
For SAM decoders, use the corresponding real encoder embeddings and boxes.
Hash the frozen calibration manifest independently before creating the case.

Both tools reread the declared real tensors, check their hashes/shapes/dtypes,
and add seed-0 input as a separate numerical check. The seed-0 input is not PTQ
calibration data. The CPU reference tool also verifies the calibration manifest
bytes. Use a new output location and preserve original reports and launch
receipts, including failed comparisons.

In the cached GPU SDK, with CUDA mapped and `NVIDIA_TF32_OVERRIDE=0`:

```bash
python3 /repo/utils/tools/validate_onnx_cuda.py \
  --case /work/frozen-real-inputs.json --output /work/cpu-cuda-parity.json
```

The report records the actual ONNX Runtime version, loaded providers, TF32
setting, input identities, every output comparison and the producer SHA.
A source-graph provider check is independent of target/march. It never
certifies a compiled HBM, SDK runtime compatibility or a board measurement.

If CPU/CUDA parity fails, keep that GPU attempt failed. A fresh CPU-only PTQ
attempt can use a separate frozen comparison between the source-export CPU
environment and the cached CPU SDK. First freeze all reference outputs in the
original source-export environment:

```bash
python3 utils/tools/validate_onnx_cpu_reference.py \
  --case /work/frozen-real-inputs-cpu-sdk.json --reference /work/cpu-reference
```

Then compare those exact outputs in the cached CPU SDK:

```bash
python3 /repo/utils/tools/validate_onnx_cpu_reference.py \
  --case /work/frozen-real-inputs-cpu-sdk.json --reference /work/cpu-reference \
  --output /work/cpu-sdk-parity.json
```

All heads must pass before that CPU PTQ attempt. Keep the checkpoint, graph,
calibration tensors and tolerances fixed. Use a new artifact/variant identifying
CPU PTQ, and retain the failed GPU report. Supplementary checks performed after
an existing compilation retain their actual execution order; do not relabel
them as checks that ran before compilation.

The current CPU verifier rejects missing, reordered or truncated reference cases,
unfrozen references and missing or extra output tensors. Historical run receipts
keep their actual producer hashes; upgrading the verifier does not relabel an
already executed check.
