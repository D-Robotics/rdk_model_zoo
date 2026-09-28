# Paraformer Python CPU bridge

[简体中文](README_cn.md)

This directory currently provides the CPU continuous integrate-and-fire (CIF)
bridge. The complete unified audio frontend, three-model runtime, command-line
entry and C++ bridge are still being migrated. This is not a runnable board
sample yet. Do not interpret the host checks below as model inference or accuracy
validation. The archived [S runtime](../../../../../platforms/s/samples/speech/paraformer/runtime/python/README.md)
remains a historical reference, not the unified entry.

## Place in the pipeline

The source pipeline is audio → FunASR frontend → encoder → predictor → CPU CIF
→ decoder → token text. CIF takes predictor weights and hidden states and produces
fixed-size acoustic embeddings plus the token count for the decoder. It performs
no model execution, file access, token decoding or device selection. Keeping it
in [cif.py](cif.py) allows inference and calibration to use the same arithmetic
without adding unrelated helpers to a model inference class.

## Requirements and runnable host example

Use Python with NumPy installed. No Torch, FunASR, vendor SDK or board is needed
for this numerical bridge. From the repository root:

```bash
python - <<'PYCODE'
import numpy as np
from samples.speech.paraformer.runtime.python.cif import cif_numpy

weights = np.zeros((1, 401), dtype=np.float32)
hidden = np.zeros((1, 401, 512), dtype=np.float32)
weights[0, :3] = [0.75, 0.75, 0.5]
hidden[0, :3] = np.array([2, 6, 10], dtype=np.float32)[:, None]
embeddings, token_count = cif_numpy(weights, hidden, real_T=3)
print(embeddings.shape, token_count.tolist(), embeddings[0, :2, 0].tolist())
PYCODE
```

Expected output: `(1, 100, 512) [2] [3.0, 8.0]`. The two embeddings integrate the
weighted hidden states across the two integer crossings. This synthetic example
is not a speech-recognition result.

## API contract

| Argument/result | Contract |
| --- | --- |
| `alphas` | finite, nonnegative `float32`, shape `[1,401]` |
| `concat5` | finite `float32`, shape `[1,401,512]` |
| `real_T` | required keyword; integer `0…400` for inference, explicitly `None` only for unmasked calibration |
| acoustic embeddings | owned `float32 [1,100,512]`; unused rows zero |
| token count | owned `int32 [1]`, capped at 100 |

Inference masks weights at and after `real_T` before accumulation. Zero valid
frames or total weight below one returns zero embeddings and a zero count. The
caller must handle that count; this function does not decide whether to execute
the decoder. Partial weight below the next integer is not emitted. More than 100
emitted embeddings retain the first 100, matching the source contract.

The arithmetic preserves the source's float64 cumulative sums rounded to float32
and its one-fire-per-frame rule. It is not a general multi-fire integrator for
weights above one. Inputs remain unchanged. Invalid shape, dtype, non-finite
values, negative weights or invalid frame count raise an error before integration;
there is no implicit cast or batch-size expansion. Explicit `real_T=None` preserves
the source calibration's unmasked distribution and must not be used to replace
inference masking.

## Verification and remaining work

```bash
python -m unittest discover -s samples/speech/paraformer/tests -v
```

Seven host tests cover empty output, manually derived fractional crossings,
padding versus calibration mode, the 100-token cap, 24 source comparisons,
input ownership and invalid contracts. The comparison loads the archived source
from S commit `380e1a2bf42041af54be6f34935e50197cfadff9`; its no-fire case raises
`IndexError`, which the unified bridge fixes. See the
[review and evidence](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-cif-review.md).

The source publishes S100 models only. No X5, S100P or S600 adaptation is claimed.
Real frontend equivalence, SDK bindings, three-stage inference, native C++, full
conversion/evaluation workflows and bilingual sample documentation remain open.
Board inference, OE compilation, dataset CER and latency have not been run.
