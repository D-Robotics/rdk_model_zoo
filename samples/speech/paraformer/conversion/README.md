# Paraformer conversion

[中文](README_cn.md) · [Sample](../README.md) · [Python runtime](../runtime/python/README.md)

This directory currently provides **host graph transformations**, not a complete
Paraformer exporter or an OE compilation recipe. The unified export, stage
extraction, calibration and compiler orchestration are still being migrated.
Do not treat passing graph tests as validation of a converted speech model.

The source model is
`iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404`.
The preserved S implementation splits it into encoder, predictor and decoder;
CPU CIF connects predictor outputs to decoder inputs. Published runtime assets
are described in [model preparation](../model/README.md). Converting your own
weights is a separate operation; graph transformations do not download weights,
produce HBM files or certify those published assets.

## Requirements and verification

The graph module imports NumPy and ONNX. Numerical tests additionally require
ONNX Runtime with its CPU execution provider. Host verification used Python
3.14.7, NumPy 2.5.3, ONNX 1.23.0 and ONNX Runtime 1.30.0. These are the observed
**graph-test environment**, not a supported FunASR export or OE environment.
Use a separate environment for the source Torch/FunASR exporter; compatibility
of that exporter with these versions has not been established.

From the repository root, in an environment containing these dependencies:

```bash
python -c 'import numpy, onnx, onnxruntime; print(numpy.__version__, onnx.__version__, onnxruntime.__version__)'
python -m unittest samples.speech.paraformer.tests.test_conversion_graph_ops -v
```

An absent ONNX/ORT dependency causes the test module to skip. A skipped test is
not a successful graph validation: check that all tests actually ran.

## Available transformations

All functions accept an in-memory ONNX `ModelProto`, operate on a copy and return
another model. They do not read or write files, download anything, or run OE.
Graph transformation failures raise an exception without changing the caller's
model. Standard flat tensor graphs are supported; nested control flow is
explicitly rejected. Sparse initializers and custom-domain operators are not
supported by the constant evaluator.

| Function | Behavior and failure boundary |
| --- | --- |
| `topological_sort(model)` | Orders nodes by tensor dependencies, retaining model metadata and repeated node names. Rejects cycles, missing inputs/outputs and duplicate tensor producers. It does not perform full ONNX validation; call `onnx.checker.check_model` afterwards. |
| `gather_indices_int32(model)` | Gives each Gather index an INT32 representation while preserving constants used by other consumers. Rejects constant overflow, unresolved types and dynamic INT64 indices by default. |
| `gather_indices_int32(model, allow_dynamic=True)` | Adds Cast for unresolved runtime INT64 values only under a **caller-established INT32 range contract**. It does not insert a runtime bounds check: out-of-range runtime values can wrap. Oversized direct constants are rejected, not silently treated as dynamic. |
| `fold_constant_ranges(model)` | Replaces Range only when its value follows from supported deterministic constant operations or fixed shape metadata. Rejects data-dependent, unsupported or oversized results. Never freezes a Range from one example inference. |
| `normalize_axes(model)` | Normalizes negative axes for supported input-rank operations, copying shared axis tensors per consumer. Rejects unknown rank when normalization is required. Unsqueeze is intentionally excluded because its axes use output rank. |

The last three functions run the ONNX checker before returning. Constant
propagation uses a limited operator whitelist and a one-million-element cap for
accepted constants/results. This is a supported-evaluation boundary, not a
security sandbox or a guarantee about total process memory. Supplied shape
metadata must describe the actual model inputs; these passes do not establish
that contract by observing runtime data.

## Executable API example

Run this from the repository root with ONNX installed. It is a small graph
example, not the Paraformer model and not a conversion command for its weights.
It also shows the explicit checker needed after sorting alone.

```python
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from samples.speech.paraformer.conversion.graph_ops import (
    gather_indices_int32,
    topological_sort,
)

model = helper.make_model(
    helper.make_graph(
        [helper.make_node("Gather", ["features", "index"], ["selected"])],
        "gather-example",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("selected", TensorProto.FLOAT, [1])],
        initializer=[numpy_helper.from_array(np.array([1], np.int64), "index")],
    ),
    opset_imports=[helper.make_opsetid("", 15)],
    ir_version=10,
)
original = model.SerializeToString()
rewritten = gather_indices_int32(topological_sort(model))
onnx.checker.check_model(rewritten)
assert model.SerializeToString() == original
assert rewritten.graph.node[0].input[1] != "index"
print("Gather-only rewrite validated; input model preserved")
```

When integrating a pass into a file workflow, validate the result and compare
original/rewritten outputs using representative inputs before publishing a new
file. Do not overwrite the source model. A dynamic-index cast additionally
requires a range argument based on the model contract, not merely a few passing
inputs. The API deliberately does not automate either decision.

## Source workflow and remaining migration

The source snapshot at S commit `380e1a2bf42041af54be6f34935e50197cfadff9`
contains [the full original Chinese walkthrough](../../../../platforms/s/samples/speech/paraformer/conversion/README_cn.md).
It remains historical reference, with the following migration boundaries:

| Source stage | Purpose | Unified status |
| --- | --- | --- |
| `01_reexport_fixed_shape.py` | Export fixed-shape full model, 400 feature frames and up to 100 tokens | Pending; source export mutates patches and existing outputs, so do not assume safe reruns. |
| `02_extract_decoder.py`, `07_extract_predictor.py`, `08_extract_encoder.py` | Extract the three ONNX stages | Pending; single-feed boundary probes must not be interpreted as proofs of constant values. |
| `03_convert_gather_int64_to_int32.py` through `06_shape_freeze.py` | Adapt Gather, order, Range and axes | Shared primitives above are implemented and tested on small graphs; full-model integration and simplifier validation remain pending. |
| `09_gen_calib_features.py` | Generate features from representative real audio | Pending integration with the unified, reproducible frontend. |
| `10_gen_real_calib.py` and `cif_numpy.py` | Run stages to prepare decoder/predictor calibration | Pending; source calibration deliberately uses unmasked CIF (`real_T=None`), unlike runtime valid-frame masking. |
| Three `*_int16.yaml` files | Compile encoder, predictor and decoder for `nash-e` | Historical source recipes only; no unified OE invocation or fresh compiled model validation yet. |
| `11_eval_pipeline.py` | Compare the three-stage speech pipeline | Pending migration into a dedicated evaluator. |

Source recipe settings include maximum calibration, INT16 internal operations,
O2 latency optimization and a single BPU core. These settings do not establish
the final physical input/output dtypes or compatibility of a new HBM; the runtime
still validates the actual stage signatures. Source `out/` script defaults and
root-relative YAML paths are inconsistent, so running them unchanged is not a
validated end-to-end procedure. Source benchmark numbers remain historical;
none of these host graph tests establishes CER, latency or dataset accuracy.

## What has been checked

Nine graph tests cover shared Gather constants, constant overflow and evaluation
limits, explicit dynamic casts, unique intermediate names, stable dependency
sorting, rejection of incomplete graphs, static versus dynamic Range, and shared
negative axes across different ranks. Numerical comparisons execute the
original and rewritten small graphs with real ONNX Runtime and require equal
output dtypes and values. The full sample regression is separate from real
weights, OE compilation, SDK execution and board tests, all unverified for this
conversion work. See the [host evidence](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-graph-ops-review.md).
