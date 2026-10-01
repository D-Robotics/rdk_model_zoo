# Paraformer conversion

[中文](README_cn.md) · [Sample](../README.md) · [Python runtime](../runtime/python/README.md)

This directory exports the real-weight encoder, predictor and decoder directly
from the pinned FunASR architecture, with fixed deployment geometry and numerical
checks against Torch. CIF stays in the shared CPU implementation. **Real-audio calibration and explicit OE compilation orchestration are implemented.**
Actual OE/HMCT/SDK/board validation remains pending; the [host evaluator](../evaluator/README.md) is implemented;
successful FP32 export does not certify an HBM or its board behavior.

<a id="source-model"></a>
## Source model

The source model is
`iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404`.
The preserved S implementation splits it into encoder, predictor and decoder;
CPU CIF connects predictor outputs to decoder inputs. Published runtime assets
are described in [model preparation](../model/README.md). Converting your own
weights is a separate operation; graph transformations do not download weights,
produce HBM files or certify those published assets.

<a id="toolchain-targets"></a>
## Export environment and quickstart

Use Python 3.12 in a separate environment. The verified export stack is Torch and
torchaudio 2.6.0, FunASR 1.3.14, NumPy 1.26.4, ONNX 1.17.0, ONNX Runtime 1.20.1,
protobuf 4.23.0 and ModelScope 1.40.1. The exporter uses CPU only and does not
require a board SDK, ONNX Simplifier or OE. The architecture and preprocessing
support files are pinned; a local `model.pt` must load **every parameter strictly**.
Missing keys cannot fall back to random initial weights.

From the repository root:

```bash
python3.12 -m venv .venv-paraformer-export
. .venv-paraformer-export/bin/activate
python -m pip install -r samples/speech/paraformer/conversion/requirements-export.txt
python samples/speech/paraformer/conversion/export.py --help
```

<a id="export"></a>
## Export weights to ONNX

If you do not have the source weights, explicitly download them first. This is
about 913 MB for `model.pt`, plus metadata; reserve additional space for both raw
and rewritten ONNX stages. This example writes only to `models/paraformer-source`.
It fetches the hub's `master` revision; the export report records actual local
file hashes, **not an immutable publisher weight revision**.

```python
from modelscope import snapshot_download
snapshot_download(
    "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    revision="master",
    local_dir="models/paraformer-source",
    allow_file_pattern=["model.pt", "config.yaml", "tokens.json", "am.mvn"],
)
```

Export to a new directory:

```bash
python samples/speech/paraformer/conversion/export.py \
  --model-dir models/paraformer-source \
  --output-dir outputs/paraformer_export
```

For checks with your own audio, first use the [Python frontend](../runtime/python/README.md)
to prepare its feature NPY file, then add one or more `--feature` arguments to a
new export run:

```bash
python samples/speech/paraformer/conversion/export.py \
  --model-dir models/paraformer-source \
  --output-dir outputs/paraformer_export_with_audio \
  --feature outputs/paraformer_features/feats/BAC009S0724W0121.npy
```

The example `--feature` path is the first bundled utterance prepared by the
documented two-WAV preparation command writing to the same directory, shown in the
[native guide](../runtime/cpp/README.md#quickstart) (`--preprocess-only
--output-dir outputs/paraformer_features`). That directory spelling is an example,
not a CLI default; other guides use `outputs/paraformer-prepared` and
`outputs/paraformer-features` for the same preparation. Any preparation output
directory works if `--feature` points at its `feats/<utt_id>.npy` file. The feature
path must already exist; it is not a WAV. Each file must contain a finite float32
`[1,400,560]` array. Export tests use unmasked CIF to exercise model boundaries,
not the utterance's valid-frame count, and do not calculate CER.

| Argument | Meaning |
| --- | --- |
| `--model-dir` | Required local directory containing `model.pt`, `config.yaml`, `tokens.json`, `am.mvn`. No implicit download. Config, vocabulary and CMVN must match the pinned source digests. |
| `--output-dir` | Required new directory. Existing output is rejected; source weights and earlier exports are not overwritten. |
| `--feature` | Optional repeatable prepared NPY input. Zero/random feature checks and decoder counts 0, 1, 17, 100 run even when omitted. |
| `--threads` | Positive CPU thread count; default 4. |

Outputs are `encoder.onnx`, `predictor.onnx`, `decoder.onnx`, their `*.raw.onnx`
diagnostic exports, and `export-report.json`. Only the final filenames have the
validated physical names, shapes and graph transforms. Raw graphs can retain
Torch-generated names and are not the deployment interface.

A successful report has `status: completed`, source/feature/model SHA-256 values,
package versions, stage node counts and per-case maximum absolute differences.
Each full output must satisfy `rtol=1e-4, atol=1e-4`, retain its dtype/shape and be
finite. Parser/preflight failures return 2 before creating output; later errors
return 2 and preserve partial files plus a `failed` report. Partial artifacts are
not approved exports. A forcibly terminated process may leave incomplete state.

<a id="calibration"></a>
## Prepare real calibration and compiler configs

After export, prepare a new self-contained workspace. The following two-WAV
example checks the workflow; it is **not** a representative calibration dataset
or an accuracy acceptance run:

```bash
python samples/speech/paraformer/conversion/prepare.py \
  --export-dir outputs/paraformer_export \
  --wav-dir samples/speech/paraformer/test_data \
  --sample-count 2 \
  --output-dir outputs/paraformer_calibration
```

For a real conversion, supply a representative collection of your 16 kHz WAVs.
Selection is a sorted recursive prefix of lowercase `*.wav` files, default 50,
matching the source recipe's reference count. Fewer files are explicitly recorded;
50 files alone do not establish representative coverage. Empty selection, bad
sample rate, malformed audio or invalid stage outputs fail the run; no selected
file is silently skipped. There is no implicit resampling or random calibration.

The unified frontend averages multichannel audio, computes fbank/LFR/CMVN with
CPU seed 191009, and pads/truncates to 400 frames. This reuses the documented
runtime frontend rather than constructing an entire AutoModel just to extract
features. The source calibration script's global random behavior is not a
reproducibility contract; current seed, dependencies and input hashes are recorded.
Actual/retained frame counts and truncation remain visible in each record.

Encoder and predictor execute on CPU ONNX Runtime. Calibration then calls the
same CPU CIF implementation with **`real_T=None`**, preserving the source's
unmasked calibration distribution. Runtime inference uses the utterance's valid
frame count instead. A zero-token result is stored, not silently removed.

| Calibration subdirectory | Shape | dtype | Used for |
| --- | --- | --- | --- |
| `speech` | `[1,400,560]` | float32 | encoder input |
| `encoder_after_norm_Add_1_output_0` | `[1,400,512]` | float32 | predictor and decoder context |
| `predictor_Add_output_0` | `[1,401]` | float32 | recorded predictor weights / CIF provenance |
| `predictor_Concat_5_output_0` | `[1,401,512]` | float32 | recorded predictor hidden vectors / CIF provenance |
| `shape_8609` | `[1,100,512]` | float32 | decoder acoustic embeddings |
| `token_num` | `[1]` | int32 | decoder valid-token count |
| `bias_embed` | `[1,1,512]` | float32 | zero contextual bias |

Each directory contains aligned names such as `000000.npy`. The workspace also
contains `source/{encoder,predictor,decoder}.onnx`, the original export report and
CMVN snapshot, `configs/{encoder,predictor,decoder}.yaml`, and `preparation.json`.
The report binds each source/derived array to hashes, shape, dtype and value range.
Config model/calibration paths are relative to this workspace, so run an original
config from its workspace root; the compile wrapper below remaps paths explicitly.

| Preparation argument | Default / behavior |
| --- | --- |
| `--export-dir` | Required completed exporter directory; report hashes and actual ONNX signatures are checked. External-data ONNX files are rejected. |
| `--wav-dir` | Required real WAV directory; selected files are read and hashed from the same bytes. |
| `--output-dir` | Required new workspace; no overwrite or resume into partial output. |
| `--cmvn-file` | Sample `model/am.mvn`; must match the pinned digest. |
| `--sample-count` | Positive integer, default 50; actual selected count is recorded. |
| `--threads` | Positive ONNX Runtime CPU thread count, default 4. |
| `--jobs` | Positive OE compiler jobs recorded in YAML, default 32 as in the source. |

`status: prepared` means calibration/config creation only. Later preparation
failure leaves `status: preparation_failed`, the current audio, completed records
and partial files. Exit code is 2; partial workspaces cannot be compiled by the
wrapper. Missing required dependencies/preconditions can fail before output exists.

<a id="compile"></a>
## Explicit S100 / nash-e compilation

The recipe supports **S100 / nash-e only**. It retains source max calibration,
internal INT16 quantization, NCHW feature maps, O2 latency optimization, one BPU
core and disabled compiler cache. The integer token-count input remains int32.
Do not interpret internal INT16 settings as a guarantee of final physical I/O
precision. Real HBM signatures still need SDK validation before runtime use.

Use a matching installed S OE toolchain with `hb_compile`. The source records
`ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0` and hbdk4 4.7.5; its image registry and
availability have not been verified here, so no invented pull URL is provided.
The FP32 export Python environment alone does not provide the compiler. Run
the wrapper from an available repository checkout inside the OE environment,
with NumPy and PyYAML installed; this compilation step does not import Torch.

```bash
python samples/speech/paraformer/conversion/compile.py \
  --workspace outputs/paraformer_calibration \
  --output-dir outputs/paraformer_compiled \
  --compiler hb_compile
```

`--workspace` and a **new** `--output-dir` are required; `--compiler` defaults to
`hb_compile` and may name an explicit executable. The workspace path cannot
contain `;`, which is the OE calibration-directory separator. Move/mount the
complete preparation workspace into the toolchain environment before compiling.
The wrapper checks every snapshot/config/NPY digest and rejects added calibration
files, then writes per-run absolute-path configs without changing the prepared
workspace. It invokes `hb_compile -c <stage.yaml>` sequentially and rechecks the
preparation after compilation.

Each stage retains exact argv/cwd, UTC timestamps, config hash, separate complete
stdout/stderr logs and return code. A process-start error records no return code;
a nonzero exit or a zero exit without the expected nonempty HBM is a failure and
stops later stages. `compile-report.json` preserves completed stages on failure.
Use a new output directory to retry; earlier logs and partial artifacts remain.

<a id="validation"></a>
## Validate conversion results

Export must complete all numerical/signature checks; calibration must pass its
snapshot/array checks; compilation must retain successful stage logs and artifacts.
These are separate checks: a nonempty HBM is not a model-output validation. Use the
[host evaluator](../evaluator/README.md) on FP32/PTQ graphs to obtain per-utterance
transcripts and CER, then separately verify the HBM on S100 when available.

<a id="artifacts"></a>
## Produced artifacts

Expected paths relative to the compile run:

| Stage | HBM | Optional quantized ONNX for later evaluation |
| --- | --- | --- |
| encoder | `encoder/paraformer_encoder_int16.hbm` | `encoder/paraformer_encoder_int16_ptq_model.onnx` |
| predictor | `predictor/predictor_int16.hbm` | `predictor/predictor_int16_ptq_model.onnx` |
| decoder | `decoder/decoder_int16.hbm` | `decoder/decoder_int16_ptq_model.onnx` |

Even with all three nonempty HBM files and zero return codes, the result is
**`compiled_unverified`**, not board/SDK/accuracy acceptance. Generated files are
not automatically published, renamed to official assets or copied into runtime
model directories. This host has no OE compiler; only orchestration fixtures and
the actual missing-compiler rejection were tested. Calibration preparation does
not certify quantization quality, and a missing PTQ ONNX is explicitly recorded
as absent rather than fabricated. See the [calibration/compile preparation evidence](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-calibration-review.md).

## Fixed deployment semantics

The encoder always processes 400 frames; actual utterance length is applied at
CPU CIF during inference. Predictor returns 401 weights and 401 hidden vectors,
including the source 0.45 tail and zero hidden frame. Decoder has physical width
100: `token_num` changes the valid-prefix mask, not the output shape. Its inputs
retain the published names, including `onnx::Shape_8609`; explicit output binding
avoids relying on a particular Torch internal tensor-number suffix.

The decoder composition is adapted from FunASR under the included
[MIT notice](LICENSE-FunASR). The fixed-width masks express the existing deployment
contract directly, replacing the old one-probe Range freezing. The generic upstream
decoder with an **unpadded** token sequence is not numerically interchangeable
for shorter sequences; that comparison failed and is retained in the evidence.
The deployment comparison instead uses the unmodified upstream decoder export
followed by the actual archived fixed-100 Range pass. Do not infer original
variable-length-model accuracy or HBM accuracy from these export checks.

## Graph tests

```bash
python -m unittest discover -s samples/speech/paraformer/tests -v
```

The earlier standalone graph tests also ran with Python 3.14.7, NumPy 2.5.3,
ONNX 1.23.0 and ORT 1.30.0; those versions do not establish FunASR export
compatibility. Missing optional dependencies can skip tests. Use the export
environment above and check the actual test counts rather than treating skips
as evidence of successful validation.

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
contains the full original Chinese walkthrough (historical `../../../../platforms/s/samples/speech/paraformer/conversion/README_cn.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md).
It remains historical reference, with the following migration boundaries:

| Source stage | Purpose | Unified status |
| --- | --- | --- |
| `01_reexport_fixed_shape.py` | Export fixed-shape model | Replaced by direct `export.py` stage export; no full CIF graph, global monkey patch or source overwrite. |
| `02_extract_decoder.py`, `07_extract_predictor.py`, `08_extract_encoder.py` | Extract three stages from an internal-name-dependent full graph | Replaced by explicit Torch stage boundaries with the same deployment names and shapes. |
| `03_convert_gather_int64_to_int32.py` through `06_shape_freeze.py` | Adapt Gather, order, Range and axes | Shared primitives are integrated into real-weight stage export. No simplifier is invoked, so no unchecked simplifier result is accepted. |
| `09_gen_calib_features.py` | Generate features from representative real audio | Implemented in `prepare.py` through the unified deterministic frontend. |
| `10_gen_real_calib.py` and `cif_numpy.py` | Run stages to prepare decoder/predictor calibration | Implemented with real encoder/predictor execution and shared unmasked CIF (`real_T=None`), unlike runtime valid-frame masking. |
| Three `*_int16.yaml` files | Compile encoder, predictor and decoder for `nash-e` | Generated with consistent workspace paths and source settings. Explicit OE invocation is implemented; actual compiler/SDK validation is not-run. |
| `11_eval_pipeline.py` | Compare the three-stage speech pipeline | [Dedicated evaluator](../evaluator/README.md): real FP32 smoke comparison; HMCT adapter retained, actual HMCT not-run. |

Source recipe settings include maximum calibration, INT16 internal operations,
O2 latency optimization and a single BPU core. These settings do not establish
the final physical input/output dtypes or compatibility of a new HBM; the runtime
still validates the actual stage signatures. Source `out/` script defaults and
root-relative YAML paths are inconsistent, so running them unchanged is not a
validated end-to-end procedure. Source benchmark numbers remain historical;
none of these host graph tests establishes CER, latency or dataset accuracy.

<a id="known-gaps"></a>
## What has been checked

Ten graph tests cover shared Gather constants, constant overflow and evaluation
limits, explicit dynamic casts, unique intermediate names, stable dependency
sorting, rejection of incomplete graphs, static versus dynamic Range, and shared
negative axes across different ranks. Numerical comparisons execute the
original and rewritten small graphs with real ONNX Runtime and require equal
output dtypes and values. Real-weight stage export and two complete example pipelines are checked separately.
OE compilation, SDK execution, board tests and dataset CER remain unverified. See the [host evidence](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-graph-ops-review.md).

Real-weight results, initial failures and reproduction commands are in the
[export report](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-export-review.md).

The 16 export checks use contexts produced by the real encoder (zero/random
features and two real audio features). Separate arbitrary random hidden-vector
stress tests showed much larger Torch/ORT differences; they do **not** satisfy
the export tolerance. The old and new fixed-width ONNX graphs agree in those
stress cases under the same ORT settings. This preserves source deployment
behavior but does not establish global Torch/ONNX equivalence. Both sample
transcripts also contain recognition errors against their references; no dataset
CER or accuracy improvement is claimed.
