English | [简体中文](README_cn.md)

# Paraformer conversion

[中文](README_cn.md) · [Sample](../README.md) · [Python runtime](../runtime/python/README.md)

This directory exports the real-weight encoder, predictor and decoder directly
from the pinned FunASR architecture, with fixed deployment geometry and numerical
checks against Torch, prepares real-audio calibration and orchestrates explicit
OE compilation. CIF stays in the shared CPU implementation. HMCT/OE compilation
runs in the OE toolchain environment per [compile](#compile); verify the compiled
HBM on the board per [validation](#validation) and the
[evaluator](../evaluator/README.md).

<a id="source-model"></a>
## Source model

The source model is
`iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404`.
The preserved S implementation splits it into encoder, predictor and decoder;
CPU CIF connects predictor outputs to decoder inputs. Published runtime assets
are described in [model preparation](../model/README.md). Converting your own
weights is a separate operation; graph transformations do not download weights
or produce HBM files.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── calibration.py  # Python script
├── compile.py  # Python script
├── configuration.py  # Python script
├── export.py  # Python script
├── graph_ops.py  # Python script
├── onnx_stage.py  # Python script
├── prepare.py  # Python script
├── requirements-export.txt  # Source or data file
├── torch_stages.py  # Python script
└── workspace.py  # Python script
```

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
It fetches the hub's `master` revision; the export report records the actual
local file hashes.

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
`[1,400,560]` array. Export tests use unmasked CIF to exercise model boundaries
rather than the utterance's valid-frame count.

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

<a id="fixed-graph-contract"></a>
## Fixed input geometry and ONNX preparation

The encoder speech input is float32 `[1,400,560]`: 16 kHz audio, fbank followed by LFR with `m=7,n=6`. **400 LFR frames represent about 24 seconds of audio**. The frontend pads shorter inputs and truncates longer ones to this fixed width; split longer recordings explicitly before preparation. The float32 `bias_embed` is `[1,1,512]`; zeros disable hotwords. CPU CIF caps the decoder token sequence at `max_label_len=100`; it takes about 1–2 ms in the source CPU implementation, while the resident measurements below use their own timing scope.

For the [source graph and compiler recipe](https://github.com/D-Robotics/rdk_model_zoo/blob/d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d/platforms/s/samples/speech/paraformer/conversion/README_cn.md), the preparation sequence is:

1. Export the FunASR model to `model.onnx`. Extract `/decoder/*`, treating non-decoder tensors as boundaries, to obtain the five-input `decoder_only.onnx`; extract encoder and predictor subgraphs at their respective boundaries too.
2. Convert Gather index constants from INT64 to INT32, then sort nodes topologically and run the ONNX checker. The source graph required **145 Constant INT64→INT32 conversions** to avoid HMCT `adjust_multi_output_use_quant_info_pass_fail`.
3. Replace static Range operations with Constants. The source procedure evaluated them with an ONNX Runtime probe, then propagated fixed shapes and folded constants with ONNX Simplifier: **2903 → 1214 nodes**.
4. Fold `/predictor/Gather_output_0` to Constant(1) for batch=1, removing dynamic Tile repeats, and normalize negative axes to positive axes. The resulting `decoder_only_final.onnx` contained **1137 nodes** and passed HBDK4 export. These counts describe that exact source graph, rather than a required count for every source-weight export.
5. Keep CIF on CPU and expose `pre_acoustic_embeds` as a decoder graph input. Replace CIF `remains.unsqueeze(-1)` with `remains.reshape(-1,1)` when compiling with HBDK4 4.7.5.

Use the `export.py` commands above for this checkout: they export each stage directly from Torch and validate it numerically. For custom ONNX graphs, [graph_ops.py](graph_ops.py) provides `gather_indices_int32`, `topological_sort`, `fold_constant_ranges` and `normalize_axes`, described below. `fold_constant_ranges` requires provably constant values; a single runtime probe is insufficient for a data-dependent Range. The standalone source scripts `extract_decoder_predictor.py`, `convert_gather_int64_to_int32.py`, `topsort.py`, `fold_range_ops.py` and `shape_freeze.py` identify the source procedure's stages; they are not commands shipped in this directory.

| Compiler problem | Diagnostic | Graph preparation |
| --- | --- | --- |
| INT64 Gather | HMCT `adjust_multi_output_use_quant_info_pass_fail` | Convert the 145 source Constant indices from INT64 to INT32; reject indices outside the INT32 range. |
| Range | HBDK4 `Operator Range should be optimized` | Source procedure: ORT probe → Constant; current helper: fold only proven static values. |
| Unsqueeze type inference | HBDK4 `type_inf` for CIF `remains.unsqueeze(-1)` | Use `.reshape(-1,1)`; see [CIF shape workaround](#cif-shape-workaround). |
| Dynamic GatherND index | `size of last dimension of index cannot be dynamic` | Keep CIF on CPU and feed `pre_acoustic_embeds` to the decoder. |
| Dynamic Tile repeats | `cannot create ArrayAttr from OpResult` | Fold `/predictor/Gather_output_0` to Constant(1) under batch=1. |
| Split axis=-1 | HBIR slice shape inference failure | Normalize negative axes to positive axes using the known input rank. |

<a id="calibration"></a>
## Prepare real calibration and compiler configs

After export, prepare a new self-contained workspace. The following two-WAV
example only checks the workflow; supply a representative calibration set for a
real quantization.

Calibration data must come from the real audio distribution. The source
release's two quantization-failure records serve as reference: calibrating
with `np.random.randn` random data drove the INT16 pipeline CER to 100%
(decoder argmax collapsed to `</s>` everywhere; FP16 outputs all NaN), the
root cause being the mismatch between the N(0,1) random distribution and the
encoder's actual output range (~[-0.4, 0.3]); and without masking padding via
`alphas[:, real_T:] = 0`, the FP32 pipeline CER rose from ~5% to 44.4% (first
N characters correct, garbage afterwards) because padding produced spurious
CIF fires. Both fixes are built in: calibration uses 50 real utterances and
the runtime CIF masks by valid frame count:

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

The shared frontend averages multichannel audio, computes fbank/LFR/CMVN with
CPU seed 191009, and pads/truncates to 400 frames. This reuses the documented
runtime frontend rather than constructing an entire AutoModel just to extract
features. Seed, dependencies and input hashes are recorded with each run;
actual/retained frame counts and truncation remain visible in each record.

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
config from its workspace root; the compile helper below remaps paths explicitly.

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
helper. Missing required dependencies/preconditions can fail before output exists.

<a id="compile"></a>
## Explicit S100 / nash-e compilation

The recipe supports **S100 / nash-e only**. It retains source max calibration,
internal INT16 quantization, NCHW feature maps, O2 latency optimization, one BPU
core and disabled compiler cache. The integer token-count input remains int32.
Do not interpret internal INT16 settings as a guarantee of final physical I/O
precision. Real HBM signatures still need SDK validation before runtime use.

Use a matching installed S OE toolchain with `hb_compile`. The source
recipe's image is `ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`
(hbdk4 4.7.5); the original acquisition and start commands:

```bash
docker pull ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
docker run --rm -it \
    -u $(id -u):$(id -g) \
    --entrypoint /bin/bash \
    -v /path/to/workspace:/workspace \
    -w /workspace \
    ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

The FP32 export Python environment alone does not provide the compiler. Run
the helper from an available repository checkout inside the OE environment,
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
The helper checks every snapshot/config/NPY digest and rejects added calibration
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
transcripts and CER, then verify the HBM on S100.

On the board, each stage HBM can be measured directly with the built-in
command (single-BPU-core latency):

```bash
hrt_model_exec perf --model_file encoder_int16.hbm  --thread_num 1 --frame_count 200
hrt_model_exec perf --model_file predictor_int16.hbm --thread_num 1 --frame_count 500
hrt_model_exec perf --model_file decoder_int16.hbm  --thread_num 1 --frame_count 200
```

Source-release records for this recipe (S100, single BPU core):

| Item | Encoder | Predictor | Decoder |
| --- | --- | --- | --- |
| HBM size | 211.5 MB | ~4 MB | 73.5 MB |
| Static latency (compiler) | 32.52 ms (FPS 30.75) | ~0.35 ms | 5.77 ms (FPS 173) |
| Board `hrt_model_exec perf` | 33.11 ms (FPS 30.18) | 0.67 ms | 6.12 ms (FPS 162.8) |
| Memory | static 211 MB, dynamic 3.6 MB, DDR 499 MB | — | DDR 127 MB |

Encoder compile time is ~55 min at `jobs=32`. The single BPU core is
saturated on S100; `--thread_num 8` only queues concurrently and does not
change single-frame latency. The board Python manual for `hbm_runtime` is the
[S Python API guide](https://developer.d-robotics.cc/rdk_s_doc/Algorithm_Application/python-api).


### Compiled model sizes and static estimates

Source: [S100 INT16 conversion and performance report](https://github.com/D-Robotics/rdk_model_zoo/blob/d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d/platforms/s/samples/speech/paraformer/conversion/README_cn.md). The model sizes, compiler estimates and board measurements below describe the same recipe; their timing scopes are stated separately.

Configuration: S100 `nash-e`, all-node INT16, max calibration, `O2`, latency mode, `core_num: 1` and `jobs: 32`; the common configuration uses `cache_mode: disable`. Calibration uses fbank+LFR features from 50 real recordings. Predictor inputs are FP32 encoder outputs; the decoder receives four real inputs from the full FP32 pipeline. Static latency and FPS are compiler estimates. Memory and DDR figures retain the report’s measurement scope.

#### Encoder

| Item | Value |
|---|---|
| HBM size | 211.5 MB |
| Static latency | 32.52 ms (FPS 30.75) |
| Static memory | 211 MB |
| Dynamic memory | 3.6 MB |
| DDR usage | 499 MB |
| Compilation time | ~55 min (jobs=32) |

#### Predictor

| Item | Value |
|---|---|
| HBM size | ~4 MB |
| Static latency | ~0.35 ms (FPS ~2877) |
| Outputs | `/predictor/Add_output_0` [1,401] (alphas) + `/predictor/Concat_5_output_0` [1,401,512] |

#### Decoder

| Item | Value |
|---|---|
| HBM size | 73.5 MB |
| Static latency | 5.77 ms (FPS 173) |
| DDR usage | 127 MB |
| Inputs | 4: encoder_out, token_num, bias_embed, pre_acoustic_embeds |
| Outputs | `logits [1, 100, 8404]` |

<a id="artifacts"></a>
## Produced artifacts

Expected paths relative to the compile run:

| Stage | HBM | Optional quantized ONNX for later evaluation |
| --- | --- | --- |
| encoder | `encoder/paraformer_encoder_int16.hbm` | `encoder/paraformer_encoder_int16_ptq_model.onnx` |
| predictor | `predictor/predictor_int16.hbm` | `predictor/predictor_int16_ptq_model.onnx` |
| decoder | `decoder/decoder_int16.hbm` | `decoder/decoder_int16_ptq_model.onnx` |

With all three nonempty HBM files and zero return codes, the compile report
records status **`compiled_unverified`** until the board/SDK checks under
[validation](#validation) pass.
Generated files are not automatically published, renamed to official assets or
copied into runtime model directories. A missing PTQ ONNX is recorded as absent
in the report.

## Fixed deployment semantics

The encoder always processes 400 frames; actual utterance length is applied at
CPU CIF during inference. Predictor returns 401 weights and 401 hidden vectors,
including the source 0.45 tail and zero hidden frame. Decoder has physical width
100: `token_num` changes the valid-prefix mask, not the output shape. Its inputs
retain the published names, including `onnx::Shape_8609`; explicit output binding
avoids relying on a particular Torch internal tensor-number suffix.

The decoder composition is adapted from FunASR under the included
[MIT notice](LICENSE-FunASR). The fixed-width masks express the deployment
contract directly. The generic upstream
decoder with an **unpadded** token sequence is not numerically interchangeable
for shorter sequences.
The deployment comparison uses the unmodified upstream decoder export
followed by the fixed-100 Range pass.

## Graph tests

```bash
python -m unittest discover -s samples/speech/paraformer/tests -v
```

Run the suite in the export environment above; missing optional dependencies
cause skips, so check the reported test counts.

<a id="cif-shape-workaround"></a>
## CIF shape workaround

The HBDK4 4.7.5 conversion path can raise a `type_inf` error for CIF's
`remains.unsqueeze(-1)`. Express the same final dimension with
`remains.reshape(-1, 1)`; the exporter emits an ONNX Reshape with an explicit
output rank for the compiler.

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

<a id="known-gaps"></a>
## Known gaps

- HMCT/OE compilation runs in the OE toolchain environment per [compile](#compile);
  a generated HBM keeps status `compiled_unverified` until the board/SDK checks
  under [validation](#validation) pass.
- The numerical export checks run on contexts produced by the real encoder
  (zero/random features and two real audio features). Arbitrary random
  hidden-vector inputs produce much larger Torch/ONNX Runtime differences and do
  not satisfy the export tolerance; the checks are defined on real-feature
  contexts.
- The two-utterance calibration preparation is a workflow example; supply a
  representative 16 kHz WAV collection for a real quantization run.
