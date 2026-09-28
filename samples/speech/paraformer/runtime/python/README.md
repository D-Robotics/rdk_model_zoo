# Paraformer Python pipeline and CPU bridge

[简体中文](README_cn.md)

This directory provides the CPU continuous integrate-and-fire (CIF) bridge and
three-model application pipeline with SDK adapters and metadata binding. The real
CPU audio frontend and complete Python CLI are available. Real board inference
has not been run. The [C++ bridge](../cpp/README.md#quickstart) now consumes the
prepared features through a complete native entry. Do not interpret the host checks below as model inference or accuracy
validation. The archived [S runtime](../../../../../platforms/s/samples/speech/paraformer/runtime/python/README.md)
remains a historical reference, not the unified entry.

Start with [CLI usage](#usage), [all parameters](#parameters) and [results](#results); the later sections explain the numerical/API contracts.

<a id="environment"></a>
### Install dependencies explicitly

For a host frontend-only environment, from repository root:

```bash
python3.12 -m venv .venv-paraformer
.venv-paraformer/bin/python -m pip install -r samples/speech/paraformer/runtime/python/requirements-frontend.txt
```

The verified host is macOS arm64 / Python 3.12 with Torch and torchaudio 2.6.0,
FunASR 1.3.14, NumPy 1.26.4, SoundFile 0.14.0 and protobuf 4.23.0. The requirements
preserve the source's direct version constraints; the full observed host dependency
list is in the evidence and is not a board lockfile. On Linux, the source installs
Torch/torchaudio from the official CPU wheel index; wheel availability must match
your architecture and Python version. This frontend-only environment does not
provide `hbm_runtime`. Install/build a board runtime environment separately before
using the SDK API. Runtime code never runs pip or creates virtual environments.

### Run frontend without a board

Use the environment above as `python` (activate it or substitute its interpreter
path). From repository root, the bundled source WAV should print
`(1, 400, 560) 71 71 False`. FunASR can also print an ffmpeg availability notice;
this example uses SoundFile input and does not need ffmpeg.

```bash
python - <<'PYCODE'
from pathlib import Path
import soundfile as sf
from samples.speech.paraformer.runtime.python.frontend import ParaformerFrontend

sample = Path("samples/speech/paraformer")
waveform, sample_rate = sf.read(sample / "test_data/audio/BAC009S0724W0121.wav", dtype="float32")
frontend = ParaformerFrontend(sample / "model/am.mvn")
prepared = frontend.pre_process(waveform, sample_rate)
print(prepared.tensor.shape, prepared.valid_frames, prepared.original_frames, prepared.truncated)
PYCODE
```

Seven real FunASR cases—both bundled WAVs, derived stereo, silence, 30-second input,
one 25 ms window and a 10 ms short window—are byte-identical to the pinned S source
frontend under the recorded dependencies. Repeated outputs match, the source's
RNG mutation is reproduced, and the new adapter restores CPU RNG on both success
and an injected backend error. These checks verify features, not ASR transcripts
or board inference. See [frontend evidence](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-frontend-review.md).

<a id="usage"></a>
## CLI usage

The default invocation (no mode/input flags) performs S100 inference on the two
bundled manifest entries, detects the local target, uses default model paths and
writes a new `outputs/paraformer` directory. It requires actual board SDK/models.
Help, model listing, selection preview and real frontend preparation are available
without a board; the latter needs the frontend environment. No mode downloads files
or installs dependencies. All paths below assume repository root as cwd.

```bash
# cwd: repository root; use the documented frontend environment as python
python samples/speech/paraformer/runtime/python/main.py --list-models
python samples/speech/paraformer/runtime/python/main.py --target s100 --dry-run
python samples/speech/paraformer/runtime/python/main.py --preprocess-only --output-dir outputs/paraformer-prepared
```

Preparation succeeds with exit code 0, `result.json` status `completed`, two NPY
feature files and a separate `prepared-manifest.json`; expected valid lengths are
71 and 78. It does not rewrite `test_data/manifest.json`. The first two commands
need only the core host dependencies and never load Torch/FunASR/SDK. An explicit
unsupported target is rejected even in preview/preparation; dry-run with `auto`
requires `--target s100`. For one input and a different output directory:

```bash
# cwd: repository root
python samples/speech/paraformer/runtime/python/main.py --preprocess-only --audio-file samples/speech/paraformer/test_data/audio/BAC009S0724W0168.wav --output-dir outputs/paraformer-one
```

On an S100, after explicitly preparing the model package and runtime environment,
this is the board inference command (not executed in this host-only migration):

```bash
# cwd: repository root; S100 with matching SDK and frontend dependencies only
bash samples/speech/paraformer/model/download_model.sh --target s100
python samples/speech/paraformer/runtime/python/main.py --target s100 --output-dir outputs/paraformer-inference
```

`bash samples/speech/paraformer/runtime/python/run.sh` forwards exactly the same
arguments and honors the `PYTHON` environment variable. Unlike the source wrapper,
it does not create a virtual environment, install packages or download models.
No legacy positional data-directory argument is silently accepted.

<a id="parameters"></a>
## Parameters

`null` in the table means the option is omitted at parsing time. The application
then resolves the documented fallback; paths shown below are rooted in the sample
unless explicitly described as relative to the working directory.


| Parameter | Default | Meaning |
| --- | --- | --- |
| `--target` | `auto` | auto / x5 / s100 / s100p / s600; only S100 has assets |
| `--list-models` | `false` | List the three model records without execution |
| `--dry-run` | `false` | Validate selection without loading files/SDK; use explicit target |
| `--preprocess-only` | `false` | Run real CPU frontend only; auto declares S100 without board detection |
| `--manifest` | `null` | JSON list; mutually exclusive with audio-file; effective fallback: `test_data/manifest.json` |
| `--audio-file` | `null` | One WAV; ID is its filename stem |
| `--audio-dir` | `null` | Manifest WAV directory; incompatible with audio-file; effective fallback: `manifest parent/audio` |
| `--output-dir` | `outputs/paraformer` | Must not already exist |
| `--max-utts` | `0` | Nonnegative; zero all, positive first N |
| `--cmvn-path` | `samples/speech/paraformer/model/am.mvn` | Pinned bundled CMVN; path relative to sample by default |
| `--tokens-path` | `samples/speech/paraformer/model/s100/tokens.json` | Pinned ordered vocabulary for inference; unused by preparation |
| `--random-seed` | `191009` | CPU frontend seed; integer in [0,2**63) |
| `--priority` | `null` | Optional 0–255; inference only |
| `--bpu-cores` | `null` | Optional nonempty list of nonnegative indexes; inference only |
| `--encoder-model-path` | `null` | External path: provide all three paths and matching IDs; effective fallback: `published default` |
| `--encoder-asset-id` | `null` | Exact stage ID; provide all three IDs if any is specified |
| `--predictor-model-path` | `null` | External path: provide all three paths and matching IDs; effective fallback: `published default` |
| `--predictor-asset-id` | `null` | Exact stage ID; provide all three IDs if any is specified |
| `--decoder-model-path` | `null` | External path: provide all three paths and matching IDs; effective fallback: `published default` |
| `--decoder-asset-id` | `null` | Exact stage ID; provide all three IDs if any is specified |

Model/CMVN/token defaults are absolute sample-local paths at runtime; only the
output default is relative to cwd. Help is provided by argparse (`-h`/`--help`).
The three mode flags are mutually exclusive. Input manifests must be nonempty,
IDs unique filename stems (no `/`, `\`, NUL, `.` or `..`), and optional `text`
strings. All records are structurally checked, then `max-utts` selects the prefix;
missing selected WAVs fail instead of silently reducing the evaluated set.

<a id="results"></a>
## Result files and failure behavior

- `result.json`: only a completed run. Includes UTC bounds, declared target/model
  identities, observed input/model hashes, nullable publisher hashes, frontend seed,
  reference text separately from predictions, frame counts, truncation and timing.
- Preparation: `feats/<utt_id>.npy` and `prepared-manifest.json`. Entries preserve
  input annotations and add freshly computed `feat_length`, `original_frames`,
  `truncated`, relative `feature_file` and its SHA-256. These replace any stale
  derived fields only in the new output manifest, never the input file.
- Inference: utterance records additionally contain `text`, `token_ids`,
  `token_count`, `decoder_executed` and stage `timings_ms`; no NPY feature export.
  `metadata` contains actual bound model metadata. No CER is inferred from a pair
  of transcripts. `frontend_ms` excludes file loading; stage timing is not full
  end-to-end latency.
- `failed.json`: failure after output-directory creation, with current utterance,
  completed earlier records, error type/message and identities collected so far.
  Preflight errors can occur before a directory exists and only print stderr.
  A failure to write the failure record is also reported; there is no false promise
  of evidence on a full/unwritable disk.

`inference_attempted` records pipeline entry. `inference_executed` is false for
preprocessing, true after a successful pipeline result, and null when the first
attempt failed and execution completion is unknown. A successful earlier utterance
keeps it true if a later attempt fails. It is not a board-validation certificate.
Zero-token output explicitly records decoder bypass with decoder timing null.

Input audio, manifest, CMVN, vocabulary and model bytes are rechecked before
successful completion. Existing output directories are rejected; interrupted runs
can contain partial feature files and must not be treated as complete. Use a fresh
output path rather than overwriting evidence. Read `result.json`, not stdout, as
machine-readable output; FunASR may print dependency notices on stdout.

<a id="integration-example"></a>
### Runnable synthetic pipeline example

This example injects three synthetic functions, not models. It demonstrates the
actual CPU orchestration and text decoder using NumPy only. Run from repository
root; expected output is `中中文 3 True`.

```bash
python - <<'PYCODE'
import numpy as np
from samples.speech.paraformer.runtime.python.pipeline import ParaformerPipeline, TensorNames

names = TensorNames("speech", "context", "context", "alphas", "hidden",
                    "context", "count", "bias", "acoustic", "logits")
vocabulary = [f"token{i}" for i in range(8404)]
vocabulary[3] = "中"
vocabulary[4] = "文@@"

def encoder(inputs):
    return {"context": np.zeros((1, 400, 512), np.float32)}

def predictor(inputs):
    weights = np.zeros((1, 401), np.float32)
    weights[0, :3] = 1
    return {"alphas": weights, "hidden": np.ones((1, 401, 512), np.float32)}

def decoder(inputs):
    logits = np.zeros((1, 100, 8404), np.float32)
    logits[0, np.arange(3), [3, 3, 4]] = 1
    return {"logits": logits}

pipeline = ParaformerPipeline(encoder, predictor, decoder, names, vocabulary)
result = pipeline.predict(np.zeros((1, 400, 560), np.float32), 3)
print(result.text, result.token_count, result.decoder_executed)
PYCODE
```

Six pipeline host tests additionally cover model order and named feeds, masking,
repeat preservation/BPE/special-token decoding, empty-result bypass, malformed
frontend and decoder data, vocabulary size and missing predictor output. The
published vocabulary was read from the active manifest URL: 8,404 unique tokens,
SHA-256 `2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127`.
This observed digest does not certify the publisher's currently null manifest
hash or validate any compiled model. See [pipeline evidence](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-pipeline-review.md).

## Publication selection and runtime binding

[model_binding.py](model_binding.py) reads the active S publication manifest.
`resolve_selections("s100")` returns encoder, predictor and decoder in that order;
`auto` uses the shared local-board detector. X5, S100P and S600 are rejected because
there is no corresponding published model set. Selection does not load the SDK,
connect to a board, download a file or claim that the files already exist.

From repository root, this host-safe example prints the three qualified asset IDs
and an explicit rejection for S100P:

```bash
python - <<'PYCODE'
from samples.speech.paraformer.runtime.python.model_binding import resolve_selections
for selection in resolve_selections("s100"):
    print(selection.stage, selection.asset.reference)
try:
    resolve_selections("s100p")
except ValueError as error:
    print(error)
PYCODE
```

Default paths are under `samples/speech/paraformer/model/s100/` and retain published
filenames `paraformer_large_encoder_400x560_s100.hbm`,
`paraformer_large_predictor_400x512_s100.hbm` and
`paraformer_large_decoder_400x512_s100.hbm`. Alternate locations require both
`model_paths={"encoder": ..., "predictor": ..., "decoder": ...}` and
`asset_ids={"encoder": ..., "predictor": ..., "decoder": ...}`. Each ID must exactly
match its stage's selected publication. Partial, mixed or relabelled sets fail.
The active manifest has no publisher SHA-256 for these assets: file hashing records
observed bytes and cannot by itself establish official origin or model compatibility.

Each HBM must expose one model. All physical input/output names, shapes and dtypes
are validated; tensor order is irrelevant. Names follow the fixed S extraction
and native lookup:

| Role | Exact name |
| --- | --- |
| Encoder input | `speech` |
| Encoder output / predictor input / decoder context | `/encoder/after_norm/Add_1_output_0` |
| Predictor weights | `/predictor/Add_output_0` |
| Predictor hidden | `/predictor/Concat_5_output_0` |
| Decoder count / bias | `token_num` / `bias_embed` |
| Decoder acoustic | `onnx::Shape_8609` or the source Python alias `shape_8609` |
| Decoder logits | `logits` |

The optional decoder pass-through `token_num` output must be int32 `[1]` when
exposed. All other unexpected tensors, duplicate/ambiguous names, wrong shapes
and wrong dtypes are rejected. The shapes are those in the pipeline table; only
the count uses int32, all other tensors require float32. A differing compiled
contract must be inspected and explicitly adapted, not cast silently.

[runtime.py](runtime.py) constructs three shared `NamedArrayRunner` instances.
`load_runtime` validates the entire declared model set before creating any SDK
object. On the normal path each runner checks local target identity and its model
file before importing/constructing `hbm_runtime`. It then binds observed metadata.
Only tests use an explicit `runtime_factory` to inject SDK doubles and bypass the
real board/file gates; this seam is not a customer deployment mode.

### Board integration API (board execution not-run)

The following is an integration sketch, **not a complete audio command**. It needs
an S100, matching `hbm_runtime`, three local model files, the exact vocabulary and
prepared frontend features from the frontend below. The complete CLI is documented below. This API example was also run with the real
frontend and explicit SDK doubles; its synthetic text is not an HBM result.

```python
import json
from pathlib import Path
from samples.speech.paraformer.runtime.python.model_binding import resolve_selections
from samples.speech.paraformer.runtime.python.runtime import load_runtime

# Board-only integration: the model package must already be prepared.
import soundfile as sf
from samples.speech.paraformer.runtime.python.frontend import ParaformerFrontend
vocabulary = json.loads(Path("samples/speech/paraformer/model/s100/tokens.json").read_text())
bundle = load_runtime(resolve_selections("s100"), vocabulary)
bundle.set_scheduling_params(priority=7, bpu_cores=[0])
sample = Path("samples/speech/paraformer")
waveform, rate = sf.read(sample / "test_data/audio/BAC009S0724W0121.wav", dtype="float32")
prepared = ParaformerFrontend(sample / "model/am.mvn").pre_process(waveform, rate)
result = bundle.pipeline.predict(prepared.tensor, prepared.valid_frames)
print(result.text, prepared.truncated)
```

Scheduling is optional. If specified, `priority` is an integer 0–255 and `bpu_cores`
is a nonempty sequence of nonnegative integer indexes; supported hardware/core
combinations remain the SDK's responsibility. Parameters are delegated to **all
three models** through model-keyed SDK dictionaries. Missing setters are rejected
before any setter is called. Invalid values are rejected by the shared runner;
an SDK failure is propagated and there is no rollback guarantee if an earlier
model already accepted its setting. With both arguments omitted, no SDK setter
is invoked. This corrects the source wrapper's silently ignored parameters.

The suite now includes eight binding/adapter checks (33 total, including package, waveform and CLI checks): publication and
path identity, metadata order/shape/type/names, the documented acoustic alias and
optional count output, pre-SDK rejection, and an actual shared-runner pipeline
with SDK doubles and scheduling propagation. [Binding evidence](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-binding-review.md)
distinguishes those host checks from real SDK/board execution, still not-run.

## Real CPU audio frontend

[frontend.py](frontend.py) uses the actual FunASR `WavFrontend`; it is not a NumPy
replacement for fbank. Its input is an already loaded finite float32 array with
shape `[samples]` or `[samples,channels]`, a nonempty sample/channel dimension and
integer sample rate 16000. Audio file loading stays outside the numerical class.
Stereo/multichannel samples are averaged in float32, without additional waveform
normalization. Other sample rates are rejected, not automatically resampled.

The fixed settings are hamming window, 80 mel bins, 25 ms frame length, 10 ms shift,
LFR stack 7/step 6 and the byte-pinned `am.mvn`. FunASR's source defaults retain
its sample scaling and dither. For audio shorter than 25 ms, FunASR shortens its
analysis window; the 10 ms test case is verified against the source. Extremely
short input can still fail the upstream feature algorithm; such errors propagate.
No VAD, punctuation restoration, timestamps, streaming or hotword bias is added.

`ParaformerFrontend(cmvn_path, random_seed=191009)` verifies the fixed CMVN before
loading dependencies. `pre_process(waveform, sample_rate)` returns `PreparedFeatures`:

| Field | Meaning |
| --- | --- |
| `tensor` | owned contiguous float32 `[1,400,560]`, padded rows zero |
| `valid_frames` | frames supplied to encoder/CIF, maximum 400 |
| `original_frames` | frontend frame count before truncation |
| `truncated` | true when original frame count exceeds 400 |
| `sample_count` | mono sample count before feature extraction |

Pass `prepared.tensor` and `prepared.valid_frames` to `bundle.pipeline.predict`.
The source's first-400 behavior is preserved and made explicit by the other fields;
this is **not** long-audio chunking. A 30-second fixture produces 500 LFR frames,
uses the first 400 and reports truncation. The caller must surface that flag instead
of presenting the result as a full-length transcript.

The default seed matches the source's fbank dither. CPU RNG changes are scoped and
restored even if the frontend raises. GPU RNG is not reseeded. Calls through this
adapter serialize around the CPU RNG context; unrelated threads that also use
Torch's global RNG require application coordination. This is not a claim of
process-wide concurrency isolation. `random_seed` accepts an integer in `[0,2**63)`;
changing it changes dither and is outside the fixed-seed source comparison.

<a id="stage-io"></a>
## Three-model application pipeline

[pipeline.py](pipeline.py) composes three independent raw model callables. Each
callable consumes a mapping of physical input names to arrays and returns a
mapping of physical output names to arrays. `TensorNames` supplies the exact
names discovered by the model binder; there is no first-output or
substring fallback in the pipeline. This interface is for application composition,
not a model `forward` containing CPU processing between several SDK executions.

| Stage | Required input | Required output |
| --- | --- | --- |
| Encoder | float32 `[1,400,560]` | float32 context `[1,400,512]` |
| Predictor | context `[1,400,512]` | float32 weights `[1,401]`, hidden `[1,401,512]` |
| CPU CIF | predictor arrays + valid frame count | float32 acoustic `[1,100,512]`, int32 count `[1]` |
| Decoder | context, acoustic, count, zero float32 bias `[1,1,512]` | float32 logits `[1,100,8404]` |

`predict(features, feature_length)` requires finite float32 prepared features and
an integer valid length 1–400. Shape/dtype/finiteness are checked at every consumed
boundary. Intermediate values are copied so later runner buffers cannot overwrite
retained encoder context. The pipeline does not load an SDK or change scheduling;
physical-model validation and scheduling are provided by the runtime adapter below.
INT16 in a compiled artifact's name does not prove INT16 physical I/O.

`Prediction` includes text, selected token IDs, CIF count, per-stage milliseconds,
and `decoder_executed`. Text follows the S source: argmax on the valid prefix,
remove tokens enclosed by `<...>`, strip `@@`, concatenate without a separator.
Repeated tokens remain repeated: this is not CTC. The token count and ID list
include subsequently removed special tokens. A zero CIF count bypasses the decoder
and returns empty text, empty IDs, `decoder_executed=False` and decoder timing
`None`. A stage error propagates and produces no successful prediction.

Timings measure the three runner calls and CPU CIF separately. They exclude
frontend, loading, validation/copying outside the calls, text decoding and file I/O;
their sum is not end-to-end latency. No SDK or board performance is measured here.


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

Seven CIF host tests cover empty output, manually derived fractional crossings,
padding versus calibration mode, the 100-token cap, 24 source comparisons,
input ownership and invalid contracts. The comparison loads the archived source
from S commit `380e1a2bf42041af54be6f34935e50197cfadff9`; its no-fire case raises
`IndexError`, which the unified bridge fixes. See the
[review and evidence](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-cif-review.md).

The source publishes S100 models only. No X5, S100P or S600 adaptation is claimed.
Actual SDK verification and board three-stage inference remain open. The
[host evaluator](../../evaluator/README.md) now provides FP32/HMCT entry points and explicit CER reporting. [Conversion tools](../../conversion/README.md) now provide verified
FP32 export and real-audio calibration plus explicit OE orchestration. Native application host verification
is recorded separately in the C++ guide.
Board inference, OE compilation, dataset CER and latency have not been run.

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Cause / action |
| --- | --- |
| Output directory must be new | Choose a fresh output path; preserve prior records |
| Missing selected WAV | Fix the manifest/audio directory; no implicit skip |
| Paraformer requires 16000 Hz | Prepare matching audio explicitly; CLI does not resample |
| Target mismatch / missing board identity | Use preprocess-only on a host; actual inference requires S100 |
| Vocabulary or CMVN differs from pin | Use the fixed package; do not rename another model's files |
| Tensor name/type/shape mismatch | Inspect actual model metadata and publication identity |

The real host CLI was checked for help/list/preview, default two-WAV preparation,
single WAV, prefix limit, output reuse, invalid rate and host inference refusal.
The full CLI inference branch is exercised only with explicit SDK doubles, using
the actual shared runners and pipeline; that fixture is not an HBM result. See
[CLI evidence](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-cli-review.md).
