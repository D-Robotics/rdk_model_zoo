# Paraformer evaluator

Evaluate prepared 16 kHz speech features with the three FP32 ONNX stages on CPU,
or with OE's HMCT executor and three quantized `*_ptq_model.onnx` graphs. This is
host simulation, not HBM execution or a board benchmark. It reuses the runtime's
encoder → predictor → CPU CIF → decoder pipeline and greedy token decoding.
No dataset, models or toolchain are downloaded by this entry.

<a id="dataset"></a>
## Dataset

The repository includes two WAVs with reference text for smoke checks, not a full
AISHELL benchmark. For dataset evaluation, obtain the dataset under its applicable
license and construct a manifest with one `utt_id`/`text` pair per WAV; the matching
audio directory must contain `<utt_id>.wav`. Use the preparation command below to
produce validated feature files and retain its truncation report. No dataset split
or text normalization is selected implicitly.

<a id="environment"></a>
## Environment and preparation

Run all commands from the repository root. The tested FP32 environment is Python
3.12, NumPy 1.26.4, ONNX 1.17.0 and ONNX Runtime 1.20.1. Install the complete
[export/frontend requirements](../conversion/requirements-export.txt) in a dedicated
environment when also preparing audio. HMCT is supplied by the matching vendor OE
environment; it is not a substitute package to install from an arbitrary index.
Actual HMCT evaluation has not been run in this migration.

First follow [conversion](../conversion/README.md) to export the three self-contained
ONNX graphs into `outputs/paraformer_export`. Obtain the pinned `tokens.json` through
[model preparation](../model/README.md); its order must match the 8404-class decoder.
Prepare the bundled two utterances using the real frontend without a board:

```bash
python samples/speech/paraformer/runtime/python/main.py \
  --target s100 --preprocess-only \
  --manifest samples/speech/paraformer/test_data/manifest.json \
  --audio-dir samples/speech/paraformer/test_data/audio \
  --output-dir outputs/paraformer-features
```

Use a new output directory for every run. For a dataset, supply your own WAV
manifest and audio directory to the same preparation command. Input audio must be
16 kHz; the frontend does not silently resample. More than 400 LFR frames are
truncated with metadata, so a full-length reference may penalize a truncated utterance.
Inspect this before treating the result as a dataset benchmark.

<a id="command"></a>
## FP32 evaluation

```bash
python samples/speech/paraformer/evaluator/main.py \
  --pipeline fp32 \
  --encoder outputs/paraformer_export/encoder.onnx \
  --predictor outputs/paraformer_export/predictor.onnx \
  --decoder outputs/paraformer_export/decoder.onnx \
  --manifest outputs/paraformer-features/prepared-manifest.json \
  --vocab samples/speech/paraformer/model/s100/tokens.json \
  --output-dir outputs/paraformer-eval-fp32
```

Success returns 0 and writes `evaluation.json` with `status: completed`, all selected
utterances and aggregate metrics. A valid evaluation can still have poor recognition
accuracy; there is no built-in CER pass threshold. Execution/validation failure returns
2 and retains a failed report in the newly created output directory. Existing output
is rejected without modification.

## Quantized simulation

After successful OE compilation, locate each stage's actual `*_ptq_model.onnx`
artifact in the compiler output. In the corresponding OE environment run:

```bash
python samples/speech/paraformer/evaluator/main.py \
  --pipeline int16 \
  --encoder /absolute/path/to/paraformer_encoder_int16_ptq_model.onnx \
  --predictor /absolute/path/to/predictor_int16_ptq_model.onnx \
  --decoder /absolute/path/to/decoder_int16_ptq_model.onnx \
  --manifest outputs/paraformer-features/prepared-manifest.json \
  --vocab samples/speech/paraformer/model/s100/tokens.json \
  --output-dir outputs/paraformer-eval-int16
```

These three model paths are placeholders to replace with your actual artifacts.
The adapter preserves the source `ORTExecutor(path).create_session().forward(feed)`
API. The `int16` selector chooses that executor; it does not prove quantization
precision or certify compiler output. HBM files are not accepted by this evaluator.

## Arguments and input contract

| Option | Default / meaning |
|---|---|
| `--pipeline` | Required: `fp32` CPU ORT or `int16` HMCT simulation |
| `--encoder`, `--predictor`, `--decoder` | Required self-contained ONNX paths; aliases `--enc`, `--pred`, `--dec` |
| `--manifest` | Required prepared or legacy JSON list |
| `--vocab` | Required published vocabulary; SHA-256 and 8404 ordered tokens checked |
| `--output-dir` | Required new directory |
| `--max-utts` | `0`: all; positive integer selects a prefix after full manifest validation |
| `--threads` | `4`: FP32 ORT intra-op threads; inter-op is 1; no HMCT thread override |

Each manifest entry requires a unique filename-stem `utt_id`, a string `text`
(empty references are allowed), and integer `feat_length` in 1–400. Prepared entries
include `feature_file` relative to the manifest directory or absolute, optional
`feature_sha256`, and optional consistent `original_frames`/`truncated` metadata.
Legacy entries without `feature_file` resolve to `feats/<utt_id>.npy` next to the
manifest. Unknown annotations are retained in results. Every selected feature must
be one finite float32 NPY array `[1,400,560]`; missing files, non-finite values,
incorrect types, hash mismatches, NPZ and trailing bytes fail explicitly.

The entire manifest is validated before prefix selection. Feature files are opened
only for selected entries; each digest covers the bytes actually loaded. Stage I/O
uses exact semantic-name binding, shapes and dtypes, including the documented acoustic
alias and optional decoder `token_num` output. No positional fallback or implicit cast.

<a id="metrics"></a>
## Metric definition

CER is total Unicode character edit distance divided by total reference characters;
`cer` is a ratio, not a percentage. Substitution/deletion/insertion counts and per-item
errors are retained. No whitespace, case or punctuation normalization occurs. If every
reference is empty, CER is null, while insertion counts remain meaningful. Decoding
removes `<...>` tokens and `@@` markers, concatenates tokens and does not perform CTC
repeat collapse. The old evaluation script did not strip `@@`; this entry follows the
unified runtime. Zero CIF tokens produce empty text and skip decoder execution.
Stage timings exclude frontend, model loading and file I/O; they are neither BPU nor
end-to-end latency.

<a id="outputs"></a>
## Output records

`evaluation.json` records UTC start/end, executor/version facts, model/manifest/vocabulary
paths and hashes, tensor interfaces, selected count and per-utterance source metadata,
feature digest, reference, hypothesis, token IDs/count and stage timings. Model,
manifest and vocabulary hashes are checked again before successful completion.
A failure preserves completed utterances and `current_utterance`, but `metrics` stays
null; partial results are not presented as a complete evaluation.

<a id="reference-results"></a>
## Reference results and limits

The archived S conversion guide (historical `../../../../platforms/s/samples/speech/paraformer/conversion/README_cn.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md)
records AISHELL dev 300-utterance CER: FP32 5.20%, HMCT INT16 5.02%, S100 Python
3.13%, S100 C++ 3.13%. These are source historical measurements, not reproduced
migration results or proof that one backend is more accurate.

The current host FP32 run on the two bundled utterances measured 4 edits / 28 reference
characters, CER 14.2857%. This small smoke run is not a reproduction of that 300-item
benchmark. See the [evaluation review](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-evaluator-review.md)
for source comparison and exact evidence.

<a id="boundaries"></a>
## Validation boundaries

Real HMCT, OE compilation, HBM execution,
board latency and full-dataset accuracy remain unverified. The conversion guide also
discloses the random-input Torch/ORT discrepancy; this two-utterance run does not
establish arbitrary-input equivalence.

## Troubleshooting and code layout

Missing HMCT means the matching OE environment is absent; do not substitute CPU ORT
for graphs containing vendor quantization operators. Vocabulary mismatch requires the
published token order, not renaming a JSON file. Invalid frame metadata requires
regenerating features from the corresponding audio. Existing output requires choosing
a new directory. An interrupted process may leave `status: running`; treat it as incomplete.

`inputs.py` owns manifest/NPY validation, `backends.py` owns model execution adapters,
and `main.py` owns orchestration and report files. Runtime `pipeline.py`, `cif.py` and
`decoding.py` supply inference math; shared `text_metrics.py` supplies CER. File and
metric helpers are kept outside model inference code.
