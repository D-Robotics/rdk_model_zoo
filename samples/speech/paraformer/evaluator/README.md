English | [简体中文](README_cn.md)

# Paraformer evaluator

Evaluate prepared 16 kHz speech features with the three FP32 ONNX stages on CPU,
or with OE's HMCT executor and three quantized `*_ptq_model.onnx` graphs. It reuses
the runtime's encoder → predictor → CPU CIF → decoder pipeline and greedy token
decoding. No dataset, models or toolchain are downloaded by this entry. For HBM
board execution, use the [runtime guide](../runtime/python/README.md).

<a id="dataset"></a>
## Dataset

The repository includes two WAVs with reference text for smoke checks, not a full
AISHELL benchmark. For dataset evaluation, obtain the dataset under its applicable
license and construct a manifest with one `utt_id`/`text` pair per WAV; the matching
audio directory must contain `<utt_id>.wav`. Use the preparation command below to
produce validated feature files and retain its truncation report. No dataset split
or text normalization is selected implicitly.

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── backends.py  # Python script
├── inputs.py  # Python script
└── main.py  # Command-line entry
```

<a id="environment"></a>
## Environment and preparation

Run all commands from the repository root. The tested FP32 environment is Python
3.12, NumPy 1.26.4, ONNX 1.17.0 and ONNX Runtime 1.20.1. Install the complete
[export/frontend requirements](../conversion/requirements-export.txt) in a dedicated
environment when also preparing audio. HMCT is supplied by the matching vendor OE
environment; it is not a substitute package to install from an arbitrary index.
Run actual HMCT evaluation in the matching OE environment.

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
The adapter preserves the source `ORTExecutor(path).create_session.forward(feed)`
API. The `int16` selector chooses that executor; read the actual quantization
precision and compiler output from the model metadata. HBM files are not
accepted by this evaluator.

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
repeat collapse. Zero CIF tokens produce empty text and skip decoder execution.
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
## Source results

The S source conversion record reports AISHELL dev (speech_asr_aishell_devsets),
300 utterances from 40 speakers, using official reference transcripts. The pipeline
uses fbank+LFR, three HBM stages with CPU CIF, and corresponding ONNX stages for
HMCT simulation:

| S source pipeline | CER | Difference from FP32 |
| --- | ---: | ---: |
| FP32 ONNX baseline | 5.20% | — |
| HMCT INT16 simulation | 5.02% | -0.18 percentage points |
| S100 INT16 Python hbm_runtime | 3.13% | -2.07 percentage points |
| S100 INT16 C++ UCP | 3.13% | -2.07 percentage points |

The same source record reports S100 stage times in ms/utterance: Encoder
33.63/33.15, Predictor 1.44/1.00, CPU CIF 3.41/0.38, Decoder 7.12/6.29
(Python/C++ UCP). The Python HBM pipeline totals 45.61 ms/utterance with
RTF ~0.008, excluding WAV preprocessing. The C++ pipeline totals 40.81 ms;
300 utterances took 13.4 s wall-clock (RTF ~0.007), with one model load of
about 1.85 s.

For the two bundled utterances, the evaluator records 4 edits across 28
reference characters (CER 14.2857%).


## Additional reference measurements

Each table retains its published model, board and measurement conditions. Measurements from different configurations are separate reference sets.

### S100 Python and C++ pipeline latency

| 阶段 | Python | **C++ UCP** | 加速 |
|---|---|---|---|
| Encoder (BPU) | 33.63 | **33.15** | ≈ |
| Predictor (BPU) | 1.44 | **1.00** | ≈ |
| CIF (CPU) | 3.41 | **0.38** | **9x** |
| Decoder (BPU) | 7.12 | **6.29** | ≈ |
| **端到端** | 45.61 | **40.81 ms/utt** | **1.12x** |
| **CER** | 3.13% | **3.13%** | 一致 |

**wall-clock**：300 条 13.4 秒（44.7 ms/utt）；一次模型加载 ~1.85 s

C++ 版本主要优势：CPU 侧 CIF 快 9x（numpy 开销比手写循环大），BPU 部分相同（都通过同一底层库）。


### Per-model BPU latency and full-pipeline comparison

Source: [S100 INT16 conversion and performance report](https://github.com/D-Robotics/rdk_model_zoo/blob/d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d/platforms/s/samples/speech/paraformer/conversion/README_cn.md). The model sizes, compiler estimates and board measurements below describe the same recipe; their timing scopes are stated separately.

Board conditions: S100, Ubuntu 22.04 aarch64 and one BPU core. `hrt_model_exec perf` uses one thread, 200 frames for encoder/decoder and 500 frames for predictor. The table compares single-frame measurements with compiler estimates; concurrent queued tasks do not reduce single-frame latency.

```bash
# cwd: repository root on S100; matching model package prepared
hrt_model_exec perf --model_file samples/speech/paraformer/model/s100/paraformer_large_encoder_400x560_s100.hbm --thread_num 1 --frame_count 200
hrt_model_exec perf --model_file samples/speech/paraformer/model/s100/paraformer_large_predictor_400x512_s100.hbm --thread_num 1 --frame_count 500
hrt_model_exec perf --model_file samples/speech/paraformer/model/s100/paraformer_large_decoder_400x512_s100.hbm --thread_num 1 --frame_count 200
```

| Module | Board perf latency | Compiler static estimate | FPS |
|---|---|---|---|
| Encoder INT16 | **33.11 ms** | 32.52 ms | 30.18 |
| Predictor INT16 | **0.67 ms** | 0.35 ms | 1462 |
| Decoder INT16 | **6.12 ms** | 5.77 ms | 162.8 |

The summary uses 300 AISHELL dev utterances from 40 speakers. Resident Python and C++ UCP reuse all three loaded models. Pipeline latency includes encoder, predictor, CPU CIF and decoder, excluding WAV preprocessing. Python wall-clock is 14.0 s (46.7 ms/utterance, including NumPy I/O); C++ wall-clock is 13.4 s (44.7 ms/utterance), with one model load of about 1.85 s. `~41 ms` is the approximate sum of the three BPU perf latencies, excluding CPU CIF. `~289 MB` is the approximate combined HBM file size, not peak process memory.

| Module | Quantization | HBM size | Board perf per frame | Resident Python | Resident C++ UCP |
|---|---|---|---|---|---|
| Encoder | INT16 all | 211.5 MB | 33.11 ms | 33.63 ms | 33.15 ms |
| Predictor | INT16 all | ~4 MB | 0.67 ms | 1.44 ms | 1.00 ms |
| CIF | CPU numpy | — | — | 3.41 ms | 0.38 ms |
| Decoder | INT16 all | 73.5 MB | 6.12 ms | 7.12 ms | 6.29 ms |
| **Pipeline total** | — | **~289 MB** | ~41 ms | **45.61 ms** | **40.81 ms** |
| **CER** | — | — | — | **3.13%** | **3.13%** |

<a id="boundaries"></a>
## Evaluation workflow

Use the [conversion guide](../conversion/README.md) to prepare FP32/PTQ stages
and this evaluator to score a manifest against its reference transcripts. For
board inference and S100 pipeline timing, use the [runtime guide](../runtime/python/README.md).

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
