# Paraformer speech recognition

[简体中文](README_cn.md)

<a id="overview"></a>
## Overview

Paraformer converts 16 kHz speech into text using a FunASR frontend, an encoder,
a predictor, CPU continuous integrate-and-fire (CIF), and a decoder. The deployment
uses three separately published S100 HBM files and an ordered 8,404-token vocabulary.
CPU CIF is explicit between predictor and decoder; text decoding preserves repeated
tokens and removes source special/BPE markers. There is no VAD, streaming,
punctuation restoration, timestamp output or custom hotword support in this sample.

Upstream toolkit: [FunASR](https://github.com/modelscope/FunASR); the shipped
weights are the published S release's Paraformer-large models (see
[conversion](conversion/README.md) for the exact sources). The sample lives at
`samples/speech/paraformer`, providing a Python CLI, a native C++ application,
real CPU preprocessing, three-stage FP32 export, real-audio calibration,
explicit OE compilation orchestration and a host evaluator. OE/HMCT
compilation runs in the OE environment per the conversion guide.

<a id="directory"></a>
## Directory structure

```text
paraformer/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="support-matrix"></a>
## Support matrix

| Published deployment | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| large, encoder 400×560 / predictor 400×512 / decoder 100×8404 | not-supported | supported | not-supported | not-supported | implemented | implemented (board run via [C++ guide](runtime/cpp/README.md)) |

`supported` denotes the declared S100 deployment with its published artifacts
and available implementations. The [native guide](runtime/cpp/README.md#quickstart)
provides the full prepared-feature launcher/build/run flow and result reports.

<a id="prerequisites"></a>
## Prerequisites

For host preprocessing: Python 3.12 was verified with Torch/torchaudio 2.6.0,
FunASR 1.3.14, NumPy 1.26.4, SoundFile 0.14.0 and protobuf 4.23.0. Install the
[frontend requirements](runtime/python/requirements-frontend.txt) explicitly in a
separate environment as described in the [runtime guide](runtime/python/README.md#environment).
Use that interpreter as `python` in the commands below. Help/list/dry-run need only
the core Python, NumPy and PyYAML environment, not Torch/FunASR or a board SDK.

For inference: an actual S100 with compatible board-provided `hbm_runtime`, the
three HBM files and the fixed vocabulary. The board image/SDK build follows the
artifact's release notes; the host frontend environment alone does not provide
that SDK.
No conversion toolchain is needed to use already published artifacts.

Keep space for the complete model package and outputs. Prepared features require
896,128 bytes per default `.npy` file (400×560 float32 plus header), besides reports.
The frontend reads the entire WAV before truncating features and inference loads
all three models; size the board for the three resident models plus the features.

<a id="quickstart"></a>
## Quick start

All commands use the repository root as cwd. First inspect the model package;
preview does not download, load SDKs or write files:

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100 --dry-run
python samples/speech/paraformer/runtime/python/main.py --target s100 --dry-run
```

Without a board, execute the actual frontend on the two bundled WAVs. The CMVN and
input audio are already in the repository; HBM and vocabulary downloads are not
needed for this mode:

```bash
python samples/speech/paraformer/runtime/python/main.py --preprocess-only --output-dir outputs/paraformer-prepared
```

Success: exit code 0, `outputs/paraformer-prepared/result.json` with status
`completed`, two feature files under `feats/`, and `prepared-manifest.json` with
`feat_length` 71 and 78. The source manifest remains unchanged. Use a new output
directory for every run; existing directories are rejected.

On S100, explicitly prepare the six-file package and then run inference:

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100
python samples/speech/paraformer/runtime/python/main.py --target s100 --output-dir outputs/paraformer-inference
```

The downloader prints paths and observed hashes and never overwrites existing
files. HBM publisher hashes are absent from the manifest; local hashes bind bytes
without independently authenticating origin. Inference validates local board
identity, declared assets and physical tensor contracts. It never auto-downloads.
The optional `runtime/python/run.sh` helper forwards the same flags and uses
`PYTHON` if set; it does not install dependencies.

<a id="expected-results"></a>
## Expected results and limits

Both bundled inputs yield float32 `[1,400,560]`, with 71 and 78 valid frames and
zero padding. Seven real frontend cases form the frontend comparison set.
A 30-second test produces 500 LFR frames; only the first 400 are used, with an
explicit `truncated` flag. That behavior is not whole-recording chunked transcription.

Inference `result.json` contains model/input hashes, bound metadata, per-utterance
text/token IDs, counts and timings, and reference annotations separately. Expected
transcripts, CER and model latency for the published HBMs come from the board run
and the [evaluator](evaluator/README.md).
Empty CIF output returns empty text and marks the decoder as bypassed. Failures
after output creation leave `failed.json` with partial records when writable;
missing input, incompatible files and target mismatch are errors, not skipped cases.

The [evaluator](evaluator/README.md) covers the CER definition, run records and
the historical dataset metrics.

<a id="entry-points"></a>
## Entry points

- [Model package](model/README.md): all six files, identities, download and rerun behavior.
- [Python runtime](runtime/python/README.md#usage): default/custom commands, every parameter,
  result fields, stage interfaces, complete CPU examples and failure handling.
- [Test data](test_data/README.md): input provenance and reference text.
- [C++ runtime](runtime/cpp/README.md): full native application and launcher; board run via the C++ quickstart.
- [Conversion](conversion/README.md): strict local weight loading, real three-stage FP32 export, numeric checks, real-audio calibration and explicit OE orchestration commands.
- [Evaluator](evaluator/README.md): feature preparation, FP32 and HMCT commands, CER definition,
  failure records and historical metrics.

<a id="license"></a>
## License

Sample code follows the repository [Apache-2.0 license](../../../LICENSE), except
FunASR-derived export composition covered by its [MIT notice](conversion/LICENSE-FunASR).
Upstream models and data have their own terms; the repository code license does
not establish a weight/data license. The active binary manifest does not record
per-artifact licensing; no additional rights are granted here.
