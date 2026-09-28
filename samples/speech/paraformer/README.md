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

Upstream toolkit: [FunASR](https://github.com/modelscope/FunASR). This migration is
based on S commit `380e1a2bf42041af54be6f34935e50197cfadff9`, not whatever upstream
main currently provides. The sample lives at `samples/speech/paraformer`.
Python and native C++ entries and real CPU preprocessing are implemented;
real-weight three-stage FP32 export is also implemented. Calibration, OE compilation
and evaluator migration remain open. This is not full sample acceptance.

<a id="support-matrix"></a>
## Support matrix

| Published deployment | x5 | s100 | s100p | s600 | Python | Unified C++ |
| --- | --- | --- | --- | --- | --- | --- |
| large, encoder 400×560 / predictor 400×512 / decoder 100×8404 | not-supported | supported-not-run | not-supported | not-supported | implemented; host-checked | implemented; host application checked, SDK/board not-run |

`supported-not-run` denotes the declared S100 deployment and available Python
implementation, not a new board result. Real SDK/model metadata and board inference
have not been exercised here. The [native guide](runtime/cpp/README.md#quickstart) provides the full prepared-feature
launcher/build/run flow and result reports. Host transport tests do not establish
real dual-language SDK/model equivalence.

<a id="prerequisites"></a>
## Prerequisites

For host preprocessing: Python 3.12 was verified with Torch/torchaudio 2.6.0,
FunASR 1.3.14, NumPy 1.26.4, SoundFile 0.14.0 and protobuf 4.23.0. Install the
[frontend requirements](runtime/python/requirements-frontend.txt) explicitly in a
separate environment as described in the [runtime guide](runtime/python/README.md#environment).
Use that interpreter as `python` in the commands below. Help/list/dry-run need only
the core Python, NumPy and PyYAML environment, not Torch/FunASR or a board SDK.

For inference: an actual S100 with compatible board-provided `hbm_runtime`, the
three HBM files and the fixed vocabulary. A precise image/SDK build is not
established by the archived sample or this host work; no invented minimum version
is asserted. The host frontend environment alone does not provide that SDK.
No conversion toolchain is needed to use already published artifacts.

Keep space for the complete model package and outputs. Prepared features require
896,128 bytes per default `.npy` file (400×560 float32 plus header), besides reports.
The frontend reads the entire WAV before truncating features and inference loads
all three models; maximum RAM/board capacity has not been measured in this work.

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

On S100, explicitly prepare the six-file package and then run inference. These
commands are provided for the actual board environment and were not board-tested:

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100
python samples/speech/paraformer/runtime/python/main.py --target s100 --output-dir outputs/paraformer-inference
```

The downloader prints paths and observed hashes and never overwrites existing
files. HBM publisher hashes are absent from the manifest; local hashes bind bytes
without independently authenticating origin. Inference validates local board
identity, declared assets and physical tensor contracts. It never auto-downloads.
The optional `runtime/python/run.sh` wrapper forwards the same flags and uses
`PYTHON` if set; it does not install dependencies.

<a id="expected-results"></a>
## Expected results and limits

Both bundled inputs yield float32 `[1,400,560]`, with 71 and 78 valid frames and
zero padding. Seven real frontend comparisons are byte-identical to the pinned
source. These are features, not a newly measured speech-recognition accuracy.
A 30-second test produces 500 LFR frames; only the first 400 are used, with an
explicit `truncated` flag. That behavior is not whole-recording chunked transcription.

Inference `result.json` contains model/input hashes, bound metadata, per-utterance
text/token IDs, counts and timings, and reference annotations separately. No
expected transcript, CER or model latency is invented for unrun HBM inference.
Empty CIF output returns empty text and marks the decoder as bypassed. Failures
after output creation leave `failed.json` with partial records when writable;
missing input, incompatible files and target mismatch are errors, not skipped cases.

[Host evidence](../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-cli-review.md)
covers the real preprocessing CLI and explicitly distinguishes synthetic SDK tests
from actual inference. Source historical evaluator results remain historical until
the evaluator is migrated and labelled separately.

<a id="directory"></a>
## Directory layout

```text
paraformer/
├── model/           # explicit six-file preparation, CMVN/config and model guide
├── runtime/python/  # CLI/I/O, real frontend, three raw runners, CPU CIF and text
├── runtime/cpp/     # native application, SDK adapter and prepared-feature input
├── conversion/      # real-weight FP32 export and graph tools; calibration/OE pending
├── test_data/       # unchanged source WAVs and reference manifest
├── tests/           # host behavior and SDK-boundary tests
└── README.md        # overview, complete commands and validation boundaries
```

<a id="entry-points"></a>
## Entry points

- [Model package](model/README.md): all six files, identities, download and rerun behavior.
- [Python runtime](runtime/python/README.md#usage): default/custom commands, every parameter,
  result fields, stage interfaces, complete CPU examples and failure handling.
- [Test data](test_data/README.md): input provenance and reference text.
- [C++ runtime](runtime/cpp/README.md): full native application and launcher are host-checked with explicit transport doubles; SDK/board inference is unverified.
- [Conversion](conversion/README.md): strict local weight loading, real three-stage FP32 export, numeric checks and report; calibration/OE remain pending.
- Evaluator: unified implementation/documentation pending; do not infer dataset metrics
  from the two bundled smoke inputs.

<a id="license"></a>
## License

Sample code follows the repository [Apache-2.0 license](../../../LICENSE), except
FunASR-derived export composition covered by its [MIT notice](conversion/LICENSE-FunASR).
Upstream models and data have their own terms; the repository code license does
not establish a weight/data license. The active binary manifest does not record
per-artifact licensing, and this migration does not supply a new rights claim.
