# ASR — chunked speech recognition

English | [简体中文](README_cn.md)

<a id="overview"></a>
## Overview

This sample transcribes audio using the published S-series Wav2Vec2 ASR model and a fixed 3503-token vocabulary. It reads WAV/FLAC in bounded chunks, mixes channels to mono, resamples to 16 kHz, normalizes each chunk and submits 30000 samples per inference (1.875 seconds). It handles the whole file, including the final padded chunk. This is independent-window processing, not an acoustic model with hidden streaming state or overlap stitching.

The canonical Python workflow is implemented. Native migration is in progress: portable CTC/normalization code and host tests exist, while native audio/SDK/executable integration remains pending. Preserve the original [S source](../../../platforms/s/samples/speech/asr/README.md) for historical implementation context. No board test or new real model transcription has run in this migration.

<a id="support-matrix"></a>
## Support matrix

| Target | Publication | Canonical Python | Canonical C++ | Board validation |
| --- | --- | --- | --- | --- |
| S100 | `s:asr:s100/asr.hbm` | Implemented | Core only; integration pending | not-run |
| S600 | `s:asr:s600/asr.hbm` | Implemented | Core only; integration pending | not-run |
| X5 / S100P | None | Rejected | Unsupported | not-run |

Published targets follow the active manifest, not contradictory source comments. S600's publication does not certify runtime behavior: its actual model/SDK metadata still needs validation. No target fallback is provided.

<a id="prerequisites"></a>
## Prerequisites

Python 3.10+, NumPy, PyYAML, SoundFile and SciPy; the board needs the matching BSP `hbm_runtime`. Help/list/dry-run do not import the board SDK, SoundFile or SciPy. Do not install an unrelated PyPI `hbm_runtime`. Model downloads and dependency setup are explicit. See the [runtime environment](runtime/python/README.md).

<a id="quickstart"></a>
## Quick start

From the repository root, inspect available targets on any host:

```bash
bash samples/speech/asr/runtime/python/run.sh --list-models
bash samples/speech/asr/runtime/python/run.sh --target s100 --dry-run
bash samples/speech/asr/runtime/python/run.sh --target s600 --dry-run
```

On the corresponding board, prepare its model and run (these inference commands are not current board evidence):

```sh
bash samples/speech/asr/model/download.sh --target s100
bash samples/speech/asr/runtime/python/run.sh --target s100 --output-dir outputs/asr-run1
```

Use `s600` in both commands for an S600; never rename the S100 HBM. `PYTHON` selects the interpreter. Output directories must be new. External models require the exact target-qualified asset ID; see [model preparation](model/README.md).

<a id="expected-results"></a>
## Expected results and decoding change

The console prints a full-file transcription and the path to `result.json`. The report binds model/audio/vocabulary hashes, real metadata, decoder mode and each chunk's source position, valid sample count and text. Errors return 2; a failure after processing starts saves `failed.json` with completed chunks, not a partial success transcript.

Default `ctc` collapses adjacent duplicate token IDs **before** removing blank ID 0. The archived implementation concatenated repeated IDs and only removed `<pad>`. `--decode-mode legacy` retains that old behavior for source comparisons. IDs `[1,1,0,1,2,2]` with tokens `<pad>,a,b` yield `aab` in CTC and `aaabb` in legacy. This is an explicit decoder correction, not evidence of changed model logits. Other tokens, punctuation and `|` remain verbatim. CTC state resets for every independent chunk; no cross-chunk deduplication is invented.

<a id="directory"></a>
## Directory

| Location | Responsibility |
| --- | --- |
| `model/` | Exact manifest selection and explicit target download |
| `runtime/python/` | Audio reader, pure frontend/decoder, shared raw runner and CLI |
| `runtime/cpp/` | Portable numeric contract and host tests; full native integration pending |
| `test_data/` | Original WAV, fixed vocabulary and historical figures |
| `conversion/` | Honest export/compiler prerequisites missing from the source |
| `evaluator/` | Saved-transcript character error metrics and historical results |
| `tests/` | Host numeric/metadata/identity/report tests |

<a id="entry-points"></a>
## Guides

[Python usage/API](runtime/python/README.md) · [Native progress](runtime/cpp/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md) · [Input identities](test_data/README.md).

Host source comparisons cover original 16 kHz speech, 44.1 kHz stereo and 8 kHz constant audio. Their feature agreement is separate from SDK/board/dataset acceptance. Original latency and cosine screenshots remain historical, not new measurements.

<a id="license"></a>
## License

The sample follows the repository [Apache-2.0 license](../../../LICENSE). Source copyrights, data and screenshots are retained from the pinned S branch; external dependencies retain their own licenses.
