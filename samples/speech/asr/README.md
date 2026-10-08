English | [简体中文](README_cn.md)

# ASR — chunked speech recognition

<a id="overview"></a>
## Overview

This sample transcribes audio using the published S-series Wav2Vec2 ASR model and a fixed 3503-token vocabulary. It reads WAV/FLAC in bounded chunks, mixes channels to mono, resamples to 16 kHz, normalizes each chunk and submits 30000 samples per inference (1.875 seconds). It handles the whole file, including the final padded chunk. Each chunk is processed independently with fresh decoder state; the runtime does not carry acoustic state or overlap adjacent windows.

Python and C++ runtimes are provided. Both expose full-file transcription; the
Python runtime includes CTC and legacy decoding, while the native entry is built
and run with the matching board SDK.

<a id="directory"></a>
## Directory structure

```text
asr/
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

| Target | Published artifact | Python runtime | C++ runtime |
| --- | --- | --- | --- |
| S100 | `s:asr:s100/asr.hbm` | Available | Available |
| S600 | `s:asr:s600/asr.hbm` | Available | Available |
| X5 / S100P | None | Not supported | Not supported |

Use the artifact published for the board target. Before inference, the runtime
checks board identity and model tensor metadata.

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

On the corresponding board, prepare its model and run:

```sh
bash samples/speech/asr/model/download.sh --target s100
bash samples/speech/asr/runtime/python/run.sh --target s100 --output-dir outputs/asr-run1
```

Use `s600` in both commands for an S600; never rename the S100 HBM. `PYTHON` selects the interpreter. Output directories must be new. External models require the exact target-qualified asset ID; see [model preparation](model/README.md).

<a id="expected-results"></a>
## Output and decoding behavior

The console prints a full-file transcription and the path to `result.json`. The report binds model/audio/vocabulary hashes, model metadata, decoder mode and each chunk's source position, valid sample count and text. Errors return 2; a failure after processing starts saves `failed.json` with completed chunks.

Default `ctc` collapses adjacent duplicate token IDs before removing blank ID 0. `--decode-mode legacy` removes `<pad>` while retaining repeated IDs. For IDs `[1,1,0,1,2,2]` and tokens `<pad>,a,b`, the results are `aab` (`ctc`) and `aaabb` (`legacy`). Other tokens, punctuation and `|` remain verbatim. Each independent chunk starts with a fresh CTC state.

<a id="entry-points"></a>
## Guides

[Python usage/API](runtime/python/README.md) · [Native usage/API](runtime/cpp/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md) · [Input identities](test_data/README.md).

The Python audio frontend reads WAV/FLAC, mixes channels to mono and resamples
to 16 kHz. The [C++ guide](runtime/cpp/README.md) documents its audio
preprocessing and native tensor requirements.

<a id="license"></a>
## License

The sample follows the repository [Apache-2.0 license](../../../LICENSE). Source copyrights, data and screenshots are retained from the pinned S branch; external dependencies retain their own licenses.

## Native entry

On the matching board, after explicit model/dependency preparation, run
`bash samples/speech/asr/runtime/cpp/run.sh --target s100 --build` from the
repository root (use s600 for S600). No implicit model download occurs.
[The native guide](runtime/cpp/README.md) covers host dry-run, complete build/run
parameters, results, API examples and the Python/native resampling difference.
