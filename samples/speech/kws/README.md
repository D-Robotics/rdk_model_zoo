# KWS — MDTC keyword spotting

English | [简体中文](README_cn.md)

<a id="overview"></a>
## Overview

This sample runs the published MDTC wake-word model on mono 16 kHz audio. It returns one confidence score, the maximum model probability over the supplied clip, and an optional threshold decision for “hey snips”. It is not speech transcription or an arbitrary-keyword model. Only the first 60000 samples are used; shorter clips are zero-padded. At 16 kHz this is **3.75 seconds**, correcting the source comment's 60 seconds. Longer recordings need explicitly chosen windows; the sample does not silently scan them.

The source implementation, algorithm context and historical examples remain in the [S snapshot](../../../platforms/s/samples/speech/kws/README.md). The canonical task keeps preprocess → forward → postprocess → predict; audio files, SDK transport, feature extraction and numeric scoring have separate modules.

<a id="support-matrix"></a>
## Support matrix

| Target | Publication | Python | Native C++ | Current validation |
| --- | --- | --- | --- | --- |
| S100 | `s:kws:s100/kws.hbm` | Implemented | No source implementation | Host frontend/contracts only; board not-run |
| X5 / S100P / S600 | None | Explicitly rejected | None | Unsupported |

A board identity does not certify model precision or SDK compatibility. There is no S600 fallback despite the archived wrapper's broader docstring. Model metadata is validated when loaded; synthetic metadata in host tests is not an observed board descriptor.

<a id="prerequisites"></a>
## Prerequisites

Python 3.10+, NumPy, PyYAML, SoundFile, PaddlePaddle and PaddleAudio. Full inference requires S100's matching BSP `hbm_runtime` and a downloaded model; do not install the unrelated PyPI package of the same name. The [runtime guide](runtime/python/README.md) records tested host frontend versions and explicit dependency setup. Host help/list/dry-run need NumPy/PyYAML, but no SDK or Paddle; inference never installs packages or downloads files.

<a id="quickstart"></a>
## Quick start

Run from the repository root. First inspect selection on any host:

```bash
bash samples/speech/kws/runtime/python/run.sh --list-models
bash samples/speech/kws/runtime/python/run.sh --target s100 --dry-run
```

On S100 after explicit dependency preparation, download then run (not executed as board validation in this migration):

```sh
bash samples/speech/kws/model/download.sh --target s100
bash samples/speech/kws/runtime/python/run.sh --target s100 --output-dir outputs/kws-run1
```

Each result directory must be new. `PYTHON=/path/to/python` selects the wrapper interpreter. For an existing external model use its exact publication identity:

```sh
bash samples/speech/kws/runtime/python/run.sh --target s100 \
  --asset-id s:kws:s100/kws.hbm --model-path /path/to/kws.hbm \
  --audio-file /path/to/mono-16k.wav --output-dir outputs/kws-run2
```

<a id="expected-results"></a>
## Expected results

Success exits 0 and writes `result.json`: score, threshold, `detected`, actual SDK metadata, input/model digests and audio padding/truncation counts. Detection uses `score >= threshold` (default 0.5), a configurable application decision rather than a calibrated false-alarm guarantee. The bundled “hey snips” clip historically scored about 0.985 on the source S100 implementation; that value has **not** been remeasured for this migration. Failures exit 2; no stale report is relabeled as success.

<a id="directory"></a>
## Directory

| Location | Purpose |
| --- | --- |
| `model/` | Explicit manifest-backed download and artifact identity |
| `runtime/python/` | Pure task stages, frontend, audio I/O, shared SDK runner and CLI |
| `test_data/` | Original 2.5-second mono clip and hash/source explanation |
| `conversion/` | Actual missing-recipe prerequisites; no invented compiler commands |
| `evaluator/` | Offline labeled-score metrics and preserved historical performance |
| `tests/` | Host model/feature/scoring/error-path tests, without real SDK |

<a id="entry-points"></a>
## Guides and validation

- [Model preparation](model/README.md)
- [Python run, parameters and API](runtime/python/README.md)
- [Conversion availability](conversion/README.md)
- [Metrics and historical performance](evaluator/README.md)
- [Test audio](test_data/README.md)

Real PaddleAudio frontend comparisons cover the bundled clip, silence and a truncated long waveform. SDK descriptors, board scores, real latency and dataset accuracy remain not-run. Host tests alone do not close independent migration acceptance.

<a id="license"></a>
## License

Follow the repository [Apache-2.0 license](../../../LICENSE). Original source copyright is retained through the archived S implementation; external dependencies have their own licenses.
