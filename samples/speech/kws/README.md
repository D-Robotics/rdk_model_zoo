# KWS — MDTC keyword spotting

English | [简体中文](README_cn.md)

<a id="overview"></a>
## Overview

This sample runs the published MDTC wake-word model on mono 16 kHz audio. It returns the maximum model probability over the supplied clip and an optional threshold decision for “hey snips”; it does not transcribe speech or detect arbitrary keywords. Only the first 60000 samples are used; shorter clips are zero-padded. At 16 kHz this is **3.75 seconds**. Longer recordings require explicitly selected windows.

### Algorithm and pipeline (MDTC)

The published S100 asset implements MDTC (Multi-Scale Dynamic Temporal Convolution) from the PaddlePaddle + PaddleAudio speech stack. Its published description uses multi-scale temporal convolutions to capture features at different time scales and dynamic convolution to adapt weights across speakers and environments.

The deployment pipeline cuts mono 16 kHz float32 audio to 60000 samples (3.75 seconds; shorter clips are zero-padded). PaddleAudio fbank uses 25 ms frames, 10 ms shift and 80 mel bins to produce `[1, 373, 80]`; the BPU model returns keyword probabilities, and post-processing takes their maximum without adding sigmoid. The application decision is `score >= threshold` (default `0.5`). Set the threshold for the intended application using labeled positive and negative clips. The task composes preprocess → infer → postprocess through predict; audio files, SDK transport, feature extraction and scoring have separate modules.

<a id="support-matrix"></a>
## Support matrix

| Target | Publication | Python | Native C++ |
| --- | --- | --- | --- |
| S100 | `s:kws:s100/kws.hbm` | Available | Not provided |
| X5 / S100P / S600 | None | Unsupported | Not provided |

Use the S100 MDTC artifact with its matching image-provided SDK; loading checks the actual tensor metadata against the model binding.

<a id="prerequisites"></a>
## Prerequisites

Python 3.10+, NumPy, PyYAML, SoundFile, PaddlePaddle and PaddleAudio. Inference requires S100's matching BSP `hbm_runtime` and a prepared model. Install the frontend dependencies in the environment described by the [runtime guide](runtime/python/README.md). Help/list/dry-run need NumPy/PyYAML but no SDK or Paddle; inference does not install packages or download files.

<a id="quickstart"></a>
## Quick start

Run from the repository root. First inspect selection on any host:

```bash
bash samples/speech/kws/runtime/python/run.sh --list-models
bash samples/speech/kws/runtime/python/run.sh --target s100 --dry-run
```

On S100 after explicit dependency preparation, download then run:

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

Success exits 0 and writes `result.json`: score, threshold, `detected`, actual SDK metadata, input/model digests and audio padding/truncation counts. Detection uses `score >= threshold` (default 0.5). The source S100 record for the bundled “hey snips” clip is approximately 0.985. Failures exit 2.

<a id="directory"></a>
## Directory

| Location | Purpose |
| --- | --- |
| `model/` | Explicit manifest-backed download and artifact identity |
| `runtime/python/` | Pure task stages, frontend, audio I/O, shared SDK runner and CLI |
| `test_data/` | Original 2.5-second mono clip and hash/source explanation |
| `conversion/` | Conversion prerequisites and what the source release provides |
| `evaluator/` | Offline labeled-score metrics and preserved historical performance |

<a id="entry-points"></a>
## Guides

- [Model preparation](model/README.md)
- [Python run, parameters and API](runtime/python/README.md)
- [Conversion availability](conversion/README.md)
- [Metrics and historical performance](evaluator/README.md)
- [Test audio](test_data/README.md)

<a id="license"></a>
## License

Sample code follows the repository [Apache-2.0 license](../../../LICENSE); external dependencies retain their respective licenses and copyright notices.
