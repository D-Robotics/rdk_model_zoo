English | [简体中文](README_cn.md)

# LiteAvatar Python runtime

[Sample](../../README.md) · [Model artifacts](../../model/README.md) · [Conversion](../../conversion/README.md)

<a id="overview"></a>
## Overview

Offline audio-driven talking-avatar rendering for S100: one Python runtime
chains the four published HBM artifacts (audio encoder, audio2mouth, face
encoder, face decoder) into a whole-clip `predict` and delivers the frames
to a video writer. `main.py` keeps the flow visible — parse → select →
construct → predict → deliver; the writers bound every resource (FFmpeg
pipe, S100 hardware encoder worker with a shared-memory ring) and clean up
on failure.

| Target | Language | Status |
| --- | --- | --- |
| s100 | python | supported |

Thread policy (the measured board configuration): the entry exports
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=4` before the first heavy
import (an explicitly exported environment wins) and pins
`torch.set_num_threads(2)` / `cv2.setNumThreads(2)`, keeping the Python
renderer, the BPU runtime and the encoder worker off each other's cores.
Importing the pipeline API alone changes no process-wide setting.

<a id="directory"></a>
## Directory structure

```text
runtime/python/
├── README.md         # English instructions
├── README_cn.md      # Chinese instructions
├── main.py           # Entry: parse -> select -> construct -> predict -> deliver
├── cli.py            # Options, published selection, video writers, delivery
├── avatar.py         # Stages, pipeline, renderer, tensor contracts
├── frontend.py       # Pinned Paraformer audio frontend (fbank, LFR, CMVN)
├── encoder_worker.py # S100 hardware H.264 encoder worker (ring/ACK protocol)
└── run.sh            # Launcher
```

<a id="environment"></a>
## Environment

Board execution needs the `hbm_runtime` SDK from the board image, the
board Python stack, and `ffmpeg`:

```bash
# on the board (Python 3.10, aarch64): one scoped venv on a user data disk,
# pinned to the proven versions (standard aarch64 wheels from PyPI)
python3 -m venv --system-site-packages /mnt/sdcard/lite-avatar-venv
/mnt/sdcard/lite-avatar-venv/bin/pip install \
    torch==2.3.1 torchaudio==2.3.1 numpy==1.26.4 \
    opencv-python==4.11.0.86 scipy soundfile pyyaml
. /mnt/sdcard/lite-avatar-venv/bin/activate
# (--system-site-packages keeps the image SDK, hbm_runtime, importable)
```

`ffmpeg` on PATH is required for video output with **both** encoders — the
software writer pipes raw frames into an ffmpeg libx264 process, and the
hardware path's bundled `encoder_worker.py` muxes the H.264 stream with
the audio through ffmpeg; `--no-encode` runs without it.

The renderer's background cache loads the system liblz4 through `ctypes`
library discovery: the board image provides it (verified on S100), Ubuntu
hosts provide it via the `liblz4-1` package, and a macOS host needs
`brew install lz4` (a python.org interpreter additionally needs
`DYLD_LIBRARY_PATH="$(brew --prefix lz4)/lib"`).

The x86 export/calibration host uses a separate recipe
(`torch==2.3.1+cpu` wheels): see
[conversion](../../conversion/README.md#toolchain-targets). `--help`,
`--list-models` and `--dry-run` run on any workstation without board
packages.

<a id="usage"></a>
## Usage

Run from the repository root:

```bash
# SDK-free listing of the published artifact set
bash samples/vision/lite_avatar/runtime/python/run.sh --list-models

# resolve the four-model selection without loading an SDK
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 --dry-run

# full render with the verified defaults (hardware encoder on S100)
# input: prepared artifacts + avatar data; output: outputs/lite_avatar/avatar.mp4
# success: exit 0 and the "Video written" summary line
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 \
    --audio samples/vision/lite_avatar/test_data/asr_example.wav \
    --output outputs/lite_avatar/avatar.mp4
```

Exit codes: `0` success, `1` dry-run with missing artifacts, `2`
usage/verification/runtime errors (including a full run with missing
artifacts, which lists the exact paths to prepare).

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target: `auto`, `x5`, `s100`, `s100p`, `s600`; S100 is the only LiteAvatar build and must match detected hardware |
| `--list-models` | flag | false | list manifest-backed asset references without board access |
| `--dry-run` | flag | false | resolve/check the four-model selection without loading an SDK |
| `--audio` | string | null | input WAV (any sample rate; first channel of stereo is used) |
| `--avatar-data` | string | samples/vision/lite_avatar/test_data/preload | avatar data directory (`bg_video.mp4`, `face_box.txt`, `ref_frames/`) |
| `--cmvn-path` | string | samples/vision/lite_avatar/model/am.mvn | CMVN file for the audio frontend (pinned `am.mvn`) |
| `--output` | string | outputs/lite_avatar/avatar.mp4 | output video path |
| `--encoder` | choice | auto | `auto` (hardware on S100, software elsewhere), `software` (FFmpeg pipe) or `hardware` (S100 H.264 worker) |
| `--no-encode` | flag | false | skip video writing entirely (renderer-only run) |
| `--model-dir` | string | null | directory holding the manifest-relative model paths |
| `--models-manifest` | string | null | JSON `{filename: sha256}` manifest identifying local rebuilds |
| `--priority` | int | null | runtime scheduling priority (0-255) |
| `--bpu-cores` | int list | null | runtime BPU core indexes |

Defaults above are the values defined in `build_parser` ([cli.py](cli.py)).
`--priority`/`--bpu-cores` set BPU scheduling for all four models; the
request is applied before each model's first inference and restored on
every reload.

<a id="results"></a>
## Results

The run prints the audio/renderer phase timings, the render rate
(`Rendered 166 frames ... in ...s (... fps)` for the bundled example) and
`Video written: <output> (encoder: hardware|software)`, then exits 0. The
output is a decodable H.264 MP4 of the avatar's geometry — the background
loop with the animated mouth ROI — at 30 fps with the input audio muxed.
With `--no-encode` no file is written and only the render summary prints.

<a id="integration-example"></a>
## Integration example

Prerequisites: artifacts prepared (see
[model/README.md](../../model/README.md)) and the board packages above
importable. Every input variable is defined in the example:

```python
from samples.vision.lite_avatar.runtime.python.avatar import (
    AvatarInput, LiteAvatarPipeline,
)
from samples.vision.lite_avatar.runtime.python.cli import (
    read_audio, resolve_selections,
)

selections = resolve_selections("s100")
pipeline = LiteAvatarPipeline.from_models(
    selections,
    cmvn_path="samples/vision/lite_avatar/model/am.mvn",
    data_dir="samples/vision/lite_avatar/test_data/preload",
)
waveform, sample_rate = read_audio(
    "samples/vision/lite_avatar/test_data/asr_example.wav")
result = pipeline.predict(AvatarInput(waveform, sample_rate))
for frame in result.frames():   # uint8 BGR, one per 30 fps output frame
    ...
```

`result.render_frame(i)` and `result.frames()` return owned frames;
compositing arithmetic is preserved from the original pipeline. Session
semantics: `predict` runs the audio stages, releases the audio models
(they reload, with scheduling restored, on the next `predict`), and lazily
builds the renderer (reference pyramids encoded once, reused across
predicts; the decoder loads per rendered frame).

<a id="stage-io"></a>
## Stage I/O

The pipeline is four named stages, each with the readable
`preprocess → infer → postprocess` chain inside
[avatar.py](avatar.py); `predict` chains exactly these steps.

| Stage | Input | Output |
| --- | --- | --- |
| audio frontend (CPU, Torch) | waveform `float32` mono, any sample rate | LFR-stacked, CMVN-normalized fbank feats `float32 [1,400,560]` per 4 s chunk |
| `audio_encoder` | feats `[1,400,560]` | hidden states `[1,400,50,512]` (50 stacked streams) |
| `audio2mouth` | sliding windows `[1,30,50,512]` | mouth params `[1,30,32]` + feat; windows are crossfaded and Butterworth-smoothed to one `float32 [N,32]` curve |
| `facegen_encoder` (per reference frame) | RGB `[1,3,384,384]` | feature pyramid `f0..f3` (encoded once per reference, reused) |
| `facegen_decoder` (per output frame) | pyramid + mouth params `[1,32]` | mouth image `[1,3,384,384]`, composited over the background frame with the feathered mask and ping-pong ring order |

All HBM stages expose FLOAT32 tensors at the runtime boundary; INT16
quantization is internal to the artifacts.

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Missing model artifacts` (exit 2) | Prepare the four files under `model/s100/` per [model/README.md](../../model/README.md); run `model/verify.py`. |
| `--dry-run` exits 1 | Same preparation gap; the printed paths are the exact missing files. |
| target selection error | LiteAvatar has one published set; `--target s100` on an S100 board (or `auto`), every other target fails explicitly. |
| `FFmpeg is required for video output (not on PATH)` | Video output with **both** encoders needs `ffmpeg` on PATH (the hardware path remuxes through it too); install the board image's ffmpeg package, or run with `--no-encode`. |
| `Hardware encoder worker exited ... (exit N)` | The bundled worker crashed mid-protocol; the run aborts with the worker log path — check `dmesg`/the log for the encoder device state and retry. |
| output differs from a reference render | Compare the same artifacts, audio, avatar data and encoder before judging quality; use [evaluator/evaluate_frames.py](../../evaluator/evaluate_frames.py) against the reference frames. |
