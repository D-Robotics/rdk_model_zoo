English | [简体中文](README_cn.md)

# LiteAvatar Python runtime

[Sample](../../README.md) · [Model artifacts](../../model/README.md) · [Conversion](../../conversion/README.md)

<a id="overview"></a>
## Overview

Offline audio-driven talking-avatar rendering for S100, S100P and S600:
one Python runtime chains the four published HBM artifacts (audio encoder,
audio2mouth, face encoder, face decoder) into a whole-clip `predict` and
delivers the frames to the video encoder. Video I/O runs on the VPU: the
prepared background is decoded by the hardware H.264 `Decoder`, every
output frame is encoded by the hardware H.264 `Encoder` in a worker
process, and NumPy/Pillow/SciPy do the resize, color conversion and
compositing between the two. FFmpeg muxes the encoded elementary stream
into MP4 with the video copied (`-c:v copy`) and the input audio as AAC;
`ffprobe` then checks that the delivered file holds both tracks before
the run reports success.

| Target | Language | Background decode | Video encode | Status |
| --- | --- | --- | --- | --- |
| s100 | python | VPU H.264 (file mode) | VPU H.264 worker | supported |
| s100p | python | VPU H.264 (file mode) | VPU H.264 worker | supported |
| s600 | python | VPU H.264 (file mode) | VPU H.264 worker | supported |

Thread policy (the measured board configuration): the entry exports
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=4` before the first heavy
import (an explicitly exported environment wins) and pins
`torch.set_num_threads(2)`; this caps CPU thread counts, and CPU affinity
is not set. Importing the pipeline API alone changes no process-wide
setting.

<a id="directory"></a>
## Directory structure

```text
runtime/python/
├── README.md         # English instructions
├── README_cn.md      # Chinese instructions
├── main.py           # Entry: parse -> select -> construct -> predict -> deliver
├── cli.py            # Options, published selection, hardware writer, delivery
├── avatar.py         # Stages, pipeline, renderer, tensor contracts
├── frontend.py       # Pinned Paraformer audio frontend (fbank, LFR, CMVN)
├── media_worker.py   # VPU decode helpers + hardware H.264 encoder worker
└── run.sh            # Launcher
```

<a id="environment"></a>
## Environment

Board execution needs the `hbm_runtime` and `hobot_vio` SDK modules from
the board image, the board Python stack, and `ffmpeg` (with `ffprobe`):

```bash
# on the board (image Python: 3.10 on S100/S100P, 3.12 on S600; aarch64):
# one scoped venv on a user data disk, pinned to the proven versions
# (standard aarch64 wheels from PyPI)
python3 -m venv --system-site-packages /mnt/sdcard/lite-avatar-venv
/mnt/sdcard/lite-avatar-venv/bin/pip install \
    torch==2.3.1 torchaudio==2.3.1 numpy==1.26.4 \
    scipy soundfile pillow
. /mnt/sdcard/lite-avatar-venv/bin/activate
# (--system-site-packages keeps the image SDK modules, hbm_runtime and
#  hobot_vio, importable)
```

`ffmpeg` (with its `ffprobe`) on PATH is required for video output: it
writes the MP4 container with the AAC audio, and `ffprobe` verifies both
delivered tracks after muxing. `--no-encode` runs without it.

The renderer's background cache loads the system liblz4 through `ctypes`
library discovery: the board image provides it (verified on S100), Ubuntu
hosts provide it via the `liblz4-1` package, and a macOS host needs
`brew install lz4` (a python.org interpreter also needs
`DYLD_LIBRARY_PATH="$(brew --prefix lz4)/lib"`).

The x86 export/calibration host uses a separate recipe
(`torch==2.3.1+cpu` wheels): see
[conversion](../../conversion/README.md#toolchain-targets). `--help`,
`--list-models` and `--dry-run` run on any workstation without board
packages.

<a id="background-prep"></a>
## Preparing the background asset (one time, on a host)

The avatar data's bundled `bg_video.mp4` is MPEG-4 Part 2, which the RDK
video decoder does not support. Once per avatar, run the preparation
script on any host with FFmpeg. It transcodes the `.mp4` (which it only
reads) into `bg_video.h264`, measures the produced stream (frame count,
geometry, SHA-256) and writes the matching `bg_video.json` sidecar beside
it:

```bash
# on the host
python3 samples/vision/lite_avatar/test_data/prepare_background.py \
    --data samples/vision/lite_avatar/test_data/preload
```

Then copy `bg_video.h264` and `bg_video.json` into the avatar data
directory on the board. The runtime checks every sidecar field (`codec`,
visible `width`, `coded_width`, `height`, `fps`, `frames`, `sha256`)
before decoding and refuses a stream whose digest does not match. The
visible width is padded to a 16-pixel multiple to match the decoder's
NV12 luma stride (890 becomes 896 for the bundled asset); the pad columns
are cropped back off when the MP4 is written.

<a id="usage"></a>
## Usage

Run from the repository root:

```bash
# SDK-free listing of the published artifact set
bash samples/vision/lite_avatar/runtime/python/run.sh --list-models

# resolve the four-model selection without loading an SDK
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 --dry-run

# full render with the verified defaults
# input: prepared artifacts + avatar data; output: outputs/lite_avatar/avatar.mp4
# success: exit 0 and the "Video written" summary line
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 \
    --audio samples/vision/lite_avatar/test_data/asr_example.wav \
    --avatar-data /path/to/prepared-avatar-data \
    --output outputs/lite_avatar/avatar.mp4
```

Exit codes: `0` success, `1` dry-run with missing artifacts, `2`
usage/verification/runtime errors (including a full run with missing
artifacts, which lists the exact paths to prepare).

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target: `auto`, `x5`, `s100`, `s100p`, `s600`; LiteAvatar publishes s100/s100p/s600 sets and the target must match detected hardware |
| `--list-models` | flag | false | list manifest-backed asset references without board access |
| `--dry-run` | flag | false | resolve/check the four-model selection without loading an SDK |
| `--audio` | string | null | input WAV (any sample rate; first channel of stereo is used) |
| `--avatar-data` | string | samples/vision/lite_avatar/test_data/preload | avatar data directory (`bg_video.h264` + `bg_video.json`, `face_box.txt`, `ref_frames/`) |
| `--cmvn-path` | string | samples/vision/lite_avatar/model/am.mvn | CMVN file for the audio frontend (pinned `am.mvn`) |
| `--output` | string | outputs/lite_avatar/avatar.mp4 | output video path |
| `--encoder` | choice | auto | `auto` or `hardware` (VPU H.264) |
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
`Video written: <output> (encoder: hardware)`, then exits 0. The output
is an H.264 High MP4 — 890x1920 at 30 fps, limited-range BT.601, 166
frames for the bundled example — with the input audio as AAC.

Measured on the bundled example with the plain entry (torch 2.3.1,
hardware encode on all targets):

| Target | Render rate (166-frame window) | End-to-end wall |
| --- | --- | --- |
| s100p | 18.7 fps | 24.7 s |
| s600 | 20.5 fps | 28.3 s |
| s100 | 15.3 fps | 28.6 s |

The render-rate window covers rendering the 166 frames plus hardware
encoder drain and MP4 muxing; the one-time background decode, audio
frontend and renderer setup before it are excluded. The wall time is the
full CLI run, setup included.

Frame-level output comparison against a reference render:
[evaluator/evaluate_frames.py](../../evaluator/evaluate_frames.py).

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

`result.render_frame(i)` and `result.frames()` return owned frames.
Session semantics: `predict` runs the audio stages, releases the audio
models (they reload, with scheduling restored, on the next `predict`),
and builds the renderer lazily. The background is hardware-decoded once
and held compressed; the decoded frame list is released before the
reference pyramids load, so peak memory stays bounded. Reference pyramids
are encoded once and reused across predicts; the decoder loads per
rendered frame.

<a id="stage-io"></a>
## Stage I/O

The pipeline is four named stages, each with a readable
`preprocess → infer → postprocess` chain inside [avatar.py](avatar.py);
`predict` chains exactly these steps.

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
| `Missing model artifacts` (exit 2) | Prepare the four files under `model/<target>/` per [model/README.md](../../model/README.md); run `model/verify.py --target <t>`. |
| `--dry-run` exits 1 | Same preparation gap; the printed paths are the exact missing files. |
| target selection error | LiteAvatar publishes one four-file set per target (`s100`, `s100p`, `s600`); the target must match the detected board (or use `auto`), and any other target fails explicitly. |
| `No prepared background ... only a bg_video.mp4 file is present` | The avatar data has no prepared background yet; run the one-time host preparation above and place `bg_video.h264` + `bg_video.json` beside the data. |
| `bg_video.h264 SHA-256 ... does not match the sidecar` | The stream and the sidecar come from different transcodes (partial copy, edited file); re-run the host preparation so both files are produced together. |
| `FFmpeg is required for video output (not on PATH)` | The final mux needs `ffmpeg`; install the board image's ffmpeg package, or run with `--no-encode`. |
| `Hardware decoder close failed ...` / `Hardware encoder worker exited ... (exit N)` | The VPU reported a teardown/protocol failure; the run aborts with the worker log path — check `dmesg`/the log for the codec device state and retry. |
| output differs from a reference render | Compare the same artifacts, audio, avatar data and target before judging quality; use [evaluator/evaluate_frames.py](../../evaluator/evaluate_frames.py) against the reference frames. |
