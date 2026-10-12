English | [简体中文](README_cn.md)

# LiteAvatar audio-driven talking avatar

Offline rendering of an audio-driven talking avatar on the RDK S100: speech
audio in, a composited avatar video out.

Sources: [HumanAIGC/lite-avatar](https://github.com/HumanAIGC/lite-avatar) (MIT)

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

The sample ships one Python runtime for S100. The `LiteAvatarPipeline` class
chains the four published HBM artifacts through a readable
`preprocess → infer → postprocess` flow per stage and one whole-clip
`predict`: the audio frontend extracts Paraformer features (Kaldi fbank,
7/6 low-frame-rate stacking, CMVN), the audio encoder produces 50 stacked
hidden-state streams, audio2mouth maps sliding 30-frame windows to mouth
parameters (crossfaded and Butterworth-smoothed exactly as upstream), and
the face decoder renders each frame conditioned on the reference-frame
feature pyramids. Frames composite the decoded mouth over the background
loop with the original feathered mask and ping-pong ring order, then the
CLI delivers them to the S100 hardware H.264 encoder (or an FFmpeg pipe)
with bounded, failure-cleaning writers
([runtime/python/README.md](runtime/python/README.md)).

### Algorithm background

LiteAvatar (HumanAIGC) drives a lightweight face generator from speech
features: the paraformer encoder's intermediate hidden states carry
articulation information that a small audio2mouth network decodes into
32 mouth parameters per video frame; a decoder network (InstanceNorm-styled
blocks with x2 upsamples) synthesizes the mouth region for the avatar's
current pose, which is blended into the background video. The four S100
artifacts run INT16 activations on the BPU; the decoder's 21 InstanceNorm
subgraph nodes stay FP32 on the CPU for numerical fidelity (see
[conversion](conversion/README.md)).

<a id="directory"></a>
## Directory structure

```text
lite_avatar/
├── conversion/  # Export, transforms, calibration and compile configs
├── evaluator/  # Frame-level output quality comparison
├── model/  # Published artifact identities, preparation and verification
├── runtime/  # Python inference (main.py, cli.py, avatar.py, frontend.py)
├── test_data/  # Bundled example audio; avatar data preparation
├── tests/  # Host tests (SDK doubles, no board needed)
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Host-side Python dependencies
```

<a id="support-matrix"></a>
## Support matrix

| Target | Language | Status |
| --- | --- | --- |
| s100 | python | supported |

<a id="prerequisites"></a>
## Prerequisites

- The four HBM artifacts prepared under `model/s100/` — from a local bundle
  or self-compilation, verified per [model preparation](model/README.md).
- The avatar data directory extracted from the pinned upstream
  `sample_data.zip` per [test data](test_data/README.md) (default location:
  `test_data/preload`).
- The board stack in [runtime/python/README.md](runtime/python/README.md)
  (`hbm_runtime`, Torch 2.3.1/torchaudio, NumPy, SciPy, OpenCV, soundfile)
  and `ffmpeg` on PATH — required for video output with **both** encoders
  (the hardware path muxes the H.264 stream through ffmpeg); only
  `--no-encode` runs without it.

<a id="quickstart"></a>
## Quick start

One complete path on an S100 board, commands run from the repository root:

```bash
# 1. Verify the prepared artifact set
#    success: verify.py exits 0 and prints each digest match
python3 samples/vision/lite_avatar/model/verify.py

# 2. Prepare the avatar data (one-time, ~5 MB)
#    success: shasum prints "sample_data.zip: OK"
cd <anywhere>
wget https://github.com/HumanAIGC/lite-avatar/raw/5b7ec850945e03d56fb290b05fb68440c359fa86/data/sample_data.zip
echo "d1c77b2df48b7fd7f19e6eef52bc5107e3131dd9777162faf2576f9451f69d7b  sample_data.zip" | shasum -a 256 -c -
unzip sample_data.zip -d <repo>/samples/vision/lite_avatar/test_data/
cd <repo>

# 3. Render the bundled example utterance
#    output: outputs/lite_avatar/avatar.mp4 (166 frames @ 30 fps, audio muxed)
#    success: exit code 0 and the "Video written" summary line
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 \
    --audio samples/vision/lite_avatar/test_data/asr_example.wav
```

`--encoder auto` (the default) resolves to the S100 hardware H.264 encoder;
pass `--encoder software` for the FFmpeg pipe or `--no-encode` to render
without writing. Full CLI and API reference:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The run prints the audio/renderer phase timings, the render rate
(`Rendered 166 frames ... in ...s (... fps)`) and `Video written:
outputs/lite_avatar/avatar.mp4 (encoder: hardware)`, then exits 0. The
output is a decodable H.264 MP4 of the avatar's geometry (background loop
with the animated mouth ROI) at 30 fps with the input audio muxed. Missing
artifacts fail with exit code 2 and the preparation pointers; a
self-compiled set runs through `--models-manifest` as described in
[model preparation](model/README.md).

Output quality is measured against a reference render with
[evaluator/evaluate_frames.py](evaluator/evaluate_frames.py); artifact-level
validation of a self-compiled set is described in
[conversion](conversion/README.md#validation).

### Performance

Measured with the plain entry on S100 (torch 2.3.1, hardware encoder,
default thread policy): the bundled 166-frame clip renders in **7.54 s —
22.01 fps** (`Rendered 166 frames ... in 7.54s (22.01 fps)`). The timed
window covers rendering all 166 frames plus encoder drain and MP4 muxing;
the one-time offline audio-frontend and renderer-initialization phases
before it are excluded.

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)
- Test data: [test_data/README.md](test_data/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
upstream project is [lite-avatar](https://github.com/HumanAIGC/lite-avatar)
(MIT, pinned at `5b7ec850945e03d56fb290b05fb68440c359fa86`); its complete
license text ships as [LICENSE-LiteAvatar](LICENSE-LiteAvatar) beside this
README. The vendored FunASR frontend the conversion scripts call for CMVN
(`am.mvn`) is MIT — see
[conversion/LICENSE-FunASR](conversion/LICENSE-FunASR) — and the example
utterance `test_data/asr_example.wav` ships with the ModelScope paraformer
package (Apache-2.0; see [test_data/README.md](test_data/README.md)). The
model weights are downloaded from the upstream ModelScope repository
[HumanAIGC-Engineering/LiteAvatarGallery](https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery);
its model card states no license for the weights, so they remain
user-prepared conversion inputs (see [conversion](conversion/README.md)).
