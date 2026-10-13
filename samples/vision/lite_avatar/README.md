English | [简体中文](README_cn.md)

# LiteAvatar audio-driven talking avatar

Offline rendering of an audio-driven talking avatar on the RDK S100, S100P
and S600: speech audio in, a composited avatar video out.

Sources: [HumanAIGC/lite-avatar](https://github.com/HumanAIGC/lite-avatar) (MIT)

[中文说明](README_cn.md)

<a id="overview"></a>
## Overview

The sample ships one Python runtime for S100, S100P and S600. The `LiteAvatarPipeline` class
chains the four published HBM artifacts through a readable
`preprocess → infer → postprocess` flow per stage and one whole-clip
`predict`: the audio frontend extracts Paraformer features (Kaldi fbank,
7/6 low-frame-rate stacking, CMVN), the audio encoder produces 50 stacked
hidden-state streams, audio2mouth maps sliding 30-frame windows to mouth
parameters (crossfaded and Butterworth-smoothed), and the face decoder
renders each frame conditioned on the reference-frame feature pyramids.
Each frame is composited over the background loop with the feathered mask
and handed to the board's VPU H.264 encoder. FFmpeg muxes the input audio
as AAC into the MP4, and `ffprobe` verifies the delivered file holds both
tracks. Full CLI and API reference:
[runtime/python/README.md](runtime/python/README.md).

### Algorithm background

LiteAvatar (HumanAIGC) drives a lightweight face generator from speech
features: the paraformer encoder's intermediate hidden states carry
articulation information that a small audio2mouth network decodes into
32 mouth parameters per video frame; a decoder network (InstanceNorm-styled
blocks with x2 upsamples) synthesizes the mouth region for the avatar's
current pose, which is blended into the background video. The four per-target
artifacts are quantized with INT16 configured on all nodes (the compiler
still assigns a few ops other dtypes); the decoder's normalization path
stays FP32 through its 21-node `run_on_cpu` list (see
[conversion](conversion/README.md)).

<a id="directory"></a>
## Directory structure

```text
lite_avatar/
├── conversion/  # Export, transforms, calibration and compile configs
├── evaluator/  # Frame-level output quality comparison
├── model/  # Published artifact identities, preparation and verification
├── runtime/  # Python inference (main.py, cli.py, avatar.py, frontend.py, media_worker.py)
├── test_data/  # Bundled example audio; avatar data preparation
├── tests/  # Host tests (SDK doubles, no board needed)
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Host-side Python dependencies
```

<a id="support-matrix"></a>
## Support matrix

| Target | Language | Background decode | Video encoder | Status |
| --- | --- | --- | --- | --- |
| s100 | python | VPU H.264 (file mode) | VPU H.264 worker | supported |
| s100p | python | VPU H.264 (file mode) | VPU H.264 worker | supported |
| s600 | python | VPU H.264 (file mode) | VPU H.264 worker | supported |

<a id="prerequisites"></a>
## Prerequisites

- The four HBM artifacts prepared under `model/<target>/` — from a local
  bundle or self-compilation, verified per
  [model preparation](model/README.md) (`verify.py --target <t>`).
- The avatar data directory extracted from the pinned upstream
  `sample_data.zip` per [test data](test_data/README.md) (default location:
  `test_data/preload`), plus the **one-time host background preparation**:
  the bundled `bg_video.mp4` is MPEG-4 Part 2 and must be transcoded once
  on a host into `bg_video.h264` + its `bg_video.json` sidecar
  ([runtime background preparation](runtime/python/README.md#background-prep)).
- The board stack in [runtime/python/README.md](runtime/python/README.md)
  (`hbm_runtime`, `hobot_vio`, Torch 2.3.1/torchaudio, NumPy, SciPy,
  Pillow, soundfile), the system `liblz4`, and `ffmpeg` (with `ffprobe`)
  on PATH — required for video output; only `--no-encode` runs without it.

<a id="quickstart"></a>
## Quick start

Commands run from the repository root. Steps 1–2 prepare the data on a
host with FFmpeg; steps 3–4 run on the board with the prepared data copied
over.

On the host:

```bash
# 1. Download and verify the avatar data (one-time, ~5 MB)
#    success: shasum prints "sample_data.zip: OK"
cd <anywhere>
wget https://github.com/HumanAIGC/lite-avatar/raw/5b7ec850945e03d56fb290b05fb68440c359fa86/data/sample_data.zip
echo "d1c77b2df48b7fd7f19e6eef52bc5107e3131dd9777162faf2576f9451f69d7b  sample_data.zip" | shasum -a 256 -c -
unzip sample_data.zip -d <repo>/samples/vision/lite_avatar/test_data/
cd <repo>

# 2. Prepare the background once (the script only reads the original .mp4)
#    success: bg_video.h264 + bg_video.json sit in test_data/preload/
python3 samples/vision/lite_avatar/test_data/prepare_background.py \
    --data samples/vision/lite_avatar/test_data/preload
```

Copy the prepared avatar data (including `bg_video.h264` and
`bg_video.json`) and the model artifacts to the board, then on the board:

```bash
# 3. Verify the artifact set (add --target s100p / s600 as needed)
#    success: verify.py exits 0 and prints each digest match
python3 samples/vision/lite_avatar/model/verify.py --target s100

# 4. Render the bundled example utterance
#    output: outputs/lite_avatar/avatar.mp4 (166 frames @ 30 fps, audio muxed)
#    success: exit code 0 and the "Video written" summary line
bash samples/vision/lite_avatar/runtime/python/run.sh --target s100 \
    --audio samples/vision/lite_avatar/test_data/asr_example.wav
```

`--target` selects the artifact set and must match the board (`s100p` /
`s600` on the other boards); `--encoder auto` (the default) resolves to
the VPU H.264 encoder on every supported target, and `--no-encode`
renders without writing.

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

Measured on the bundled 166-frame clip with the plain entry (torch 2.3.1,
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
