English | [简体中文](README_cn.md)

# LiteAvatar test data

[Python runtime](../runtime/python/README.md) · [Model package](../model/README.md)

One small input is bundled here; the avatar data itself is a one-time
download from the MIT-licensed upstream repository.

<a id="bundled"></a>
## Bundled files

| File | Role | SHA-256 |
| --- | --- | --- |
| `asr_example.wav` | example utterance (16 kHz mono, 166 output frames at 30 fps) | `a1bd32dc78493c123f9625a66deee562aed2895f53fbc39f2cca3be7e6f4f20f` |

`asr_example.wav` ships with the ModelScope paraformer package (Apache-2.0)
and is the audio used for pipeline verification.

<a id="avatar-data"></a>
## Avatar data preparation

The runtime consumes an avatar data directory (default
`samples/vision/lite_avatar/test_data/preload`) containing:

```text
preload/
├── bg_video.h264   # prepared hardware-decodable background (one-time host transcode)
├── bg_video.json   # its verified sidecar (geometry + SHA-256 of the stream)
├── bg_video.mp4    # original background loop (MPEG-4 Part 2; conversion input, kept as-is)
├── face_box.txt    # compositing ROI (y1 y2 x1 x2: "355 644 278 697")
└── ref_frames/     # reference frames ref_*.jpg (first 150 are used)
```

Download the upstream sample data pinned to the verified commit, verify it,
and extract it (one-time, ~5 MB):

```bash
cd <anywhere>
wget https://github.com/HumanAIGC/lite-avatar/raw/5b7ec850945e03d56fb290b05fb68440c359fa86/data/sample_data.zip
echo "d1c77b2df48b7fd7f19e6eef52bc5107e3131dd9777162faf2576f9451f69d7b  sample_data.zip" | shasum -a 256 -c -
unzip sample_data.zip -d <repo>/samples/vision/lite_avatar/test_data/
```

The bundled `bg_video.mp4` is MPEG-4 Part 2, which the RDK video decoder
does not support, so the runtime reads a prepared `bg_video.h264` + its
`bg_video.json` sidecar instead. Prepare them once on any host with
FFmpeg by running the script in this directory — it reads the original
`.mp4`, transcodes it, measures the produced stream and writes the
sidecar itself:

```bash
python3 samples/vision/lite_avatar/test_data/prepare_background.py \
    --data samples/vision/lite_avatar/test_data/preload
```

Then copy `bg_video.h264` and `bg_video.json` into the avatar data
directory on the board. Details in
[runtime background preparation](../runtime/python/README.md#background-prep).

The zip is MIT-licensed (upstream repository at commit
`5b7ec850945e03d56fb290b05fb68440c359fa86`). Besides the runtime data it
carries the network weights: `net_encode.pt` and `net_decode.pt` are
**required conversion inputs** (the ONNX export in
[conversion](../conversion/README.md) loads both TorchScript files);
`net.pth` and `neutral_pose.npy` are unused by this sample. If you place
the extracted directory somewhere else, point `--avatar-data` at it.

<a id="directory"></a>
## Directory structure

```text
test_data/
├── README.md              # English instructions
├── README_cn.md           # Chinese instructions
├── asr_example.wav        # Example utterance
├── prepare_background.py  # One-time host background preparation
└── preload/               # Extracted avatar data (not committed)
```
