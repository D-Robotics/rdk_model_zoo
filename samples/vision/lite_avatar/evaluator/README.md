English | [简体中文](README_cn.md)

# LiteAvatar evaluation

<a id="dataset"></a>
## Dataset

LiteAvatar is a generative pipeline, not a classifier: quality is measured by
comparing a rendered output against a reference render — a stored golden
render, or another run of the same renderer on identical inputs — frame by
frame. The compared sequences come from the same input WAV (e.g.
the bundled `asr_example.wav`, 166 frames at 30 fps) and the same avatar
data directory, so any difference isolates the compared runtime from the
reference. There is no labeled dataset; the reference render is the truth.

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md           # English instructions
├── README_cn.md        # Chinese instructions
└── evaluate_frames.py  # Frame-level PSNR/MAE comparison with gate
```

<a id="environment"></a>
## Environment

Host-side comparison needs the sample's host dependencies
([requirements-host.txt](../requirements-host.txt)): NumPy and OpenCV for
frame decode and the arithmetic; no board SDK. Frames may come from a
directory of `.png`/`.jpg` images or (for the reference) a video file
decoded through OpenCV.

<a id="command"></a>
## Command

Compare a rendered frame directory against a reference, from the repository
root (success: exit 0; gate failure: exit 1; input error: exit 2):

```bash
# input: rendered frames + reference frames (or one reference video)
# output: evaluation.json plus the per-run summary on stdout
python3 samples/vision/lite_avatar/evaluator/evaluate_frames.py \
    --actual /path/to/rendered_frames \
    --reference /path/to/reference_frames \
    --gate 40.0 \
    --output /path/to/avatar-evaluation/evaluation.json
```

`--actual` and `--reference` accept directories of `.png`/`.jpg` frames
(matched by sorted order); `--reference` also accepts a video file. Both
sides stream in lockstep — only per-frame metrics are kept, never both full
sequences — and the streams are released on every exit path.

Count semantics: the default full comparison **fails** (exit 1) when the
frame counts differ — a truncated delivery cannot pass. `--limit N` opts
into a clearly labeled prefix check of the first N frames and **requires at
least N frames on both sides** (a shorter sequence fails; the check never
silently verifies fewer frames than advertised). `--limit` must be
positive and `--gate` finite; both are usage errors (exit 2) otherwise.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| per-frame PSNR | `10*log10(255^2 / MSE)` on uint8 pixel values, per frame | identical geometry, both sequences uint8 BGR |
| per-frame PSNR of a perfect frame | `null` (no finite PSNR exists); counted in `perfect_frames` | bit-identical frame |
| mean / min PSNR | mean / min over the frames with a real MSE (`null` when every compared frame is perfect; a fully perfect run passes any finite gate) | aggregates computed from raw values, rounded only for output |
| per-frame / mean MAE | mean absolute difference on uint8 pixel values (0–255 scale; `0` = identical frames, `255` = maximally different) | same conditions |
| cardinality check | default full comparison requires equal counts (mismatch → FAIL, exit 1); `--limit N` requires ≥ N frames on each side | part of the verdict, not a separate error |

<a id="outputs"></a>
## Outputs

The command prints the per-run summary (frames compared, perfect-frame
count, mean PSNR, mean MAE, counts-equal verdict) and writes a JSON record
holding every frame's PSNR and MAE plus the summary and the compared paths
(`psnr_db` is `null` for perfect frames — never a capped value). Exit code
is 0 on pass, 1 on comparison or gate failure (count mismatch, prefix
shorter than `--limit`, gate miss) and 2 on usage/input errors (missing or
undecodable frames with their exact file names, geometry mismatch,
non-finite `--gate`).

<a id="reference-results"></a>
## Reference results

This script has no fixed reference numbers: the values depend entirely on
the chosen reference render, the encoder and the artifact set, so each
comparison establishes its own baseline JSON record. Artifact-level
validation numbers of a self-compiled set (e.g. the decoder-vs-FP32-ONNX
tensor check, computed on the decoder's normalized float output scale —
not by this script and not on its uint8 frame scale) are recorded in
[conversion](../conversion/README.md#validation).

<a id="boundaries"></a>
## Evaluation boundary

The metric compares rendered pixels only. It does not judge lip-sync
quality, temporal smoothness or audio-video alignment — for those, watch
the output video; the numeric gate is a regression check that the sample
runtime reproduces the reference render. When comparing against a golden
reference, keep the artifact set, avatar data, audio and encoder fixed:
differences then trace to the compared runtime, not the inputs. End-to-end
video checks (decodability, geometry, duration, muxed audio) are part of
the sample's board verification, not of this script.
