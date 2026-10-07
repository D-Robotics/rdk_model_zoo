[English](README.md) | [简体中文](README_cn.md)

# Evaluation records and reproducibility boundaries

<a id="dataset"></a>
## Dataset

The source contains one demonstration image, `furseal.jpg`, and a rendered
reference. It supplies no depth ground truth, split, dataset license, valid-depth
mask or relative/metric alignment protocol. This directory therefore ships no
dataset evaluator. Obtain those inputs before reporting
AbsRel, RMSE or threshold accuracy; color-image similarity is not a substitute.

![Source input](../test_data/furseal.jpg)

<a id="environment"></a>
## Environment

The performance commands require the compatible board SDK's
`hrt_model_exec` and `hrt_ucp_monitor`. The source record does not pin firmware, SDK,
clock settings, model digest or measurement provenance. Runtime requirements are in the
[Python guide](../runtime/python/README.md).

<a id="command"></a>
## Commands and scope

The source performance command, to run on a prepared S100 board:

```bash
hrt_model_exec perf --model_file samples/vision/depth_anything_v2/model/s100/depth_any.hbm --frame_count 100 --thread_num 1
hrt_ucp_monitor
```

Run from repository root after explicit model preparation. Changing the thread
count requires a separately identified run.

For correctness, run the canonical entry on the same input and keep
`raw_depth.npy`, `depth_native.npy`, `report.json` and the reference's corresponding
raw array and provenance. Compare identical preprocessing/resize modes before
visualization; letterbox mode crops the padding before restoring the original
size (see [expected results](../README.md#expected-results)).

<a id="metrics"></a>
## Metrics and interpretation

Raw-array comparison should state shape, dtype, finite-value count, maximum and
mean absolute difference, and any explicit relative tolerance. No universal pass
threshold is inferred from one display image. Relative depth is not meters;
scale/shift or median alignment must be stated if used for dataset metrics.

Display normalization subtracts each image's minimum and divides by its range,
then maps to uint8 INFERNO. It discards scale and offset information. Constant
outputs have zero grayscale instead of NaN; this is defined behavior,
not an indication that a constant prediction is accurate. Resizing uses OpenCV
linear interpolation instead of the source's Torch interpolation and matches it
up to floating-point rounding. A host analytic affine
plane verifies the half-pixel geometry; HBM output parity is verified with the
board runtime.

<a id="outputs"></a>
## Records to retain

Keep input/model hashes, concrete board identity, runtime/firmware version,
preprocessing mode, raw tensor metadata, full argv and stdout/stderr for each
measurement. Preserve unnormalized float arrays and any ground-truth IDs/protocol.
The canonical CLI reports metadata and hashes but deliberately does not measure
latency. Source HRT timing, end-to-end application timing and display IO must be
reported separately. Observed hashes do not fill the missing publisher digest.

<a id="reference-results"></a>
## Source performance record

| Threads | Frames | Total latency (ms) | Reported average (ms) | FPS |
| --- | --- | --- | --- | --- |
| 1 | 100 | 13738.43 | 137.38 | 7.27 |
| 2 | 100 | 26375.53 | 263.74 | 7.54 |
| 4 | 100 | 52214.07 | 521.90 | 7.54 |
| 8 | 100 | 102309.64 | 1020.35 | 7.54 |

Values are from the source record. Totals, reported averages and concurrency/FPS
cannot all be reconciled from the available record. The source's rendered result:

![Source depth result](../test_data/readme_img/depth_color.png)

Source monitor record: BPU occupancy 95.4%, ION memory about 300 MB, read bandwidth
about15920 and write bandwidth about11650. Bandwidth units, interval and complete
environment are unspecified in the source record.

![Source monitor record](../test_data/readme_img/image.png)

<a id="boundaries"></a>
## Boundaries

- No dataset evaluation implementation is included; host fixtures exercise
  contracts, geometry, constant-map handling, IO and identity refusal.
- The source mentions S100P, but no matching artifact is published; support follows the published artifacts.
