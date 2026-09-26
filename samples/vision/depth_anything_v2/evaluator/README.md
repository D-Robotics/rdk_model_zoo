[English](README.md) | [简体中文](README_cn.md)

# Evaluation records and reproducibility boundaries

<a id="dataset"></a>
## Dataset

The source contains one demonstration image, `furseal.jpg`, and a rendered
reference. It supplies no depth ground truth, split, dataset license, valid-depth
mask or relative/metric alignment protocol. This directory therefore does not
pretend to implement a dataset evaluator. Obtain those inputs before reporting
AbsRel, RMSE or threshold accuracy; color-image similarity is not a substitute.

![Source input](../test_data/furseal.jpg)

<a id="environment"></a>
## Environment

The historical performance commands require the compatible board SDK's
`hrt_model_exec` and `hrt_ucp_monitor`. The source does not pin firmware, SDK,
clock settings, model digest or measurement provenance. Current host tests do
not execute these commands. Canonical runtime requirements are in the
[Python guide](../runtime/python/README.md).

<a id="command"></a>
## Commands and scope

The source performance command is retained for a future prepared S100 board:

```bash
hrt_model_exec perf --model_file samples/vision/depth_anything_v2/model/s100/depth_any.hbm --frame_count 100 --thread_num 1
hrt_ucp_monitor
```

Run from repository root after explicit model preparation. Changing thread count
requires a separately identified run; the table below is not output from this
command in the current migration. No board connection is attempted here.

For correctness, run the canonical entry on the same input and keep
`raw_depth.npy`, `depth_native.npy`, `report.json` and the reference's corresponding
raw array and provenance. Compare identical preprocessing/resize modes before
visualization; the changed letterbox cropping is an intentional source fix.

<a id="metrics"></a>
## Metrics and interpretation

Raw-array comparison should state shape, dtype, finite-value count, maximum and
mean absolute difference, and any explicit relative tolerance. No universal pass
threshold is inferred from one display image. Relative depth is not meters;
scale/shift or median alignment must be stated if used for dataset metrics.

Display normalization subtracts each image's minimum and divides by its range,
then maps to uint8 INFERNO. It discards scale and offset information. Constant
outputs now have zero grayscale instead of source NaNs; this is defined behavior,
not an indication that a constant prediction is accurate. OpenCV linear replaces
source Torch interpolation; bit equality is not claimed. A host analytic affine
plane verifies half-pixel geometry, not actual HBM output parity.

<a id="outputs"></a>
## Records to retain

Keep input/model hashes, concrete board identity, runtime/firmware version,
preprocessing mode, raw tensor metadata, full argv and stdout/stderr for each
measurement. Preserve unnormalized float arrays and any ground-truth IDs/protocol.
The canonical CLI reports metadata and hashes but deliberately does not measure
latency. Source HRT timing, end-to-end application timing and display IO must be
reported separately. Observed hashes do not fill the missing publisher digest.

<a id="reference-results"></a>
## Historical source results — not remeasured

| Threads | Frames | Total latency (ms) | Reported average (ms) | FPS |
| --- | --- | --- | --- | --- |
| 1 | 100 | 13738.43 | 137.38 | 7.27 |
| 2 | 100 | 26375.53 | 263.74 | 7.54 |
| 4 | 100 | 52214.07 | 521.90 | 7.54 |
| 8 | 100 | 102309.64 | 1020.35 | 7.54 |

Values are copied unchanged. Totals, reported averages and concurrency/FPS cannot
all be reconstructed from the available records; no invented correction or new
benchmark claim is made. The source's rendered result follows:

![Historical source depth](../test_data/readme_img/depth_color.png)

Historical monitor: BPU occupancy 95.4%, ION memory about 300 MB, read bandwidth
about15920 and write bandwidth about11650. Bandwidth units, interval and complete
environment are unspecified in the source; do not assume MB/s or reproduce them
as an acceptance target.

![Historical monitor](../test_data/readme_img/image.png)

<a id="boundaries"></a>
## Boundaries

No dataset evaluation, HRT benchmark, monitor run, model download or real SDK
execution occurred in this host migration. The source's S100P mention does not
establish a matching artifact. Host fixtures exercise contracts, geometry,
constant-map handling, IO and identity refusal; they do not prove board accuracy,
throughput or memory use. Independent acceptance and board validation remain open.
