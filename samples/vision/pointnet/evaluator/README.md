English | [简体中文](README_cn.md)

# PointNet validation and historical results

<a id="dataset"></a>
## Input data

`../test_data/chair.pts` is a single delivered XYZ cloud, not a labeled evaluation
split. There is no dataset downloader, label array, split version or dataset-wide
accuracy evaluator in the source. Use the delivered file unchanged for the
functional smoke command; prepare external labeled data separately for real metrics.

<a id="environment"></a>
## Environment

Host tests need NumPy and PyYAML. Board smoke additionally needs the matching S100
HBM/SDK; plotting optionally needs matplotlib. Tests exercise the same PointNet
stages and binding as the CLI. This directory contains records, not another
inference implementation.

<a id="command"></a>
## Commands

```bash
# cwd: repository root; host numerical/contract checks, no SDK
python3 -m unittest discover -s samples/vision/pointnet/tests
```
```bash
# cwd: repository root; on S100 with the published HBM already prepared
python3 samples/vision/pointnet/runtime/python/main.py --target s100 --no-plot --output-dir outputs/pointnet-check
```

Smoke options: `--target s100` (parser default auto→s100), `--no-plot` (default
false), `--output-dir outputs/pointnet-check` (default outputs/pointnet).
Input defaults to the delivered chair; other options are in the
[complete runtime table](../runtime/python/README.md#parameters). Host tests are
small fixture checks; board duration is not measured in this migration.

<a id="metrics"></a>
## Interpretation

A functional run should produce N labels in 0..3; counts sum to N. Inspect point
regions using the historical figures below. Seeing all four colors is neither a
mandatory condition for every cloud nor an accuracy metric. Dataset mIoU, per-part
IoU and throughput of the new entry have not been measured. The source's “trans”
and “pred” quantization numbers lack a named metric and cannot be called mIoU.

<a id="outputs"></a>
## Outputs

`outputs/pointnet-check/labels.npy` stores point IDs and `result.json` stores counts,
normalization and metadata. Remove `--no-plot` to also write `result_orig.png` and
`result.png`. No metric summary is fabricated from those pictures.

![Original source chair](../test_data/readme_img/chair.png)
![Source segmentation](../test_data/readme_img/chair_res.png)

<a id="reference-results"></a>
## Historical reference results

The original S branch evaluator records these `hrt_model_exec` values. Its table
does not state latency units, exact invocation, SDK version, or artifact digest;
values below are retained verbatim and are not current unified-entry measurements.

| Threads | Frames | Total Latency | Average Latency | FPS |
| --- | --- | --- | --- | --- |
| 1 | 100 | 143.63 | 1.43 | 689.61 |
| 2 | 100 | 216.20 | 2.16 | 914.32 |
| 4 | 100 | 429.76 | 4.30 | 910.70 |
| 8 | 100 | 839.86 | 8.35 | 910.84 |

The original conversion record also reports int16 “trans > 0.9999” and “pred > 0.98”;
see the [preserved screenshot and limitations](../conversion/README.md#calibration).
Unified-entry board status: **not-run**. Host checks: see the test command above;
fixture success is not hardware acceptance.

<a id="boundaries"></a>
## Boundaries

No labeled dataset evaluator, BPU benchmark wrapper or current board numerical
comparison is provided here. Reproducing historical performance requires the
missing artifact/environment/command identity. S100P/S600 have no published
PointNet asset and are not silently mapped to S100. Code follows
[Apache-2.0](../../../../LICENSE).
