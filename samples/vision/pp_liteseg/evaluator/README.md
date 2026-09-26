English | [简体中文](README_cn.md)

# PP-LiteSeg validation

<a id="dataset"></a>
## Dataset

Delivered street.png/test.jpg support single-image inspection only. Cityscapes is the class vocabulary; no licensed validation set or dataset-level evaluation loop is bundled. For mIoU, obtain the appropriate labeled split, record label-ID mapping/ignore rules and evaluate float and compiled models on exactly the same samples.

<a id="environment"></a>
## Environment

Single-image entry requires the same X5 OS 3.5.0+, Python 3.10+, NumPy/OpenCV/PyYAML and board SDK as the runtime. --help is host-safe. OE hb_perf runs in the conversion container; hrt_model_exec runs on X5. These are distinct measurement environments.

<a id="command"></a>
## Commands

```bash
# cwd: repository root, on X5; model explicitly prepared
bash samples/vision/pp_liteseg/model/download.sh --target x5
python3 samples/vision/pp_liteseg/evaluator/infer_board.py --model samples/vision/pp_liteseg/model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin --image samples/vision/pp_liteseg/test_data/street.png --output outputs/pp_liteseg/eval.png --alpha 0.55
```

Compatibility options: --model and --image are required; --output defaults to result.jpg, --alpha to 0.55. All inference delegates to the canonical runtime. Success returns 0; runtime errors propagate 2. The source entry’s image/model arguments are retained without a second implementation.

<a id="metrics"></a>
## Metrics

No dataset score or timing metric is computed by infer_board.py. Its outputs permit per-pixel exact ID comparison, class-count inspection and visual review. Logit cosine ≥0.95 from the source document applies only to explicit matching logit probes, never the class-ID map. mIoU and latency require separate real measurements with dataset, warmup, repetitions, core configuration and tool version recorded.

<a id="outputs"></a>
## Outputs

For --output outputs/pp_liteseg/eval.png, writes that three-panel image plus eval.labels.npy and eval.report.json alongside it. The mask is int32 512×1024, IDs 0..18. The report preserves runtime metadata/version, artifact identity/path and actual class names; no expected class list is hardcoded. Existing output files are replaced.

<a id="reference-results"></a>
## Reference results

Source README expected ≈95 FPS and ≈10.5 ms at 1024×512 single-core but did not supply a reproducible measurement record. Retained as an unverified source expectation, not an acceptance threshold or a new result. Host source-parity and fixture checks do not prove inference accuracy. This migration has no board, OE or dataset result. See the [source audit](../../../../docs/releases/unified-migration/evidence/2026-09-26-b8-ppliteseg-audit.json).

<a id="boundaries"></a>
## Boundaries

Board testing is not-run because the board environment is unavailable. S100/S100P/S600 and C++ are not supported by this sample. A locally compiled model must satisfy the same tensor contract; the declared asset-id and unknown publisher SHA are not provenance proof. For real performance commands and graph comparison prerequisites, use the [conversion validation section](../conversion/README.md#validation).
