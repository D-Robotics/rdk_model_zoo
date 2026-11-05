English | [简体中文](README_cn.md)

# PP-LiteSeg validation

<a id="dataset"></a>
## Dataset

Delivered street.png/test.jpg support single-image inspection only. Cityscapes is the class vocabulary; no licensed validation set or dataset-level evaluation loop is bundled. For mIoU, obtain the appropriate labeled split, record label-ID mapping/ignore rules and evaluate float and compiled models on exactly the same samples.

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── infer_board.py  # Python script
```

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

Options: --model and --image are required; --output defaults to result.jpg, --alpha to 0.55. All inference delegates to the runtime. Success returns 0; runtime errors propagate 2.

<a id="metrics"></a>
## Metrics

No dataset score or timing metric is computed by infer_board.py. Its outputs permit per-pixel exact ID comparison, class-count inspection and visual review. Logit cosine ≥0.95 from the source document applies only to explicit matching logit probes, never the class-ID map. mIoU and latency require separate real measurements with dataset, warmup, repetitions, core configuration and tool version recorded.

<a id="outputs"></a>
## Outputs

For --output outputs/pp_liteseg/eval.png, writes that three-panel image plus eval.labels.npy and eval.report.json alongside it. The mask is int32 512×1024, IDs 0..18. The report preserves runtime metadata/version, artifact identity/path and actual class names; no expected class list is hardcoded. Existing output files are replaced.

<a id="reference-results"></a>
## Reference results

The source README records ≈95 FPS and ≈10.5 ms at 1024×512 single-core without a reproducible measurement protocol; treat it as a source expectation and set thresholds from your own measurements. Board, OE and dataset results are recorded when produced.

<a id="boundaries"></a>
## Scope

Board execution requires an RDK X5 environment. S100/S100P/S600 and C++ are not supported by this sample. A locally compiled model must satisfy the same tensor contract; with no publisher checksum published, run it with the matching declared asset-id and record its actual digest. For real performance commands and graph comparison prerequisites, use the [conversion validation section](../conversion/README.md#validation).

## S100 / S100P / S600 logits variant

The new `stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104` variant uses the official trained PaddleSeg STDC1 Cityscapes checkpoint. Its S input is normalized RGB float32 NCHW `(1,3,512,1024)`: RGB/255, mean .5, std .5. The HBM exposes 19-class float logits; the Python runtime applies one CPU argmax. Select the exact converted variant or manifest asset. The existing X5 NV12/int32 class-map contract remains the default.

```bash
python3 samples/vision/pp_liteseg/model/download.py --target s100 --variant stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104
python3 samples/vision/pp_liteseg/runtime/python/main.py --target s100 --variant stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104 --test-img /path/to/street.png
```

Use `--list-models` to obtain the exact asset reference before an external `--model-path` override. S100P and S600 require their own target and HBM. Board smoke, Cityscapes accuracy, Runtime 1/2 concurrency and C++ 1/2 streams remain not run; compiler estimates and source-host checks are conversion evidence. See `conversion/converted_targets_20261103.json` and `conversion/source_export_20261104.json`.
