English | [简体中文](README_cn.md)

# UNetMobileNet validation

<a id="dataset"></a>
## Dataset

Cityscapes defines the 19-class task, but no labeled validation split or dataset runner is supplied by the source. segmentation.png is a smoke input and result.jpg a historical illustration. Dataset evaluation needs licensed images/labels, explicit train-ID mapping, ignored-label policy and a recorded split.

<a id="environment"></a>
## Environment

Host suite needs Python 3.10+, NumPy/OpenCV/PyYAML and a C++17 compiler for native pure/fake-interface tests. Real runtime checks require matching S100/S600 images, models and SDKs described in the runtime guides. The fake headers under native tests are deliberately not installation substitutes or real SDK compile evidence.

<a id="command"></a>
## Commands

```bash
# cwd: repository root; host regression (no model/SDK required)
python3 -m unittest discover -s samples/vision/unetmobilenet/tests
# On S100, after explicit preparation; single-image result, not dataset accuracy
bash samples/vision/unetmobilenet/model/download.sh --target s100
python3 samples/vision/unetmobilenet/runtime/python/main.py --target s100 --mask-save-path outputs/unetmobilenet/python.npy --report-path outputs/unetmobilenet/python.json
bash samples/vision/unetmobilenet/runtime/cpp/run.sh --target s100 --build --mask-save-path outputs/unetmobilenet/native.png --report-path outputs/unetmobilenet/native.json
```

Keep both runtime commands on the same target, artifact and input. Use --target s600 for every corresponding step when checking S600. These board commands are provided for future validation and were not run during this migration.

<a id="metrics"></a>
## Metrics

Host checks establish source preprocessing/render parity for fixtures, class-ID decoding, per-channel SCALE counterexamples, direct nearest restoration, target rejection and resource cleanup on injected failures. They do not establish mIoU or latency. Compare Python NPY versus C++ PNG as integer arrays; exact per-pixel agreement is a useful same-artifact smoke criterion. Never compute logit cosine on categorical class IDs.

<a id="outputs"></a>
## Outputs

Python writes int32 NPY labels; C++ writes lossless uint8 PNG IDs while its API mask remains int32. Both retain original dimensions and produce metadata reports plus overlays. Numeric masks should be compared before visualization; JPEG compression prevents pixel-exact overlay comparison.

<a id="reference-results"></a>
## Reference results

No source mIoU/FPS/latency table is available for this sample. The [historical figure](../test_data/result.jpg) is retained without a new measurement claim. [Pinned source audit](../../../../docs/releases/unified-migration/evidence/2026-09-26-b8-unetmobilenet-audit.json) records the deliberate fixes to raw-quantized argmax and native intermediate resize. No board acceptance is inferred from host tests.

<a id="boundaries"></a>
## Boundaries

No dataset evaluation loop, pretrained source model, real SDK compilation, board inference or performance run is delivered as evidence here. Independent review remains pending. The implementation intentionally rejects absent/unsupported integer quantization metadata rather than assuming channel ordering. Native task tests use fake interfaces for failure paths only; actual SDK compatibility still needs a board/toolchain environment.
