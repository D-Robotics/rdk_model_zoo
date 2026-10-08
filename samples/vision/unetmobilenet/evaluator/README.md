English | [简体中文](README_cn.md)

# UNetMobileNet validation

<a id="dataset"></a>
## Dataset

Cityscapes defines the 19-class task, but no labeled validation split or dataset runner is supplied by the source. segmentation.png is a smoke input and result.jpg a source-recorded illustration. Dataset evaluation needs licensed images/labels, explicit train-ID mapping, ignored-label policy and a recorded split.

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="environment"></a>
## Environment

Use a matching S100 or S600 board image, model and SDK. Python requires Python 3.10+, NumPy, OpenCV and PyYAML. C++ requires C++17, CMake, OpenCV development libraries and board DNN/UCP headers and libraries.

<a id="command"></a>
## Commands

From the repository root, prepare the S100 model and run both implementations on the same image:

```bash
bash samples/vision/unetmobilenet/model/download.sh --target s100
python3 samples/vision/unetmobilenet/runtime/python/main.py --target s100 \
  --test-img samples/vision/unetmobilenet/test_data/segmentation.png \
  --mask-save-path outputs/unetmobilenet/python-labels.npy
bash samples/vision/unetmobilenet/runtime/cpp/run.sh --target s100 --build \
  --test-img samples/vision/unetmobilenet/test_data/segmentation.png \
  --mask-save-path outputs/unetmobilenet/cpp-labels.png
```

Keep target, artifact and input identical for comparison. For S600, use `--target s600` in every command.

<a id="metrics"></a>
## Metrics

Compare Python NPY and C++ PNG outputs as integer class-ID arrays using the same artifact and image. Report per-pixel agreement. Dataset mIoU requires Cityscapes ground-truth masks and the dataset evaluation protocol.

<a id="outputs"></a>
## Outputs

Python writes int32 NPY labels; C++ writes lossless uint8 PNG IDs while its API mask remains int32. Both retain original dimensions and produce metadata reports plus overlays. Numeric masks should be compared before visualization; JPEG compression prevents pixel-exact overlay comparison.

<a id="reference-results"></a>
## Reference results

No source mIoU/FPS/latency table is available for this sample. The [reference figure](../test_data/result.jpg) comes from the source record.

<a id="boundaries"></a>
## Scope

Use the Python and C++ runtime commands above to prepare matching single-image outputs. Compare integer label masks and quantized score interpretation using the documented inputs and metrics.
