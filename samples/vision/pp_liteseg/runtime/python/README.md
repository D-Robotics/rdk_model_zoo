English | [简体中文](README_cn.md)

# PP-LiteSeg Python runtime

<a id="overview"></a>
## Python inference

Generate a 19-class Cityscapes segmentation map with PP-LiteSeg.

<a id="directory"></a>
## Directory structure

```text
python/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── cli.py  # Arguments, model selection and result output
├── main.py  # CLI entry: construct model and call predict
├── pp_liteseg.py  # Model initialization and inference stages
└── run.sh  # Run the sample
```

`main.py` constructs the model and calls `predict`; `cli.py` handles options, model selection, and image/report output. The task file owns model stages and image loading uses `utils.py_utils.image.read_bgr_image`.

<a id="environment"></a>
## Environment

RDK X5 OS 3.5.0+, Python 3.10+, board-provided hbm_runtime. Install general dependencies from the repository root with `python3 -m pip install numpy opencv-python PyYAML`. Do not install an arbitrary same-named SDK package. Help/list/dry-run work on a host without SDK or model.

<a id="usage"></a>
## Usage

```bash
# cwd: repository root; prepare explicitly, then run on RDK X5
bash samples/vision/pp_liteseg/model/download.sh --target x5
python3 samples/vision/pp_liteseg/runtime/python/main.py
# Host-only inspection, no SDK/model/download required:
python3 samples/vision/pp_liteseg/runtime/python/main.py --dry-run --target x5
```

```bash
# cwd: repository root; custom paths, same explicitly prepared asset
python3 samples/vision/pp_liteseg/runtime/python/main.py --target x5 --test-img samples/vision/pp_liteseg/test_data/test.jpg --output outputs/pp_liteseg/custom.png --alpha 0.4
```

Success returns 0, validation/load errors return 2. run.sh forwards these same arguments and sets cwd to the repository root; inference never downloads.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--target` | choice | `auto` | auto resolves x5; actual execution checks board identity |
| `--asset-id` | str | `None` | exact published identity |
| `--model-path` | Path | `None` | default model/ BIN; external path requires asset-id |
| `--test-img` | Path | `samples/vision/pp_liteseg/test_data/street.png` | sample-relative absolute default |
| `--output` | Path | `outputs/pp_liteseg/result.jpg` | three-panel image |
| `--mask-save-path` | Path | `outputs/pp_liteseg/labels.npy` | int32 class-ID array;.npy required |
| `--report-path` | Path | `outputs/pp_liteseg/result.json` | JSON evidence |
| `--alpha` | float | `0.55` | overlay weight in [0,1] |
| `--input-width` | int | `1024` | fixed compiled width |
| `--input-height` | int | `512` | fixed compiled height |
| `--priority` | int | `None` | 0..255; omit to retain SDK default |
| `--bpu-cores` | int list | `None` | nonnegative SDK indices; omit to retain default |
| `--list-models` | flag | `false` | manifest listing only |
| `--dry-run` | flag | `false` | resolve selection only; exclusive with list-models |

`--help` / `-h` prints help. Output paths are relative to cwd and existing files are replaced. Dry-run checks selection and CLI configuration, not artifact bytes, metadata or board compatibility.

<a id="results"></a>
## Results

The 3078×548 image contains three 1024×512 panels, separators and a 36-pixel header. labels.npy preserves int32 IDs 0..18 at 512×1024. JSON/stdout fields: target, asset_id, model_path, input_path, publisher_sha256, runtime_version, metadata, class_ids, class_names, mask_shape, output_shape, output and mask_save_path. No confidence scores, original-size mask, latency or mIoU are returned.

<a id="integration-example"></a>
## Integration example

```python
# cwd: repository root; on X5 after explicit model preparation
import cv2
from samples.vision.pp_liteseg.runtime.python.cli import resolve_selection, SAMPLE_DIR
from samples.vision.pp_liteseg.runtime.python.pp_liteseg import PPLiteSegSegmenter

image = cv2.imread(str(SAMPLE_DIR / "test_data/street.png"))
selection = resolve_selection("x5")
task = PPLiteSegSegmenter(selection)
prepared = task.preprocess(image)
raw = task.infer(prepared.tensors)
mask = task.postprocess(raw)
mask_again = task.predict(image)
print(mask.shape, mask.dtype)  # (512, 1024), int32
```

The model initializes the runtime and implements preprocessing, inference, postprocessing and `predict`. Use a separate model instance for each concurrent worker.

<a id="stage-io"></a>
## Stage IO

preprocess takes nonempty HWC BGR uint8, stretches to 1024×512 with INTER_LINEAR, then packs contiguous NV12 uint8 `(768,1024)`. No CPU normalization or letterbox. Returned context carries independent original dimensions; it is not mutable task state. infer returns raw int32 `(1,512,1024,1)` unchanged. postprocess requires IDs 0..18 and returns an owned int32 `(512,1024)` array. It never applies softmax, argmax, dequantization or resize-back. Binding accepts logical NV12 metadata NCHW/NHWC or physical packed shape; logits exports and wrong dtype/geometry are rejected.

<a id="troubleshooting"></a>
## Troubleshooting

Missing BIN: run explicit preparation. Missing SDK: use the X5 board image, not a host workaround. S-series targets have no asset and cannot fall back to X5. Wrong output metadata: inspect the deployed graph; do not add argmax speculatively to an already-decoded map. Keep 1024×512 geometry. Missing image: use the delivered street.png (not street.jpg). For external files, supply the exact asset-id and retain the file provenance separately.
