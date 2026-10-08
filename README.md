# RDK Model Zoo

[简体中文](README_cn.md)

RDK Model Zoo provides model preparation, preprocessing, BPU inference,
postprocessing and application-validation examples for D-Robotics devices.
Each sample README is the shared entry for people and Agents: commands,
I/O, source structure, conversion and evaluation. Model inference runs
with Python and the board SDK alone. Source version: 2.0.0
(`VERSION` at the repository root).

## Start by task

The [sample index](samples/README.md) lists 51 samples: 45 vision, three
speech, one robotics policy and two LLM samples. Each guide states its own
targets, variants and languages.

| Task | Entry |
|---|---|
| Detection, segmentation, pose, classification, oriented boxes | [Ultralytics YOLO](samples/vision/ultralytics_yolo/README.md) |
| Prompt-free instance segmentation | [YOLOE](samples/vision/yoloe/README.md) |
| Speech recognition | [ASR](samples/speech/asr/README.md), [Paraformer](samples/speech/paraformer/README.md) |
| Keyword spotting | [KWS](samples/speech/kws/README.md) |
| Offline robotics policy | [HIMLoco](samples/robotics/himloco/README.md): six-frame observations to actions, without robot control |
| Vision-language model | [Gemma4-E2B](samples/llm/gemma4-e2b/README.md): native chat, HTTP, single-shot inference and verification tools |
| Text generation | [MiniCPM5-2B](samples/llm/minicpm5-2b/README.md): S100/S100P OELLM 1.0.0 and S600 OELLM 2.0 beta native entries |
| Image classification | [ResNet](samples/vision/resnet/README.md), MobileNet, EfficientNet, ConvNeXt, Rep families and others in the full index |
| Text detection and recognition | [PaddleOCR](samples/vision/paddle_ocr/README.md) |
| Prompted segmentation | [EfficientSAM](samples/vision/efficient_sam/README.md), [MobileSAM](samples/vision/mobile_sam/README.md) |
| Detection, open vocabulary and tracking | [YOLOv5](samples/vision/yolov5/README.md), [FCOS](samples/vision/fcos/README.md), [YOLOWorld](samples/vision/yoloworld/README.md), [ByteTrack](samples/vision/bytetrack/README.md) |
| License plates and image matting | [LPRNet](samples/vision/lprnet/README.md), [MODNet](samples/vision/modnet/README.md) |
| Point cloud part segmentation | [PointNet](samples/vision/pointnet/README.md) |
| Semantic segmentation | [UNet](samples/vision/unet/README.md) · [PP-LiteSeg](samples/vision/pp_liteseg/README.md) · [UNetMobileNet](samples/vision/unetmobilenet/README.md) |
| Monocular depth | [YOLO26 Depth](samples/vision/yolo26_depth/README.md) · [Depth Anything V2](samples/vision/depth_anything_v2/README.md) |
| Lane embeddings and binary labels | [LaneNet](samples/vision/lanenet/README.md) |
| Prepared-feature trajectory planning | [DiffusionDrive](samples/vision/diffusiondrive/README.md) |
| Features, image-text matching and video classification | [DINOv2](samples/vision/dinov2/README.md), [SigLIP](samples/vision/siglip/README.md), [CLIP](samples/vision/clip/README.md), [3DResNet](samples/vision/3dresnet/README.md) |

ACT/Pi0 are integrated as pinned upstream Git submodules for
vision-language-action policy work — see the [VLA guide](samples/vla/README.md):
S100 and S600 ACT use different source versions, Pi0 targets S600, and
model resources are operator-supplied.

## Boards, artifacts and environments

| Target | Artifact | March | Notes |
|---|---|---|---|
| RDK X5 | `.bin` | bayes-e | 4GB/8GB memory variants |
| RDK S100 | `.hbm` | nash-e | Not interchangeable with S100P/S600 |
| RDK S100P | `.hbm` | nash-m | Only published combinations |
| RDK S600 | `.hbm` | nash-p | Input geometry follows the actual artifact |

Samples select the board through their native CLI — ResNet uses
`--target auto|x5|s100|s100p|s600`, YOLO uses `--platform`; `auto`
resolves the target from the system board identity (SoC name/board type
plus socinfo and device-tree fallbacks) and an unknown board raises an
explicit error. Use the SDK supplied by the matching board image (the
`hbm_runtime` module ships with that image). Sample guides control
host/board dependencies, image versions and memory requirements. Common
Python dependencies include NumPy, OpenCV, SciPy and PyYAML; tasks do not
all share one input protocol or install set. C++ needs board development
headers/libraries; conversion needs host training/OE environments.

## Quick start

Clone the complete repository and prepare the dependencies documented by
the chosen sample in its Python environment. Ordinary inference does not
require VLA submodule initialization.

```bash
git clone --branch develop https://github.com/D-Robotics/rdk_model_zoo.git
cd rdk_model_zoo
python3 samples/vision/ultralytics_yolo/runtime/python/main.py --help
python3 samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --list-models
```

Then, on a matching X5, explicitly prepare a model and run detection from
the repository root:

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n
python3 samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --model-path samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin \
  --label-file datasets/coco/coco_classes.names \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/rdk-yolov8n.jpg
```

Success prints `[Saved]` and writes `/tmp/rdk-yolov8n.jpg`; the input is
bundled in `test_data`. Other boards require matching targets/artifact
paths. Without a board use `--dry-run` for an argument-resolution check.
See [YOLO model preparation](samples/vision/ultralytics_yolo/model/README.md)
and [runtime instructions](samples/vision/ultralytics_yolo/runtime/python/README.md)
for offline copying, hash handling and other tasks. For a first
classification run see the [ResNet quick start](samples/vision/resnet/README.md#quickstart);
for a board test checklist see the
[board smoke test](docs/validation/board-smoke-test.md).

## Read and extend the code

```text
samples/                  # task implementations and guides
docs/release/             # artifact/benchmark facts and target identity
docs/sample-standards/    # README and inference contracts
docs/architecture/        # readable runtime architecture
docs/validation/          # board smoke-test checklist
datasets/                 # dataset entry points
utils/                    # compatibility utilities
tools/                    # catalog, contract checks and validation tooling
```

Every Python runtime follows one shape: `main.py` stays a thin entry —
parse arguments, construct the model class, call `predict`, show the
result; the model class implements the `preprocess → infer → postprocess`
chain in one readable file, assembled by `predict`; sample-local CLI
helper modules (`cli.py`, `yolo_cli.py`) own argument, listing, dry-run
and file-IO work; runner/binding modules isolate the SDK session and
tensor contracts. The pattern is described in
[docs/architecture/model-examples.md](docs/architecture/model-examples.md),
with ResNet ([classify.py](samples/vision/resnet/runtime/python/classify.py))
and YOLO detection ([detect.py](samples/vision/ultralytics_yolo/runtime/python/detect.py))
as the reference implementations. The two `samples/llm` samples provide
native generate/stream/reset C++ interfaces. `conversion/` and
`evaluator/` each have actionable guides; shared mechanisms live under
[utils/py_utils](utils/py_utils/README.md).

Read [AGENTS.md](AGENTS.md), the [inference
contract](docs/sample-standards/inference-contract.md) and the [README
contract](docs/sample-standards/readme-contract.md) before development.
People and Agents use the same native commands.

## Data and validation

- [Canonical release facts](docs/release) hold artifacts and historical
  measurements (`docs/release/{x5,s}/models.yaml` are the active
  manifests; target identity aliases live in
  `docs/release/platforms.json`).
- Dataset preparation: [datasets](datasets). Large datasets/models are
  generally not in Git.
- [TROS](docs/tros/README.md) documentation for board runtime stacks.
- Performance tables in sample guides record published measurements with
  their stated conditions; dataset-accuracy evaluation is documented per
  sample in its `evaluator/` guide.

## Catalog data (maintainers)

`tools/catalog-publisher` validates platform manifests and produces
derived data. Use Node 22.12+ and below 23 as required by package.json.
From the repository root:

```bash
npm --prefix tools/catalog-publisher ci
npm --prefix tools/catalog-publisher run check
npm --prefix tools/catalog-publisher run catalog:build
```

Generated `dist/catalog.meta.json` binds `catalog.json` by SHA-256; CI
uploads data artifacts.

## Community, contribution and license

- [Online model catalog](https://d-robotics.github.io/rdk_model_zoo/)
- [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues) —
  include target, image/SDK, model reference, commit and reproducible
  commands; preserve original failure output and submit matching
  documentation/tests with fixes.
- [D-Robotics developer community](https://developer.d-robotics.cc/) and
  its [user manual](https://developer.d-robotics.cc/information)

Unified code uses the root [LICENSE](LICENSE). Model weights, datasets
and upstream projects retain their respective licenses and provenance.
