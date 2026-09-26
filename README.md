# RDK Model Zoo

[简体中文](README_cn.md)

RDK Model Zoo provides model preparation, preprocessing, BPU inference, postprocessing and application-validation examples for D-Robotics devices. Each sample README is the shared entry for people and Agents: commands, I/O, source structure, conversion/evaluation and limitations. Inference needs neither an Agent, Node.js nor catalog-publishing tools.

> `develop` is the ongoing X5/S integration branch, not a completed customer migration release. Customer delivery should use the matching published platform branch/tag and its documentation. The unified tree extends beyond the original three pilots; migration, host checks, board tests, conversion and release readiness are tracked separately.

## Start by task

The [sample index](samples/README.md) lists all 44 current unified vision samples. Each guide states its own targets, variants, languages and validation scope.

| Task | Unified entry |
|---|---|
| Detection, segmentation, pose, classification, oriented boxes | [Ultralytics YOLO](samples/vision/ultralytics_yolo/README.md) |
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

Depth, semantic segmentation, point clouds, speech, robotics, LLM/VLA and other pending capabilities remain accessible through the [X5 source entry](platforms/x5/README.md) and [S source entry](platforms/s/README.md). Pending migration does not mean the source capability was deleted. See the [migration ledger](docs/releases/unified-migration/x5-s-migration-map.md).

## Boards, artifacts and environments

| Target | Artifact | Distinction |
|---|---|---|
| RDK X5 | `.bin`, bayes-e | 4GB/8GB memory differs; validation distinguishes board slots |
| RDK S100 | `.hbm`, nash-e | Not interchangeable with S100P/S600 |
| RDK S100P | `.hbm`, nash-m | Only published combinations; no silent S100 fallback |
| RDK S600 | `.hbm`, nash-p | Input geometry/combinations follow the actual artifact |
| RDK X3 | Historical directory | Original documentation/releases retained; outside new X5/S adaptation |

Use the SDK supplied by the matching board image; a module named `hbm_runtime` is not a cross-platform installation package. Sample guides control host/board dependencies, image versions and memory requirements. Common Python dependencies include NumPy, OpenCV, SciPy and PyYAML; tasks do not all share one input protocol or install set. C++ needs board development headers/libraries; conversion needs host training/OE environments.

## Quick start: inspect the integration and run one example

This checkout path is for contributors/reviewers of `develop`, not a substitute for customer release selection. Keep the complete repository and prepare the dependencies documented by the chosen sample in its Python environment. Ordinary inference does not require VLA submodule initialization.

```bash
git clone --branch develop https://github.com/D-Robotics/rdk_model_zoo.git
cd rdk_model_zoo
python3 samples/vision/ultralytics_yolo/runtime/python/main.py --help
python3 samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --list-models
```
Then, on a matching X5, explicitly prepare a model and run detection from the repository root:

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n
python3 samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --model-path samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/rdk-yolov8n.jpg
```
Success prints `[Saved]` and writes `/tmp/rdk-yolov8n.jpg`; the input is bundled in `test_data`. Other boards require matching targets/artifact paths. Without a board use `--dry-run`; host argument resolution is not inference validation. See [YOLO model preparation](samples/vision/ultralytics_yolo/model/README.md) and [runtime instructions](samples/vision/ultralytics_yolo/runtime/python/README.md) for provenance, offline copying, hash limitations and other tasks.

## Read and extend the code

```text
samples/                  # unified task implementations and guides
platforms/{x5,s}/         # retained source material and compatibility entries
platforms/x3/             # historical X3 distribution
docs/release/             # artifact/benchmark facts and target identity
docs/sample-standards/    # README and inference contracts
docs/releases/           # migration ledger, reviews and validation evidence
datasets/                # dataset entry points
utils/                   # compatibility utilities
tools/                   # catalog, contract checks and validation tooling
```
`main.py` owns CLI, files and rendering; task modules own preprocessing, inference, postprocessing and optional `predict`; runner/binding isolate SDK and tensor contracts. `conversion/` and `evaluator/` each have actionable guides. Share mechanisms used by multiple samples while retaining real differences such as OCR vocabularies and DFL/LTRB. A common directory structure does not imply identical audit maturity.

Read [AGENTS.md](AGENTS.md), the [inference contract](docs/sample-standards/inference-contract.md) and [README contract](docs/sample-standards/readme-contract.md) before development. People and Agents use the same native commands, not separate hidden execution paths.

## Data, source-branch resources and validation

- [Canonical release facts](docs/release) hold artifacts and historical measurements; the [platform registry](platforms/README.md) explains source branches, directories, tags and runtimes.
- Dataset preparation: [datasets](datasets), [X5 datasets](platforms/x5/datasets), [S datasets](platforms/s/datasets). Large datasets/models are generally not in Git.
- Retained references: [X5 guidelines](platforms/x5/docs/Model_Zoo_Repository_Guidelines.md), [S Python API](platforms/s/docs/Python_API_User_Guide.md), [S UCP](platforms/s/docs/UCP_User_Guide.md), [TROS](docs/tros/README.md).
- The [migration ledger](docs/releases/unified-migration/x5-s-migration-map.md) and batch reports distinguish implementation, host checks, board tests, independent review and closure; the [current host completion plan](docs/superpowers/plans/2026-09-26-host-completion.md) does not treat pending board tests as passed.

SAM has partial board evidence and must no longer be described as wholly untested. Likewise, one YOLO/classification case or video excerpt cannot certify all targets/variants, dataset accuracy or release readiness. Read exact commits, inputs, artifacts and scope in each sample and evidence record.

## Common questions

**Can a development computer run BPU inference?** Help, inventory, dry-run and host tests work with their dependencies installed; BPU inference needs matching board SDK/hardware.

**Can artifacts be reused across targets?** Renaming files/extensions or changing target does not change march, input layout/dtype or output protocol. A missing published asset remains an explicit gap.

**Do evaluation tables measure the current refactor?** Historical tables retain their original conditions. New code needs corresponding records; host checks, fixed-image comparisons, dataset accuracy and performance are different validations.

## Catalog data (maintainers)

`tools/catalog-publisher` validates platform manifests and produces derived data. Use Node 22.12+ and below 23 as required by package.json; it is not a board-inference dependency. From the repository root:

```bash
npm --prefix tools/catalog-publisher ci
npm --prefix tools/catalog-publisher run check
npm --prefix tools/catalog-publisher run catalog:build
```
Generated `dist/catalog.meta.json` binds `catalog.json` by SHA-256; CI uploads data artifacts. Catalog data, website publication and board samples are separate deliveries. Migration does not automatically publish new models or rewrite historical tags.

## Community, contribution and license

Use repository Issues with target, image/SDK, model reference, commit and reproducible commands. Preserve original failure output and submit matching documentation/tests with fixes. Source branches retain [community resources](platforms/x5/README.md#community--contribution) and platform-specific guidance.

Unified code uses the root [LICENSE](LICENSE); distributions retain [X5 LICENSE](platforms/x5/LICENSE) and [S LICENSE](platforms/s/LICENSE). Upstream X3 supplied no license file and none is invented here. Model weights, datasets and upstream projects retain their respective licenses and provenance.
