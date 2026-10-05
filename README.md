# RDK Model Zoo

[简体中文](README_cn.md)

RDK Model Zoo provides model preparation, preprocessing, BPU inference, postprocessing and application-validation examples for D-Robotics devices. Each sample README is the shared entry for people and Agents: commands, I/O, source structure, conversion/evaluation and limitations. Inference needs neither an Agent, Node.js nor catalog-publishing tools.

> `develop` is the ongoing X5/S integration branch, not a completed customer migration release. Customer delivery should use the matching published platform branch/tag and its documentation. The unified tree extends beyond the original three pilots; migration, host checks, board tests, conversion and release readiness are tracked separately.


## Start by task

The [sample index](samples/README.md) lists 51 unified samples: 45 vision, three speech, one robotics policy and two LLM samples. Each guide states its own targets, variants, languages and validation scope.

| Task | Unified entry |
|---|---|
| Detection, segmentation, pose, classification, oriented boxes | [Ultralytics YOLO](samples/vision/ultralytics_yolo/README.md) |
| Prompt-free instance segmentation | [YOLOE](samples/vision/yoloe/README.md) |
| Speech recognition | [ASR](samples/speech/asr/README.md), [Paraformer](samples/speech/paraformer/README.md) |
| Offline robotics policy | [HIMLoco](samples/robotics/himloco/README.md): six-frame observations to actions, without robot control |
| Vision-language model (migration in progress) | [Gemma4-E2B](samples/llm/gemma4-e2b/README.md): native chat, HTTP, single-shot inference and verification tools |
| Text generation (migration in progress) | [MiniCPM5-2B](samples/llm/minicpm5-2b/README.md): S100/S100P OELLM 1.0.0 and S600 OELLM 2.0 beta native entries |
| Keyword spotting | [KWS](samples/speech/kws/README.md) |
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

Capabilities not yet unified are not carried in this tree anymore: the historical
`platforms/` copies were removed (2026-10-01) and stay reachable through the pinned
commit `d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d` and the delivery branches
(`rdk_x5`, `rdk_s`). Pending migration does not mean the source capability was
deleted. See the [removal record](docs/migration/2026-09-30-model-examples.md) and the
[migration ledger](docs/releases/unified-migration/x5-s-migration-map.md).

ACT/Pi0 are integrated as complete pinned Git submodules, separately from the 51 in-repository samples above. See the [VLA guide](samples/vla/README.md): S100 and S600 ACT use different source versions, Pi0 targets S600, and model resources are operator-supplied. Robot control was not run in this migration.

## Boards, artifacts and environments

| Target | Artifact | Distinction |
|---|---|---|
| RDK X5 | `.bin`, bayes-e | 4GB/8GB memory differs; validation distinguishes board slots |
| RDK S100 | `.hbm`, nash-e | Not interchangeable with S100P/S600 |
| RDK S100P | `.hbm`, nash-m | Only published combinations; no silent S100 fallback |
| RDK S600 | `.hbm`, nash-p | Input geometry/combinations follow the actual artifact |
| RDK X3 | Historical | Removed from the active tree; reachable via the pinned commit and the `rdk_x3` branch; never a new adaptation target |

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
docs/release/             # artifact/benchmark facts and target identity
docs/sample-standards/    # README and inference contracts
docs/architecture/        # readable runtime architecture (all 51 in-repo samples)
docs/migration/           # old-to-new interface mappings
docs/releases/           # migration ledger, reviews and validation evidence
datasets/                # dataset entry points
utils/                   # compatibility utilities
tools/                   # catalog, contract checks and validation tooling
```
`main.py` owns CLI, files and rendering; task modules own preprocessing, inference, postprocessing and the public `predict`; runner/binding isolate SDK and tensor contracts. `conversion/` and `evaluator/` each have actionable guides. Share mechanisms used by multiple samples while retaining real differences such as OCR vocabularies and DFL/LTRB. A common directory structure does not imply identical audit maturity.

Read [AGENTS.md](AGENTS.md), the [inference contract](docs/sample-standards/inference-contract.md) and [README contract](docs/sample-standards/readme-contract.md) before development. People and Agents use the same native commands, not separate hidden execution paths. The readable model example pattern (thin `main.py` that visibly constructs the model and calls `predict`, a local named model class with the real `preprocess`/`infer`/`postprocess`/`predict` chain — the older `pre_process`/`forward`/`post_process` names stay compatibility delegates — sample-local `cli.py` helpers, and a thin SDK session) is described in [docs/architecture/model-examples.md](docs/architecture/model-examples.md), with ResNet ([classify.py](samples/vision/resnet/runtime/python/classify.py)) and YOLO detection ([detect.py](samples/vision/ultralytics_yolo/runtime/python/detect.py)) as the reference implementations. Since 2026-10-05 the pattern applies to all 51 in-repo samples; ACT/Pi0 remain pinned VLA gitlinks outside this scope, and the two `samples/llm` samples keep their native generate/stream/reset C++ interfaces (no fabricated Python runtime). The per-sample mapping is [2026-10-05-all-sample-readable-runtime.md](docs/migration/2026-10-05-all-sample-readable-runtime.md) with status rows in [2026-10-05-all-sample-coverage.json](docs/releases/unified-migration/2026-10-05-all-sample-coverage.json); the rollout is accepted for source architecture and host checks, as recorded in the [Codex review](docs/releases/unified-migration/2026-10-05-all-sample-codex-review.md). Real board inference and export/toolchain compilation were not rerun this round; this local development work is not a migration release.

## Data, source-branch resources and validation

- [Canonical release facts](docs/release) hold artifacts and historical measurements. The former `platforms/registry.json` statement of the delivery-branch layouts was removed with the `platforms/` tree; branch/tag facts now live in the [removal record](docs/migration/2026-09-30-model-examples.md).
- Source-branch online resources, inherited from the archived X5 (`ac11571`) and S (`380e1a2`) root guides: the [online model catalog](https://d-robotics.github.io/rdk_model_zoo/), [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues), the [D-Robotics developer community](https://developer.d-robotics.cc/) and its [user manual](https://developer.d-robotics.cc/information). They reference the delivery branches' published material; browsing the published catalog does not certify this integration branch, and no live link status is claimed here. Legacy demo material stays archived — [`rdk_x5_legacy`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5_legacy) for X5 and the separate [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) repository for S — and is not an adaptation target.
- Dataset preparation: [datasets](datasets). Large datasets/models are generally not in Git; the former per-platform dataset trees went with the `platforms/` removal.
- Retained references on the delivery branches: [X5 guidelines](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/docs/Model_Zoo_Repository_Guidelines.md), [S Python API](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_s/docs/Python_API_User_Guide.md), [S UCP](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_s/docs/UCP_User_Guide.md), [TROS](docs/tros/README.md) in this tree.
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

Use [repository Issues](https://github.com/D-Robotics/rdk_model_zoo/issues) with target, image/SDK, model reference, commit and reproducible commands. Preserve original failure output and submit matching documentation/tests with fixes. The delivery branches retain [community resources](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/README.md#community--contribution) and platform-specific guidance.

Unified code uses the root [LICENSE](LICENSE); the delivery branches carry their own X5/S license files (historical `platforms/{x5,s}/LICENSE`, reachable through the pinned commit). Upstream X3 supplied no license file and none is invented here. Model weights, datasets and upstream projects retain their respective licenses and provenance.
