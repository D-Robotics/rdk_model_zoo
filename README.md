# RDK Model Zoo

English | [简体中文](README_cn.md)

RDK Model Zoo provides model deployment examples for D-Robotics BPU devices. Each sample includes model preparation, board inference, readable source, and its existing conversion and evaluation workflow. Inference uses Python and the runtime supplied with the board image; neither an Agent nor Node.js is required.

## Start with a sample

This integration covers three representative tasks. Follow a sample's guide for setup and execution. Board results below apply to the named artifacts and fixed inputs, not every model in a family.

| Task | Sample | Representative models and boards | Development |
| --- | --- | --- | --- |
| Object detection | [Ultralytics YOLO](samples/vision/ultralytics_yolo/README.md) | YOLOv8n and YOLO26n; X5 8GB/4GB, S100, S100P, S600 | [Conversion](samples/vision/ultralytics_yolo/conversion/README.md) · [Evaluation](samples/vision/ultralytics_yolo/evaluator/README.md) |
| Image classification | [ResNet](samples/vision/resnet/README.md) | ResNet18; X5 8GB/4GB, S100, S600 | [Conversion](samples/vision/resnet/conversion/README.md) · [Evaluation](samples/vision/resnet/evaluator/README.md) |
| Text detection and recognition | [PaddleOCR](samples/vision/paddle_ocr/README.md) | X5 PP-OCRv3 and S100 PP-OCRv6; detector and recognizer composition | [Conversion](samples/vision/paddle_ocr/conversion/README.md) · [Evaluation](samples/vision/paddle_ocr/evaluator/README.md) |

The September 17 source revision was retested on both X5 boards, S100 and S100P where artifacts are available. S600 results are from September 16; connection recovery and revision retesting remain pending.

Keep a complete checkout on your board. From the repository root, inspect available arguments and models without loading the BPU or downloading artifacts:

```bash
python3 samples/vision/ultralytics_yolo/runtime/python/main.py --help
python3 samples/vision/resnet/runtime/python/main.py --list-models --target x5
python3 samples/vision/paddle_ocr/runtime/python/main.py --list-models --target x5
```

Then follow the sample README to prepare user-space dependencies and a model for your board, and run inference. X5 `.bin` and S-series `.hbm` artifacts are not interchangeable; S100, S100P and S600 also require their own artifacts. Do not replace the board runtime with a generic PyPI package.

## Read and extend the code

```text
samples/
├── _shared/                  # Target identity, artifact access, image-byte conversion
└── vision/
    ├── ultralytics_yolo/      # Detection pipeline and DFL / LTRB decoding
    ├── resnet/               # Classification pipeline and Top-K results
    └── paddle_ocr/           # Detection → crops → recognition → CTC decoding
```

Within each sample, `runtime/python/main.py` handles the command line, the task module owns inference flow, `model_runner.py` calls the model, and `model_binding.py` validates artifact and tensor contracts. `conversion/` generates models on a development host, `evaluator/` evaluates results, and `test_data/` holds minimal inputs. Sample READMEs show callable APIs and explain the files.

Shared algorithms have one maintenance location; necessary model differences remain local. The two OCR generations retain different vocabularies, and YOLO DFL and direct LTRB retain different decoders.

## Platform sources, datasets and historical releases

Models outside this batch remain available through their platform directories. X3 is preserved as historical content and is not part of this X5/S integration.

| Platform | Sources and usage | Release facts |
| --- | --- | --- |
| X5 | [Platform README](platforms/x5/README.md) | [Model and benchmark manifests](platforms/x5/docs/release) |
| S100 / S100P / S600 | [Platform README](platforms/s/README.md) | [Model and benchmark manifests](platforms/s/docs/release) |
| X3 (historical) | [Platform README](platforms/x3/README.md) | [Historical manifests](platforms/x3/release) |

Dataset preparation remains in [X5 datasets](platforms/x5/datasets) and [S datasets](platforms/s/datasets). Existing manifests remain the source of model URLs, identities and historical measurements; integrating source does not redefine published artifacts.

Historical tags retain their release-time layout. Read their documentation at the matching tag. The [platform registry guide](platforms/README.md) describes historical branches; local integration does not switch the default branch or publish a release.

The [validation record](docs/releases/unified-migration/2026-09-16-p2-validation.md) describes previous runtime checks. The [integration revision](docs/superpowers/plans/2026-09-17-representative-integration.md) tracks source consolidation, documentation, conversion and regression checks separately.

## Catalog Data

`tools/catalog-publisher` reads the three platform manifests, validates them against their own schemas, applies the documented normalisation and errata layers, and emits a versioned data package:

```bash
cd tools/catalog-publisher
npm ci
npm run check           # validate sources, run tests, typecheck, rebuild, verify
npm run catalog:build   # write dist/catalog.json and dist/catalog.meta.json
```

`dist/catalog.meta.json` pins the exact bytes of `catalog.json` with a SHA256 digest; consumers verify that digest before using the data. The package is uploaded as a build artifact by `.github/workflows/model-catalog-data.yml`. No branch of this repository deploys a website.

## License

Each platform distribution carries its own license file — see [`platforms/x5/LICENSE`](./platforms/x5/LICENSE) and [`platforms/s/LICENSE`](./platforms/s/LICENSE). Upstream X3 published no license file and none has been added here.
