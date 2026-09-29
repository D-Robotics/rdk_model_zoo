# YOLOWorld model assets

<a id="artifacts"></a>
## Artifacts

The active release manifest `docs/release/x5/models.yaml` contains one model
asset: `x5:yoloworld:yolo_world.bin`, format `.bin`, published URL under the X5
archive, and `sha256: null`. A null publisher digest is unknown; a local
observed digest does not certify origin. The offline vocabulary companion is
`test_data/offline_vocabulary_embeddings.json`, not a manifest model asset.

<a id="preparation"></a>
## Preparation

```bash
bash samples/vision/yoloworld/model/download.sh --target x5
```

`download.py` calls the repository asset helper, writes atomically, refuses an
invalid existing file, prints the observed SHA-256, and states when publisher
SHA-256 is unknown. `runtime/python/main.py` never calls it automatically.
Manual transfer is allowed only when the file is identified by the exact asset
ID and the vocabulary is copied from the same reviewed source; do not rename an
unidentified model into this path.

<a id="accompanying-files"></a>
## Accompanying files

The source fixture image is `../test_data/dog.jpeg` and the source vocabulary is
`../test_data/offline_vocabulary_embeddings.json`. The JSON must contain finite
F32 vectors of width 512 for every prompt; it supplies text embeddings and ID
mapping. It must not be replaced with COCO label names.

<a id="local-paths"></a>
## Local paths, formats and checksums

The default model path is `model/yolo_world.bin`; explicit `--model-path` needs
`--asset-id x5:yoloworld:yolo_world.bin`. The compiled artifact is X5 `.bin`.
The manifest publisher hash is unknown, so no checksum is claimed here. The
companion JSON is UTF-8 JSON and has no publisher digest in the manifest.

## Entry points

`download.py`, `download.sh`, and compatibility alias `download_model.sh` are
explicit preparation entry points. None is invoked by the runtime.

<a id="formats-checksums"></a>
## Formats & Checksums

The artifact format is `.bin`; the manifest records `sha256: null (unknown)` and no publisher digest is claimed. Both 2026-09-24 board comparisons (X5 8GB and X5 4GB) observed SHA-256 `bc8fd742319c26fb550123a4a1433c9e7bd6e0ccb6a17866d3aedb96472f6239` ([8GB evidence](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-python-comparison/), [4GB evidence](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-other-x5-variants/)); an observed digest identifies the bytes used in those runs, not publisher authentication.
