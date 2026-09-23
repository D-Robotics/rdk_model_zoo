# ByteTrack model artifacts

<a id="artifacts"></a>
## Artifacts

ByteTrack uses the YOLOv5x detector HBM listed by the S manifest. It does not have a separate neural model:

| target | asset ID | filename | URL | SHA-256 |
|---|---|---|---|---|
| S100 | `s:bytetrack:s100/yolov5x_672x672_nv12.hbm` | `s100/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S100P | `s:bytetrack:s100p/yolov5x_672x672_nv12.hbm` | `s100p/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100p/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S600 | `s:bytetrack:s600/yolov5x_672x672_nv12.hbm` | `s600/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |

The tracker code and CPU dependencies are separate runtime inputs. Publisher checksums are unknown.

<a id="preparation"></a>
## Preparation

From the repository root, choose one target and run the explicit downloader in an environment with network access; this migration did not run it:

```bash
python3 -m samples.vision.bytetrack.model.download \
  --target s100 --output-dir samples/vision/bytetrack/model
```

Success prints the exact asset ID, nested `Saved:` path, and observed digest, and creates `samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm`. Use `s100p` or `s600` for the other rows. The downloader does not fetch the input video.

<a id="accompanying-files"></a>
## Accompanying files

- `../test_data/coco_classes.names` supplies COCO labels for the detector/visualization.
- `../test_data/bus.jpg` is a still image fixture; it is not the missing tracking video.
- `../test_data/readme_img/` contains source reference GIF/PNG images only.
- `../requirements-host.txt` records CPU tracker packages; board SDK is separate.

<a id="local-paths"></a>
## Local paths

The default selection resolves to `samples/vision/bytetrack/model/<target>/yolov5x_672x672_nv12.hbm`. External paths must carry the exact qualified asset ID for the same target. `track_test.mp4` must be separately placed under `test_data/` or passed with `--input`.

<a id="formats-checksums"></a>
## Formats and checksums

All three artifacts are target-relative `.hbm` split-NV12 YOLOv5x deployments at 672x672. Manifest URL and `sha256: null (unknown)` are authoritative; an observed local digest is not publisher authentication.

<a id="license"></a>
## License

Preparation helpers follow Apache-2.0. Model and upstream tracker licenses remain governed by their sources.
