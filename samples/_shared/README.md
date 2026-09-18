# Shared sample utilities

These modules serve ResNet, Ultralytics YOLO and PaddleOCR. They do not define a global
model runtime or a global command.

`platforms.py` reads concrete identity aliases from `platforms/registry.json`.
It checks boardinfo first, then X5's socinfo, then the observed device-tree
model. Exact `X5U` was observed on the two test X5 boards; `X5H` and `X5M`
are user-supplied aliases, not additional tested hardware. Explicit selection
is sufficient for preparation; local inference also needs matching identity.

`assets.py` reads the existing `platforms/{x5,s}/docs/release/models.yaml`.
The installed dependency is PyYAML (no Node/publisher or SDK dependency).
Existing manifests have sample IDs plus filenames, without standalone asset
IDs. The new reference `group:sample:filename` qualifies those existing fields;
it does not rename them or claim a new published ID. For example:

```text
x5:resnet:resnet18_224x224_nv12.bin
x5:ultralytics_yolo:yolov8n_detect_bayese_640x640_nv12.bin
s:ultralytics_yolo:nash-p/yolov8n_detect_nashp_640x640_nv12.hbm
```

Selection policy and input/output contracts remain in each Sample. The reader
does not infer S100P support from an `s` data group. URL and expected SHA-256
come directly from the manifest, and absent publisher hashes remain unknown.
`download_asset` is an explicit preparation operation, never called by a task
class. It writes and checks a temporary file before atomic installation and
does not overwrite existing files; existing known hashes are checked too.

Host tests: `python -m unittest discover -s samples/_shared/tests`.
Synthetic data and failure injection do not certify board inference.

## Image bytes

`image.py:bgr_to_nv12_planes` is the single OpenCV I420 → interleaved NV12
implementation used by all three samples. It accepts an even-sized uint8 BGR
image and returns contiguous uint8 `(1,H,W,1)` Y and `(1,H/2,W/2,2)` UV arrays.
The sample adapters preserve their existing validation and error behavior.
Resize/interpolation, packed-versus-split transport and tensor naming remain
sample-local because their contracts differ. NumPy/OpenCV imports are lazy.

The extraction preserves the operation order and chroma byte ordering from
all three implementations. Known-color and noncontiguous-input tests cover
the byte contract; sample tests and affected-board comparisons cover consumers.
