# ByteTrack conversion

<a id="source-model"></a>
## Source model

ByteTrack itself is post-processing. The only neural artifact is the upstream YOLOv5x detector HBM, with three target-specific manifest rows. The fixed S source does not provide a separate tracker export or checkpoint.

<a id="toolchain-targets"></a>
## Toolchain and targets

S100 uses the `s100/yolov5x_672x672_nv12.hbm` Nash-e row, S100P uses its own `s100p/...` row, and S600 uses its own `s600/...` row. Conversion requires the RDK S OpenExplorer environment on an x86 Linux host; source resources are [OE overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview) and [toolchain manual](https://toolchain.d-robotics.cc/). No S YAML or exporter is in this sample.

<a id="export"></a>
## Export

There is no ByteTrack export: the tracker has no neural graph. To rebuild the detector, use the upstream YOLOv5 source/export procedure and produce a 672x672 detector with split Y/UV inputs and three output heads. The fixed source supplies no pinned checkpoint/export script; no export was run.

<a id="calibration"></a>
## Calibration

Calibration belongs to the upstream detector conversion. No calibration directory or producer is supplied here. The still images in `test_data` are not a representative calibration set.

<a id="compile"></a>
## Compile

After an external ONNX/checkpoint, target-specific YAML, and calibration set exist, run the selected OE compiler commands supplied by that environment. No exact source command/config is available for S in this fixed tree, so a generic `hb_mapper` line would not be a reproducible recipe and is intentionally omitted. The result must bind the three target-relative HBM asset IDs and S split-NV12 metadata.

<a id="validation"></a>
## Post-conversion validation

Validate detector metadata with the S YOLOv5 runtime, then run the tracker on a prepared video. Compare complete detector tensors and track IDs using `evaluator/compare.py`. No export, compile, board, or video validation was run.

<a id="artifacts"></a>
## Artifacts

The three external outputs are the manifest HBM rows in `model/README.md`; ByteTrack adds no compiled tracker artifact. `TRACKER_SOURCE_MAP.json` records source tracker file hashes and the one relative-import adaptation.

<a id="known-gaps"></a>
## Known gaps

- No S export script, checkpoint pin, YAML, calibration producer, or compiler log is present.
- The source video is absent and has only an explicit archive URL in the customer docs.
- Conversion and board validation are `not-run`; all publisher SHA-256 values are unknown.
