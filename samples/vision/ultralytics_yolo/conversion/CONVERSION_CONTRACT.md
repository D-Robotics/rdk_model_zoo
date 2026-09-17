# Conversion contract and migration map

This note records what was consolidated for representative detection. It is a
source map, not a claim that a local host run has compiled a model.

## Old symbols to maintained symbols

| Previous path/symbol | Maintained path/symbol | Meaning |
| --- | --- | --- |
| `platforms/x5/.../ultralytics_yolo/conversion/mapper.py` | `conversion/mapper.py` with `--platform x5` | Historical X5 command and defaults |
| `platforms/s/.../ultralytics_yolo/conversion/mapper.py` | `conversion/mapper.py` with `--platform s100/s100p/s600` | Historical S command and march selection |
| `mapper_x5.main` image/ONNX/YAML/compiler body | `workflow.run_conversion(..., x5_toolchain())` | X5 orchestration |
| `mapper_s.main` / `run_nash` body | `workflow.run_conversion(..., s_toolchain(march))` | S orchestration |
| `yolo26/mapper_x5.main` | `workflow.run_conversion(..., x5_toolchain())` with `calibration_data` default | YOLO26 X5 compatibility path |
| `yolo26/mapper_s.main` | `workflow.run_conversion(..., s_toolchain(march))` | YOLO26 S compatibility path |
| `resolve_path` copies | `workflow.resolve_path` | Invocation-relative path resolution |
| ONNX Runtime input checks | `workflow.inspect_onnx` | One static float NCHW input |
| Per-mapper image listing/sampling | `workflow.calibration_images` and `select_calibration_images` | Stable image pool and sampling |
| Per-mapper OpenCV/NumPy loops | `workflow.prepare_calibration` | RGB resize, NCHW float32, target file format |
| Per-mapper YAML f-strings | `workflow.render_config` | Target-specific compiler configuration |
| `os.system("hb_mapper ...")` / `os.system("hb_compile ...")` | `workflow.run_conversion` command runner | Checked compiler invocation, artifact and log move |
| Generic export opset literals | `workflow.export_defaults` | X5/S opset and YOLO26 simplify policy |
| `yolo26/export_yolo26_detect_bpu.py` platform ternary | `workflow.export_defaults("...", "yolo26")` | Direct-LTRB detector export policy |

The platform files remain executable forwarding adapters. They keep old import
and command paths; they do not carry another calibration or compiler body.
The dispatcher still passes all existing family/task arguments through, so a
legacy segmentation or pose invocation is not routed to this detection
contract.

## Target protocol that remains separate

| Property | X5 | S100/S100P/S600 |
| --- | --- | --- |
| Compiler | `hb_mapper makertbin --config config.yaml --model-type onnx` | `hb_compile --config config.yaml` |
| Compiler march | `bayes-e` | `nash-e` / `nash-m` / `nash-p` |
| Calibration file | raw float32 RGB/NCHW `*.rgbchw` | float32 NCHW `*.npy`, divided by 255 |
| Quantization extension | Softmax int8 optimization; int16 adds all-node int16 | `quant_config` for int16 |
| Compiler extras | none in the reviewed config | `input_no_padding`, `output_no_padding` |
| Artifact | `<stem>_bayese_<WxH>_nv12.bin` | `<stem>_<march-without-dash>_<WxH>_nv12.hbm` |
| Compiler log | `hb_mapper_makertbin.log` | `hb_compile.log` |

The shared code stops at these protocol adapters. It does not pretend that X5
and S can consume each other's calibration files or artifact formats.

## Reviewed scope and limitations

* Generic representative detection is the three-level DFL protocol used by
  the reviewed YOLOv8n/YOLO11-style artifacts (`reg=16`).
* YOLO26 detection is the separate three-level direct-LTRB protocol. Its ONNX
  exporter and runtime binding remain family-specific; only the host
  calibration/compiler workflow is shared.
* The workflow requires one static float32 rank-4 ONNX input and a local image
  pool. Dynamic shapes, multiple inputs, custom quantized input metadata, and
  unreviewed output protocols stop before compilation.
* Real export and OpenExplore compilation were not executed in this local
  merge. The 59 host tests cover planning, config rendering, path safety,
  entry points, and runtime contracts. Board results in the release evidence
  concern existing runtime artifacts and do not certify a newly compiled local
  artifact.
* No new segmentation, pose, classification, OBB, model family, or model
  scale support is introduced by this consolidation.
