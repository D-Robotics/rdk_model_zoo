English | [简体中文](./README_cn.md)

# YOLOE11-Seg Model Conversion Guide

This directory records the conversion reference materials, YAML configuration, and OE toolchain notes for the YOLOE-11/v8 Instance Segmentation Prompt-Free models.

## Tool Requirements

```bash
D-Robotics OpenExplore Version: >= 3.0.31
Ultralytics YOLO Version: >= 8.3.0
```

## Supported Models

```text
YOLOE-11 Instance Segmentation Prompt Free
YOLOE-v8 Instance Segmentation Prompt Free
```

## Export to ONNX

Use the export script provided by RDK Model Zoo to prepare the ONNX model before conversion. The script automatically replaces relevant modules in an equivalent manner and does not require retraining:

```text
https://github.com/D-Robotics/rdk_model_zoo/blob/main/demos/Seg/YOLOE-11-Seg-Prompt-Free/YOLOE-11-Seg-Prompt-Free_YUV420SP/cauchy_yoloe11segPF_export.py
```

Other conversion steps are essentially the same as those for Ultralytics YOLO Seg, except that the number of classes changes from 80 to 4585.

## Reference YAML

The reference quantization configuration for YOLOE-11/v8 Seg is:

```yaml
# config_ultralytics_YOLOE_Seg_YUV420SP_NV12.yaml
# (see conversion/ directory)
```

The YAML file `config_ultralytics_YOLOE_Seg_YUV420SP_NV12.yaml` is provided in this directory.

## Reference Compilation Logs

The following reference logs are provided in this directory for traceability:

| File | Description |
|---|---|
| `hb_combine_yoloe_11s_seg.txt` | hb_combine log for YOLOE-11s Seg |
| `hb_combine_yoloe_v8s_seg.txt` | hb_combine log for YOLOE-v8s Seg |
| `hb_model_info_yoloe_11s_seg.txt` | hb_model_info output for YOLOE-11s Seg |
| `hb_model_info_yoloe_v8s_seg.txt` | hb_model_info output for YOLOE-v8s Seg |
| `hrt_model_exec_model_info_yoloe_v8s_seg.txt` | hrt_model_exec model info for YOLOE-v8s Seg |
| `hrt_model_exec_model_info_yolow_11s_seg.txt` | hrt_model_exec model info for YOLOE-11s Seg |

## Class Names List

The file `thu_yoloe_prompt_free_names.list` in the `conversion/` directory contains all 4585 class names supported by the Prompt-Free models.

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore environment. Model conversion is not intended to run on the board.

- OE resource entry point: <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
