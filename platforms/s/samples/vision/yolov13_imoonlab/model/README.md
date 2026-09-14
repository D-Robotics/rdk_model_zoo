English | [简体中文](./README_cn.md)

# Model Files

This directory stores the reference `.hbm` models and the download script for YOLOv13 iMoonLab Detect.

## Download Models

```bash
cd model
bash download_model.sh
```

The script downloads the reference models into `./s100/`.

## Reference Model List

The following model URLs come from the original YOLOv13 iMoonLab demo under `source/reference_hbm_models/README.md`:

- `yolo13n_detect_nashe_640x640_nv12.hbm`
- `yolo13s_detect_nashe_640x640_nv12.hbm`
- `yolo13l_detect_nashe_640x640_nv12.hbm`
- `yolo13x_detect_nashe_640x640_nv12.hbm`

## Notes

- The currently published reference models in this sample are `nashe` variants.
- The input format is NV12, and the runtime feeds two input tensors: Y plane and UV plane.
- The Python runtime uses `yolo13n_detect_nashe_640x640_nv12.hbm` by default.

## License

This directory follows the repository top-level `LICENSE`.
