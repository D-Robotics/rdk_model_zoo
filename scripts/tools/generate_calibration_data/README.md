English | [简体中文](README_cn.md)

# Generate a calibration dataset

## Method 1

Use the `02_preprocess.sh` script from a conversion example in the OpenExplore package to generate the calibration dataset.
See the [toolchain development documentation](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview) for how to obtain OpenExplore.

If an error such as `Can't reshape 1354752 in (1,3,640,640)` occurs, edit the resolution in the adjacent `preprocess.py` to match the ONNX model you are converting. Delete all previously generated calibration data, then run the `02_preprocess.sh` script again to regenerate it.
The example currently reads calibration images from `../../../01_common/calibration data/coco` and writes the generated data to `./calibration_data_rgb_f32`.

## Method 2

Prepare calibration data using libraries such as OpenCV and NumPy. Except for channel mean subtraction and normalization configured in the YAML file, all preprocessing must match the training pipeline.
