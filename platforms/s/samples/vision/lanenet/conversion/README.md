English | [简体中文](./README_cn.md)

# LaneNet Model Conversion Guide

This directory provides the quantization YAML configuration and full conversion workflow notes for LaneNet on RDK S100.

## Environment Requirements

```bash
python >= 3.6
torch >= 1.2
torchvision >= 0.4.0
numpy >= 1.7
opencv-python
pandas
matplotlib
```

Install dependencies:

```bash
pip install torch torchvision numpy opencv-python pandas matplotlib
```

## Export ONNX

Download the pre-trained model:

```bash
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/best_model.pth
```

Export the ONNX model using the `test.py` script:

```bash
python test.py --img source/data/source_image/input.jpg --model best_model.pth
```

## Dataset Preparation

Download the [TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3) dataset and convert it to `.npy` format. Modify the `dataset_dir` parameter in the calibration script to point to the TuSimple directory:

```bash
python get_calibration_data.py
```

## Model Compilation

Run the OE toolchain Docker, mount the directory to the LaneNet development folder, and run:

```bash
hb_compile -c source/yaml/config.yaml
```

The YAML configuration file `config.yaml` is provided in this directory.

## Performance Reference

Tested on RDK S100 platform:

```bash
hrt_model_exec perf --model_file lanenet256x512.hbm
```

```text
Frame count: 200
Average latency: 14.245 ms
FPS: 69.894
```

## Accuracy Reference

After quantization, all three outputs maintain high cosine similarity. Reference result image:

![Quantization accuracy](../test_data/readme_img/result.jpg)

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore environment.

- OE resource entry point: <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
