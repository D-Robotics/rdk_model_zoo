English | [简体中文](./README_cn.md)

# ResNet18 C++ Runtime

This sample builds and runs a C++ ResNet18 image classification executable with
the D-Robotics DNN runtime.

## Dependencies

```bash
sudo apt update
sudo apt install -y libgflags-dev
```

## Directory Structure

```text
.
|-- CMakeLists.txt
|-- README.md
|-- README_cn.md
|-- inc
|   `-- resnet18.hpp
|-- run.sh
`-- src
    |-- main.cpp
    `-- resnet18.cpp
```

## Build

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Quick Run

```bash
bash run.sh
```

The script downloads the model through `../../../model/download_model.sh`,
builds the executable, and runs with the sample-local model path. It defaults
to the S100 model; for RDK S600, run
`bash ../../../model/download_model.sh s600` first and override `--model_path`
to `../../../model/s600/resnet18_224x224_nv12.hbm`.

## Direct Run

Run from `runtime/cpp/build`:

```bash
./resnet18 \
  --model_path ../../../model/s100/resnet18_224x224_nv12.hbm \
  --test_img ../../../test_data/zebra_cls.jpg \
  --label_file ../../../../../../datasets/imagenet/imagenet_classes.names \
  --top_k 5
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--model_path` | HBM model path | `../../../model/s100/resnet18_224x224_nv12.hbm` |
| `--test_img` | Input image path | `../../../test_data/zebra_cls.jpg` |
| `--label_file` | Line-wise ImageNet label file | `../../../../../../datasets/imagenet/imagenet_classes.names` |
| `--top_k` | Number of classification results to print | `5` |

Expected result for `zebra_cls.jpg`:

```text
Top-5 Classification Results:
  [0] zebra: ...
```
