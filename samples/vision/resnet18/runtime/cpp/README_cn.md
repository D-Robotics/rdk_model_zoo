[English](./README.md) | 简体中文

# ResNet18 C++ 运行示例

本示例使用地瓜机器人 DNN runtime 构建并运行 ResNet18 图像分类 C++ 可执行文件。

## 依赖

```bash
sudo apt update
sudo apt install -y libgflags-dev
```

## 目录结构

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

## 构建

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## 快速运行

```bash
bash run.sh
```

脚本会通过 `../../../model/download_model.sh` 下载模型，构建可执行文件，并使用
sample 内模型路径运行。默认下载 S100 模型，RDK S600 用户请先执行
`bash ../../../model/download_model.sh s600`，并将 `--model_path` 改为
`../../../model/s600/resnet18_224x224_nv12.hbm`。

## 直接运行

在 `runtime/cpp/build` 目录运行：

```bash
./resnet18 \
  --model_path ../../../model/s100/resnet18_224x224_nv12.hbm \
  --test_img ../../../test_data/zebra_cls.jpg \
  --label_file ../../../../../../datasets/imagenet/imagenet_classes.names \
  --top_k 5
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model_path` | HBM 模型路径 | `../../../model/s100/resnet18_224x224_nv12.hbm` |
| `--test_img` | 输入图片路径 | `../../../test_data/zebra_cls.jpg` |
| `--label_file` | 按行排列的 ImageNet 标签文件 | `../../../../../../datasets/imagenet/imagenet_classes.names` |
| `--top_k` | 打印的分类结果数量 | `5` |

`zebra_cls.jpg` 的预期结果：

```text
Top-5 Classification Results:
  [0] zebra: ...
```
