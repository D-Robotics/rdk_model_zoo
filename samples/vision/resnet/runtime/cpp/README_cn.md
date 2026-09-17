# C++ 运行时

这是合并后的 S 系列 ResNet18 原生运行时。它保留审计过的 S18
`hbDNNInferV2` 流程、图像预处理、NV12 张量生成和 Top-K 输出代码，统一维护
在 canonical sample 中。S18 旧 C++ 目录只是薄 CMake 兼容配置路径并添加此
目标，不再维护第二份实现。

## 板端前置条件

请在匹配的 RDK S100 或 S600 板端构建运行。环境中需要已经提供：

* CMake 和 C++17 编译器；
* OpenCV 开发头文件和库；
* `gflags` 与 `fmt` 开发库；
* `/usr/hobot/include` 下的 Horizon DNN 头文件，以及
  `/usr/hobot/lib` 下包括 `hbDNN`、`hbucp` 的库。

CMake 读取 `/sys/class/boardinfo/soc_name` 并定义旧源代码使用的 SoC 宏。
启动器不会安装系统包、修改 SDK 或下载模型。

## 准备并运行

在仓库根目录显式准备 S100 制品，然后运行：

```bash
bash samples/vision/resnet/model/download.sh s100
bash samples/vision/resnet/runtime/cpp/run.sh
```

启动器检查模型、随附的 `zebra_cls.jpg` 和
`platforms/s/datasets/imagenet/imagenet_classes.names`，配置 CMake，构建
`resnet18` 并将这些路径传给二进制。S600 使用 S600 制品和独立 build 目录：

```bash
bash samples/vision/resnet/model/download.sh s600
MODEL_PATH="$PWD/samples/vision/resnet/model/s600/resnet18_224x224_nv12.hbm" \
BUILD_DIR="$PWD/samples/vision/resnet/runtime/cpp/build-s600" \
bash samples/vision/resnet/runtime/cpp/run.sh
```

启动器支持透传原生参数，例如：

```bash
bash samples/vision/resnet/runtime/cpp/run.sh \
  --model_path /opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm \
  --test_img /tmp/zebra_cls.jpg \
  --label_file /tmp/imagenet_classes.names \
  --top_k 5
```

覆盖值会在文件检查前解析，并继续传给二进制。路径必须在板端存在；文件
缺失不会触发下载。

## 原生流程

`main.cpp` 创建 `Resnet18`、加载 HBM，通过 `pre_process` 将 BGR 图片转换，
调用 `hbDNNInferV2`，再用 `post_process` 解码输出。C++ 运行时接收 S 系列
Y 和 UV 输入张量，并使用逐行 ImageNet 标签文件打印 Top-K。源代码使用
`platforms/s/utils/c_utils` 中已有的工具实现，canonical CMake 引用其头文件
和源文件。

默认二进制参数保留旧名称和默认值：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model_path` | 板端 S100 `/opt/hobot/model/s100/basic/...` 或 S600 对应路径 | HBM 模型路径 |
| `--test_img` | 从 build 目录计算的 `../../../test_data/zebra_cls.jpg` | BGR 测试图 |
| `--label_file` | 仓库 S ImageNet 标签 | 每行一个标签 |
| `--top_k` | `5` | 打印类别数量 |

启动器传入绝对 sample 路径，因此可在任意工作目录运行；直接从旧 build
布局调用二进制时仍可使用二进制自身的默认值。

## 兼容构建和故障排查

验证旧配置路径仍选择 canonical 目标：

```bash
cmake -S platforms/s/samples/vision/resnet18/runtime/cpp \
  -B /tmp/resnet18-legacy-build
cmake --build /tmp/resnet18-legacy-build --parallel
```

旧 `runtime/cpp/run.sh` 会委托 canonical 启动器，并保留旧模型、图片和标签
位置。如果 CMake 无法读取 SoC，说明板卡身份文件不可用；如果缺少头文件或
库，请按平台流程准备板端镜像/工具链后重试，sample 不会执行安装。如果二进制
报告模型或输入失败，先核对制品目标、张量协议、模型元数据、图片路径和标签
路径，再修改源码。

每次原生评估都应记录板卡身份、制品引用、完整构建命令和 Top-K 输出。S600
无法连接或缺少制品时记为 `not-run`，不能当作 S100 成功结果。
