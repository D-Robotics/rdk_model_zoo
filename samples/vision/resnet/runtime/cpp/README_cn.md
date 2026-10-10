[English](README.md) | 简体中文

# ResNet18 C++ 运行时（S 系列）

S 系列 ResNet18 原生运行时：`hbDNNInferV2` 推理流程、图像预处理、
NV12 张量构建与 Top-K 输出，基于共享的 `utils/c_utils`
源码构建。

<a id="overview"></a>
## C++ 推理

本目录提供C++ 推理所需的程序与操作说明。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── inc/
│   ├── classify.hpp  # Resnet18 模型类与各阶段自持有数据类型
│   └── cli.hpp       # 命令行选项与辅助函数
├── src/
│   ├── classify.cpp  # 运行时生命周期与 preprocess/infer/postprocess
│   ├── cli.cpp       # 参数解析、默认值、图片/标签加载与打印
│   └── main.cpp      # 入口：解析选项、predict、打印
├── CMakeLists.txt    # 构建（C++17，显式 RDK_TARGET 板卡选择）
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── run.sh  # 构建并运行示例
```

<a id="supported-boards"></a>
## 适用板卡

| 板卡 | 状态 |
| --- | --- |
| S100 | supported |
| S600 | supported |
| X5 | not-supported |

CMake 读取 `/sys/class/boardinfo/soc_name` 并定义对应的 SoC 宏；
身份文件不可读视为错误，不做回退。

<a id="dependencies"></a>
## 依赖

板端镜像需要：CMake 与 C++17 编译器；OpenCV 开发头文件/库；`fmt` 开发
库；`/usr/hobot/include` 下的 Horizon DNN 头文件与 `/usr/hobot/lib`
下的库（`hbDNN`、`hbucp`）。工具实现来自 CMake 目标引用的共享
`utils/c_utils` 文件。启动脚本不安装系统包、不修改 SDK、不下载模型。

<a id="build"></a>
## 构建

启动脚本会自动构建；手动构建（cwd：仓库根目录）：

```bash
# 成功判据：build 目录中生成 resnet18 二进制
cmake -S samples/vision/resnet/runtime/cpp \
  -B samples/vision/resnet/runtime/cpp/build
cmake --build samples/vision/resnet/runtime/cpp/build --parallel
```

换一个构建目录选择的是同一个目标：

```bash
cmake -S samples/vision/resnet/runtime/cpp \
  -B /tmp/resnet18-alt-build
cmake --build /tmp/resnet18-alt-build --parallel
```

<a id="run"></a>
## 运行

前置条件：`bash samples/vision/resnet/model/download.sh s100`（或 S600
制品）。可在任意工作目录执行：

```bash
# 输入：model/s100 制品、随仓 zebra_cls.jpg、S 系列 ImageNet 标签
# 输出：stdout 上的 Top-K 行 — 成功判据：退出码 0
bash samples/vision/resnet/runtime/cpp/run.sh
```

S600 需显式选择制品；同一检出目录服务两块板时使用独立 build 目录：

```bash
MODEL_PATH="$PWD/samples/vision/resnet/model/s600/resnet18_224x224_nv12.hbm" \
BUILD_DIR="$PWD/samples/vision/resnet/runtime/cpp/build-s600" \
bash samples/vision/resnet/runtime/cpp/run.sh
```

启动脚本先检查模型、图片与标签文件，再调用 CMake、构建 `resnet18`，
并传入绝对路径使命令在任意目录可用。

<a id="parameters"></a>
## 参数

`resnet18` 二进制的选项，与 Python 运行时的 kebab-case 命名一致（启动
脚本会用 sample 的绝对路径覆盖前三项）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | 按 SoC：`/opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm`（S100）或 `/opt/hobot/model/s600/basic/resnet18_224x224_nv12.hbm`（S600） | HBM 模型路径 |
| `--test-img` | `../../../test_data/zebra_cls.jpg`（相对进程工作目录） | BGR 测试图 |
| `--label-file` | 仓库内 S 系列 ImageNet 标签路径 | 逐行一个标签 |
| `--top-k` | `5` | 打印的类别数量 |
| `--help` / `-h` | — | 打印用法 |

通过启动脚本覆盖的示例：

```bash
bash samples/vision/resnet/runtime/cpp/run.sh \
  --model-path /opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm \
  --test-img /tmp/zebra_cls.jpg \
  --label-file /tmp/imagenet_classes.names \
  --top-k 5
```

<a id="interface-lifecycle"></a>
## 接口与生命周期

`main.cpp` 解析选项后构造 `Resnet18 model(model_path)` —— 构造函数加载
HBM 包、读取张量元数据并分配可复用张量缓冲 —— 随后调用
`model.predict(image, top_k)` 并打印返回的类别。所有 DNN 与 UCP 类型都
留在 `src/classify.cpp` 的私有 `Impl` 中，`inc/classify.hpp` 仅依赖
OpenCV 与标准库。

模型单独暴露预处理、推理与后处理三个阶段，各阶段返回调用方自持有的
数据：

- `Resnet18Prepared preprocess(const cv::Mat&)`：按模型输入分辨率 letterbox
  缩放并完成 BGR→NV12 转换，得到自持有的 Y/UV 平面；
- `Resnet18Raw infer(const Resnet18Prepared&)`：将平面按行宽上传到模型输入
  张量（按行 stride 寻址），执行一次 `hbDNNInferV2` BPU 任务，把 F32 输出
  拷贝为自持有的 logits 向量（在后续推理后依然有效）；
- `std::vector<Classification> postprocess(const Resnet18Raw&, int top_k)`：
  稳定 softmax 与 Top-K 选取；
- `predict` 按上述顺序组合三个阶段。

错误以 C++ 异常抛出（含 SDK 错误描述）；入口打印错误并以状态码 2 退出。
资源在所有路径上（含部分初始化失败）由 RAII 释放。没有后台线程，
进程执行一次同步推理。

<a id="results-interpretation"></a>
## 结果解释

二进制按标签文件逐行打印 Top-K 类别，每行含类别 ID、分数与标签；成功
退出码为 0。分别记录每次 S100 或 S600 运行的板卡身份、制品引用、完整
构建/运行命令与 Top-K 输出。
