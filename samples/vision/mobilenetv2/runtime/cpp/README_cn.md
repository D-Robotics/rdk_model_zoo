[English](README.md) | 简体中文

# MobileNetV2 图像分类（C++，S 系列）

本 C++ 流程在 S 系列 BPU 上运行量化 MobileNetV2 HBM 模型，打印 Top-K
类别标签与置信度。它是 S 系列 `hbDNNInferV2` 实现（`src/` 与 `inc/`
原样保留）；X5 交付线只提供 Python，因此本流程的适用范围为 S 系列。

<a id="overview"></a>
## C++ 推理

本目录提供C++ 推理所需的程序与操作说明。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── inc/
│   ├── classify.hpp  # MobileNetV2 模型类与各阶段自持有数据类型
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

仅 S100 与 S600。启动器一次性读取 `/sys/class/boardinfo/soc_name` 与
`/sys/class/boardinfo/board_type`（`board_type` 变体），身份判定与 Python
流程一致（`utils/py_utils/platforms.py`，登记于
`docs/release/platforms.json`）。S100P 以其两种登记形式被拒绝（soc_name
`s100p`；或 soc_name `s100` 且 board_type `s100p`/`rdk s100p`），未知或
不可读的身份文件按未知板卡拒绝——均显式报错，绝不静默回退 s100 制品。
`SOC_NAME_FILE`/`BOARD_TYPE_FILE` 供主机 fixture
测试（`tests/test_cpp_launcher_identity.py`）覆盖身份来源；板卡上不要
设置。

<a id="dependencies"></a>
## 依赖

CMake、C++17 编译器、OpenCV 开发包、`fmt` 开发库以及板卡镜像的
Horizon DNN 头文件/库。请使用板卡镜像自带的 SDK 开发包，并在该环境中完成完整构建（启动器
绝不调用 apt）。

<a id="build"></a>
## 构建

手工构建（cwd：`samples/vision/mobilenetv2/runtime/cpp`；成功判据：
`build/` 内有 `mobilenetv2` 二进制）：

```bash
mkdir -p build && cd build && cmake .. && make -j"$(nproc)"
```

`CMakeLists.txt` 在配置期选择板卡：板卡上原生配置时读取
`/sys/class/boardinfo/soc_name`（`-DRDK_TARGET=auto`，默认值）并定义
`SOC_S100`/`SOC_S600`；交叉编译必须显式传 `-DRDK_TARGET=s100|s600`
（交叉编译下 auto 直接报错，不支持的目标同样报错终止配置）。小内存板
满并行编译可能被 OOM 杀死——请改用 `make -j1` 或 `BUILD_JOBS=1 bash run.sh`。

<a id="run"></a>
## 运行

一键形式（cwd：任意；前置：按 [model/README_cn.md](../../model/README_cn.md)
准备制品；成功判据：退出码 0 且打印 Top-5 列表）：

```bash
bash samples/vision/mobilenetv2/runtime/cpp/run.sh
```

启动器构建到 `runtime/cpp/build/` 并以准备好的模型、
`test_data/zebra_cls.jpg` 与 `test_data/imagenet1000_labels.txt` 执行
二进制。环境变量覆盖：`MODEL_PATH`、`TEST_IMAGE`、`LABEL_FILE`、`TOP_K`、
`BUILD_DIR`、`BUILD_JOBS`。它绝不下载模型：缺制品时显式报错并给出下载
命令。

<a id="parameters"></a>
## 参数

| 参数 | 说明 | 默认值（来自启动器） |
| --- | --- | --- |
| `--model-path` | `.hbm` 制品路径 | sample 相对 `model/<soc>/mobilenetv2_224x224_nv12.hbm` |
| `--test-img` | 测试图路径 | sample 相对 `test_data/zebra_cls.jpg` |
| `--label-file` | 标签文件路径 | sample 相对 `test_data/imagenet1000_labels.txt` |
| `--top-k` | 打印的 Top-K 数量 | `5` |

选项名为与 Python 运行时一致的 kebab-case，同时接受 `--flag value` 与
`--flag=value` 两种写法。二进制编译期内建默认值指向
`/opt/hobot/model/...`；启动器始终显式传参，因此除非自行传参，不会使用
系统模型位置。

<a id="interface-lifecycle"></a>
## 接口与生命周期

`main.cpp` 解析选项后构造 `MobileNetV2 model(model_path)` —— 构造函数加载
HBM 包、读取并校验张量元数据并分配可复用张量缓冲 —— 随后调用
`model.predict(image, top_k)` 并打印返回的类别。所有 DNN 与 UCP 类型都
留在 `src/classify.cpp` 的私有 `Impl` 中，`inc/classify.hpp` 仅依赖
OpenCV 与标准库。

模型单独暴露预处理、推理与后处理三个阶段，各阶段返回调用方自持有的
数据：

- `MobileNetV2Prepared preprocess(const cv::Mat&)`：按模型输入分辨率
  letterbox 缩放并完成 BGR→NV12 转换，得到自持有的 Y/UV 平面；
- `MobileNetV2Raw infer(const MobileNetV2Prepared&)`：将平面按行宽上传到
  模型输入张量（按行 stride 寻址），执行一次 `hbDNNInferV2` BPU 任务，
  把 F32 输出拷贝为自持有的概率向量（在后续推理后依然有效）；
- `std::vector<Classification> postprocess(const MobileNetV2Raw&, int top_k)`：
  直接读取概率（模型输出节点已是 softmax 之后的分布，不做再次归一化）
  并做 Top-K 选取；
- `predict` 按上述顺序组合三个阶段。

错误以 C++ 异常抛出（含 SDK 错误描述）；入口打印错误并以状态码 2 退出。
资源在所有路径上（含部分初始化失败）由 RAII 释放。没有后台线程，
进程执行一次同步推理。

<a id="results-interpretation"></a>
## 结果解读

成功时打印 `TOP-n: label=..., prob=...` 行，标签取自标签文件，分数为
制品 softmax 之后的输出。rdk_s @s-v1.1.2 在 S100 上使用
`zebra_cls.jpg` 的记录：

```text
TOP-1: label=zebra, prob=0.992246
TOP-2: label=tiger, Panthera tigris, prob=0.00404656
TOP-3: label=hartebeest, prob=0.00133707
TOP-4: label=tiger cat, prob=0.000722661
TOP-5: label=impala, Aepyceros melampus, prob=0.000539704
```

正确的运行应在分数噪声内复现该排序；使用随附 zebra 图片时 TOP-1 为
`zebra`。分数全零或 NaN 说明制品/输入配对错误，不是调参问题。
