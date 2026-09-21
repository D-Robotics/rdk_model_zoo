English | [简体中文](./README_cn.md)

# MobileNetV2 图像分类（C++，S 系列）

本 C++ 流程在 S 系列 BPU 上运行量化 MobileNetV2 HBM 模型，打印 Top-K
类别标签与置信度。这是保留自 rdk_s @380e1a2 的经审计 S 系列
`hbDNNInferV2` 实现（`src/` 与 `inc/` 原样保留）；X5 源分支只交付了
Python，因此本流程显式声明为 S 系列范围。

<a id="supported-boards"></a>
## 适用板卡

仅 S100 与 S600。启动器一次性读取 `/sys/class/boardinfo/soc_name` 与
`/sys/class/boardinfo/board_type`（`board_type` 变体），身份判定与 Python
流程一致（`samples/_shared/platforms.py`，登记于
`docs/release/platforms.json`）。S100P 以其两种登记形式被拒绝（soc_name
`s100p`；或 soc_name `s100` 且 board_type `s100p`/`rdk s100p`），未知或
不可读的身份文件按未知板卡拒绝——均显式报错，绝不静默回退 s100 制品
（旧源启动器会回退）。`SOC_NAME_FILE`/`BOARD_TYPE_FILE` 供主机 fixture
测试（`tests/test_cpp_launcher_identity.py`）覆盖身份来源；板卡上不要
设置。

<a id="dependencies"></a>
## 依赖

CMake、C++17 编译器、OpenCV 开发包、`libgflags-dev` 以及板卡镜像的
Horizon DNN 头文件/库。请显式安装（启动器绝不调用 apt）：

```bash
# cwd：板卡上 — 成功判据：apt 报告这些包已安装
sudo apt update && sudo apt install -y libgflags-dev
```

<a id="build"></a>
## 构建

手工构建（cwd：`samples/vision/mobilenetv2/runtime/cpp`；成功判据：
`build/` 内有 `mobilenetv2` 二进制）：

```bash
mkdir -p build && cd build && cmake .. && make -j"$(nproc)"
```

`CMakeLists.txt` 在配置期经 `/sys/class/boardinfo/soc_name` 探测 SoC 并
定义 `SOC_S100`/`SOC_S600`；该文件保持源分支原样。小内存板（2026-09-21
在 S100 实测）满并行编译可能被 OOM 杀死——请改用 `make -j1` 或
`BUILD_JOBS=1 bash run.sh`。

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
| `--model_path` | `.hbm` 制品路径 | sample 相对 `model/<soc>/mobilenetv2_224x224_nv12.hbm` |
| `--test_img` | 测试图路径 | sample 相对 `test_data/zebra_cls.jpg` |
| `--label_file` | 标签文件路径 | sample 相对 `test_data/imagenet1000_labels.txt` |
| `--top_k` | 打印的 Top-K 数量 | `5` |

二进制编译期内建默认值指向 `/opt/hobot/model/...`；启动器始终显式传参，
因此除非自行传参，不会使用系统模型位置。

<a id="interface-lifecycle"></a>
## 接口与生命周期

`mobilenetv2::init()` 加载模型、分配张量并读取布局 metadata；
`pre_process`、`infer`、`post_process` 为以引用传张量的自由函数（声明见
`inc/mobilenetv2.hpp`）。源码内为 Doxygen 注释；仓库级 API 参考的构建
方式见 `docs/source_reference/README.md`。

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

正确的统一运行应在分数噪声内复现该排序。B1 板端运行（2026-09-21）已在
S100 以 BUILD_JOBS=1 构建本流程并复现基线 TOP-1（label=zebra）；S600 C++
构建保持 not-run（不在 B1 冒烟集内）。分数全零或 NaN 说明
制品/输入配对错误，不是调参问题。
