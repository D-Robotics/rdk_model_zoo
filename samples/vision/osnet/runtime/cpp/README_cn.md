[English](README.md) | 简体中文

# OSNet 原生 C++ 验证

<a id="overview"></a>
## 用途

先检查固定输入字节与原始特征，再测原生前向、前处理和端到端延迟。输入为已解码 RGB，不是 JPEG。

<a id="directory"></a>
## 目录

main.cpp 调用 osnet.cpp 中的 osnet_main；run.sh 检查板型和制品身份并编译。build/ 保存被忽略的可执行文件与 result.json。

<a id="supported-boards"></a>
## 适用板卡

X5 使用已登记 RGB BIN；S100/S100P/S600 使用各自验收的 F32 HBM 与 UCP 后端。执行前准备精确模型及对应 fixture 参考特征。

<a id="dependencies"></a>
## 依赖

AArch64 C++17 编译器、dnn/hb_dnn.h 和 libdnn（记录版本 1.24.5、HBRT 3.15.55.0）；代码使用 arm_neon.h。启动脚本只用 Python 做身份检查，不需要 OpenCV、JPEG 解码器或 CMake。

<a id="build"></a>
## 构建

在 X5 仓库根目录执行，不安装、不下载。run.sh 调用 g++ -O3 -std=c++17 -Wall -Wextra 并链接 -ldnn，编译前先检查实际板型。

```bash
bash samples/vision/osnet/runtime/cpp/run.sh
```

<a id="run"></a>
## 运行

默认使用 spatial-rgb-late8、已准备的 test_data/fixture、3000 次计时，输出 build/result.json。用环境变量定制：

```bash
VARIANT=spatial-rgb-int16 RUNS=3000 OUTPUT_JSON=/tmp/osnet-int16.json \
  bash samples/vision/osnet/runtime/cpp/run.sh
```

<a id="parameters"></a>
## 参数

可执行文件位置参数为 MODEL、images.dat、reference.f32、result.json、可选 runs（二进制默认 1000，启动脚本默认 3000）。启动变量：VARIANT=spatial-rgb-late8；MODEL_PATH=model/<variant>.bin；FIXTURE_DIR=test_data/fixture；BUILD_DIR=runtime/cpp/build；OUTPUT_JSON=<build>/result.json；RUNS=3000；CXX=g++；PYTHON_BIN=python3。未覆盖的路径由脚本目录定位。

<a id="interface-lifecycle"></a>
## 接口与生命周期

osnet.cpp 中，Resize.run 将已解码 RGB 转为有符号平面输入；OSNet 加载单模型并分配缓存张量，infer 同步等待、复制浮点输出并可选归一化。等待后释放任务；析构释放张量和模型。模型与缩放缓存按单线程使用，CLI 负责文件和统计。数值代码保留自已测源码，没有独立公开头文件。

<a id="results-interpretation"></a>
## 结果

result.json 包含 preprocess_exact、feature_max_abs_error、l2_norm_max_error，以及 native_forward、pipeline、preprocess 的 mean/median/p95/p99/min/max（ms）。输入或特征检查失败时不测速；退出 0 只表示复测完成，不代表完整检索精度。数值来自 2026-10-29 原实验：3368 query + 15913 gallery，L2 归一化平方欧氏距离，无 reranking。C++ 预热 50 次、计时 3000 次，batch=1、单路，BPU 996 MHz、CPU schedutil。E2E 为已解码 64×128 RGB 人体块 → Pillow 等价 NEON 缩放至 128×256 → 同步 Runtime → L2，不含解码、检测和匹配；其他尺寸不能套用这些延迟。

### Workbench 1/2 流基准

新基准按同一帧分别计时：已解码 RGB 裁剪图的缩放/打包、同步 Runtime（含缓存操作与输出复制）、L2 归一化。每个工作线程独占模型、张量和缩放缓存；每轮先在同一实例上预热，再开始计时。总吞吐量使用共同墙钟，包含循环开销，不计解码、磁盘读写和检索。实现为原生 NEON，不使用 OpenCV，也不修改 CPU 性能策略。

```bash
BENCHMARK=1 STREAMS=1 WARMUP=20 RUNS=200 ROUNDS=3 OUTPUT_JSON=/absolute/new/e2e-1.json bash samples/vision/osnet/runtime/cpp/run.sh
BENCHMARK=1 STREAMS=2 WARMUP=20 RUNS=200 ROUNDS=3 OUTPUT_JSON=/absolute/new/e2e-2.json bash samples/vision/osnet/runtime/cpp/run.sh
```

也可直接运行 `osnet --benchmark MODEL images.dat reference.f32 result.json STREAMS WARMUP RUNS ROUNDS`。JSON 保存四阶段逐帧数据、每轮墙钟、每个工作线程的 32 个固定样本一致性结果及真实线程数。每流 200 帧 × 3 轮，1/2 流分别计时 600/1200 帧。环境快照单独保存，不将历史数据标为本轮结果。

S 系列设置 TARGET=s100（或 s100p/s600）、VARIANT=<target>-f32-int16 和 MODEL_PATH。run.sh 自动启用 UCP，链接 libdnn/libhbucp；SDK 测试版本为 3.13.6 / HBRT 4.7.5。FIXTURE_DIR 中需有 images.dat 与 <variant>.f32，后者是同一 HBM 的独立 Python SDK 参考。完整命令见 [S 系列流程](../../evaluator/S_SERIES_cn.md)。
