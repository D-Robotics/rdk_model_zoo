[English](README.md) | 简体中文

# YOLOv5 原生 C++ runtime

本目录是 YOLOv5 样例的 C++ 原生 runtime，按普通五文件交付：`inc/detect.hpp` 与
`src/detect.cpp` 负责检测模型（SDK 无关的张量 gate、数值解码器、S 反量化器，
以及同一编译单元内由编译期目标宏选择的 X5 HB-DNN 与 S UCP 后端）；
`inc/cli.hpp` 与 `src/cli.cpp` 负责命令行、证据 dump 写入器与渲染输出；
`src/main.cpp` 构造模型、执行 `predict` 并报告。发布事实由 `launcher.py`
通过 `samples.vision.yolov5.runtime.python.model_binding` 解析；原生二进制
不会根据文件名猜测布局。

<a id="overview"></a>
## C++ 推理

通过 X5 HB-DNN 或 S UCP 适配器运行 YOLOv5 检测。程序准备 NV12 输入、解码三个检测头并保存标注图片。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── inc/  # detect.hpp（模型）、cli.hpp（CLI）
├── src/  # main.cpp、cli.cpp、detect.cpp
├── CMakeLists.txt  # 显式目标构建
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── launcher.py  # Python 脚本
└── run.sh  # 运行示例
```

<a id="supported-boards"></a>
## 支持板卡

| 板卡 | 制品 | SDK 与输入 |
| --- | --- | --- |
| X5 | 已发布的 640×640 `.bin` 变体 | X5 DNN SDK，紧凑 NV12 |
| S100 | `x-672` `.hbm` | S100 UCP/DNN SDK，拆分 NV12 |
| S600 | `x-672` `.hbm` | S600 UCP/DNN SDK，拆分 NV12 |

上表三种板卡均按所列制品与 SDK 支持。构建后在板卡上以默认 case 运行启动器；精度与性能测量按[评估器指南](../../evaluator/README_cn.md)执行。

每个后端只为一个目标编译，生成的二进制会拒绝与其编译身份不一致的
`--target`（见[接口与资源生命周期](#interface-lifecycle)），因为 S600 与其余
S 目标的 alignment 宏不同。

<a id="dependencies"></a>
## 依赖

- CMake ≥ 3.16 与 C++17 编译器。
- OpenCV 开发头文件与库（仅用于渲染输出）。
- Horizon DNN 头文件位于 `/usr/hobot/include`、库位于 `/usr/hobot/lib`；
  S 目标额外链接 `hbucp`。
- `utils/c_utils` 中的共享 C++ helper（`preprocess`、`postprocess`、`nn_math`），
  由 CMake 目标以相对路径引用。
- launcher 不安装软件包、不下载模型、不读板卡身份：`--help`、`--list-models`、
  `--dry-run` 无需 SDK 即可运行。

<a id="build"></a>
## 构建

目标是显式的，配置阶段不读取 `/sys/class/boardinfo`。

```bash
# cwd: 仓库根目录
cmake -S samples/vision/yolov5/runtime/cpp -B samples/vision/yolov5/runtime/cpp/build/x5 -DYOLOV5_TARGET=x5
cmake --build samples/vision/yolov5/runtime/cpp/build/x5 --parallel

cmake -S samples/vision/yolov5/runtime/cpp -B samples/vision/yolov5/runtime/cpp/build/s100 -DYOLOV5_TARGET=s100
cmake --build samples/vision/yolov5/runtime/cpp/build/s100 --parallel
```

`YOLOV5_TARGET` 只能取 `x5`、`s100`、`s100p` 或 `s600`，其它值在配置阶段报错。
一个 `src/detect.cpp` 服务所有目标：所选后端由编译期宏选择；CMake 同时定义
SoC 对齐宏（`SOC_S600` / `SOC_S100` / `SOC_S100P`）与运行时用于拒绝不匹配
`--target` 的 `YOLOV5_TARGET_NAME`。产物为该构建目录下的 `yolov5_cpp`。

<a id="run"></a>
## 运行

前置条件：由 [`model/download.sh`](../../model/README.md) 准备模型资产，并通过
launcher 的身份检查。

```bash
# cwd: 仓库根目录
# 不接触板卡与 SDK，仅查看解析出的发布事实
samples/vision/yolov5/runtime/cpp/run.sh --dry-run --target x5

# 真实运行：精确 asset id + 外部路径，需在与目标一致的板卡上执行
samples/vision/yolov5/runtime/cpp/run.sh --target x5 \
  --asset-id x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --model-path /absolute/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --test-img /absolute/bus.jpg --dump-dir /tmp/yolov5-x5-dump

# S split-NV12 构建
samples/vision/yolov5/runtime/cpp/run.sh --target s100 \
  --asset-id s:yolov5:s100/yolov5x_672x672_nv12.hbm \
  --dump-dir /tmp/yolov5-s100-dump
```

`--list-models --target <t>` 打印该目标的已发布资产。外部 `--model-path` 必须与
manifests 中精确的 `--asset-id` 一起给出。预期产物为 `result.jpg`（或
`--output <file>`）；给出 `--dump-dir` 时另有 `manifest.json` 与每个张量的原始
文件。

<a id="parameters"></a>
## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--target` | `auto` | `x5`、`s100`、`s100p` 或 `s600`；`auto` 由 launcher 按板卡身份解析 |
| `--variant` | X5 `s-v2.0`，S `x-672` | 资产变体；默认值按板卡如中列所示 |
| `--asset-id` | 省略 | 与 `--model-path` 成对出现，必须与 manifest 精确一致 |
| `--model-path` | manifest 解析路径 | 模型文件；仅在给出 `--asset-id` 时有效 |
| `--test-img` | 样例测试数据 | BGR 输入图像 |
| `--label-file` | 无 | 每行一个类别标签，用于渲染 |
| `--output` | `result.jpg` | 渲染输出图像 |
| `--dump-dir` | 无 | 机器可比证据 dump 目录 |
| `--score-thres` | `0.25` | 置信度阈值，`[0,1]` 内有限值 |
| `--nms-thres` | `0.45` | NMS IoU 阈值，`[0,1]` 内有限值 |
| `--priority` | `0` | 调度优先级；S 生效，X5 拒绝 |
| `--bpu-core` | `-1` | BPU core（`-1` 为 runtime 默认）；S 生效，X5 拒绝 |

<a id="interface-lifecycle"></a>
## 接口与资源生命周期

`yolov5::Yolov5`（声明于 `inc/detect.hpp`）是原生模型契约：构造函数执行构建
身份与调度 gate 并加载 runtime（RAII；部分失败的初始化只释放已分配的资源），
公开阶段为 `preprocess` → `infer` → `postprocess`，另有显式 `predict` 串联。
`src/main.cpp` 直接构造模型，通过 CLI（`inc/cli.hpp`）加载图像，并把调用方持有
的像素传给模型：`Input` 是 BGR 缓冲加源图几何；`preprocess` 在不接触 SDK 的
情况下把它转换为自持有的 `Prepared` NV12 载荷；`infer` 在入口先校验该载荷
（精确的模型平面长度与正的源图几何，先于任何 SDK 分配或拷贝），再上传其显式
传入的参数——绝不是后续调用可能改写的实例缓冲——`postprocess` 只做解码。
`predict` 串联三个阶段并返回 `Prediction`（`Result` 加本次调用的
`RunEvidence`）；`main` 把这个返回值交给 CLI 报告，dump 与渲染输出由 CLI
负责。没有"上一次调用"访问器：每个阶段的数据按次持有，返回的 `Prediction`
在后续 `predict` 之后依然有效。

- X5：要求恰好一个 packed NV12 模型，输入为紧凑 `[1,3,640,640]`，三路输出为
  原生 F32、`NONE` 量化 NHWC 头，stride 必须恰为 8/16/32。X5 后端按紧凑
  缓冲写入 NV12、按扁平 float 读头，因此 gate 要求上报的 aligned 布局与 valid
  布局一致：带 padding 的制品会被带精确原因地拒绝而不是误读，dump manifest 会
  记录其 `alignedShape`/`stride`/`alignedByteSize` 供后续跟进。分配还必须覆盖
  紧凑 NV12 帧（头为 `height*width*channels` 个 float）。
- S：要求一个 packed 模型，输入为 split `Y[1,672,672,1]` 与 `UV[1,336,336,2]`，
  三路输出由元数据描述。S SDK 不上报 `alignedShape`，存储布局由 `stride[]` 加
  `alignedByteSize` 描述。任何读取之前，反量化 gate 会证明原生 dtype、描述符
  长度，以及反量化器实际执行的寻址（元素 `(h,w,c)` 位于字节偏移
  `(h*W + w)*stride[2] + c*stride[3]`）：`stride[2]` 必须覆盖一个完整像素
  （`channels` 个元素——已发布 S100 模型 255 通道下 `stride[2]=1024` 的合法
  pixel padding 被接受；更小导致相邻像素重叠的值被拒绝）；`stride[1]` 必须等于
  `width*stride[2]`；分配必须在溢出检查的算术下覆盖到最后一个被寻址字节。
  标量 scale/zero-point 描述符（长度 1）被接受，因为模型私有反量化器对其做
  广播；共享 `c_utils` 的 `dequantizeTensorS32` 会按 `scale_data[c]` 越界
  读取，永远不会拿到这类张量。raw dump 保留完整 `alignedByteSize` 范围，并在
  manifest 中记录 stride 与完整 scale/zero-point 数组，因此带 padding 的运行
  仍可机器比对。
- 所有权：两个后端都只释放真正分配成功的资源，部分失败的分配不会变成盲目
  free。X5 通过 RAII lease 释放 task 与缓存；S 的 guard 跳过 `sysMem` 从未赋值
  的张量。
- 编译期构建身份（`YOLOV5_TARGET_NAME`）必须与 `--target` 一致，因此按某一种 S
  对齐编译的二进制不能当作另一个目标运行。

Python 与 C++ Runtime 的行为区别：

- **X5 默认变体。** 原生 launcher 在既未给 `--variant` 也未给 `--asset-id` 时
  默认 `s-v2.0` 制品；Python runtime 默认 `n-v7.0`。
- **NMS。** X5 执行按类 `cv::dnn::NMSBoxes`：score 必须严格大于
  `--score-thres`，每类最多保留 `top_k = 300` 个框。S 执行 `nms_bboxes`：等于
  `--score-thres` 的 score 保留，且无每类上限。
- **预处理。** 两个原生 adapter 均使用 letterbox；Python runtime 默认 stretch。
- **调度。** S adapter 应用调用方的 `--priority`/`--bpu-core`。`--bpu-core`
  是核**索引**（`-1` = 任意，`0..3`），并显式转换为 SDK 的 backend 位掩码
  （`HB_UCP_BPU_CORE_0..3 = 1ULL<<0..3`，`HB_UCP_BPU_CORE_ANY = 1ULL<<7`）；
  超出 `-1..3` 的索引被拒绝，原始索引绝不会直接赋给 backend 字段。X5 没有
  经验证的 HB-DNN 映射，因此非默认值被明确拒绝而不是静默忽略。
- **非有限 score。** 解码器丢弃非有限置信度。

<a id="results-interpretation"></a>
## 结果解读

- 退出码 `0` 表示运行完成；`2` 表示被拒绝或失败。失败时若给出 `--dump-dir`，仍
  会写出带 `return_code` 与 `error` 的 manifest，使失败可追溯。
- 渲染图只是便利产物。机器比对以 dump 为准：`manifest.json` 用 SHA-256 绑定
  `target`、`build_target`、`asset_id`、`model_path`、`image_path` 以及运行中的
  `binary_path`，记录观测到的输入/输出元数据（shape、dtype、量化类型、scale
  长度、完整 scale/zero-point 数值、`quantizeAxis`、`alignedByteSize`、上报的
  `stride[]`，以及 SDK 上报时的 `alignedShape`——未上报则为 null）、实际参数、
  UTC 时间戳、`argv`、`cwd`、`return_code`。张量负载按阶段分目录写入
  `input/`、`raw/`、`transformed/`，各自带 shape、字节数、文件名与 SHA-256，
  因此同一输出的原始与变换后字节不可能互相覆盖。
- input 文件保存该次推理实际提交的输入缓冲（X5 为紧凑 NV12 负载，S 为紧凑的
  Y/UV 平面负载——即上传时按 stride 步距写入的每行有效字节）；未初始化的
  padding 字节有意不导出。X5 的原始与变换后
  张量同为原生 F32 头；S 的原始张量保留完整分配范围（`alignedByteSize`，含
  pixel padding）、变换后为反量化浮点，因此板端比对可以分别检查两个阶段，
  并依据 manifest 中的 stride 解释带 padding 的布局。dump 记录的是本二进制
  的产出本身；跨 runtime 的评估由板端 evaluator 另行执行。
