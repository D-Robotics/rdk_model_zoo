# YOLOv5 原生 C++ runtime

本目录是统一 YOLOv5 样例的 C++ 版本，保留 X5 HB-DNN adapter 与 S UCP adapter
两套硬件调用，只共享与 SDK 无关的部分：张量元数据 gate（`yolov5_gate.*`）、数值
解码器（`yolov5_decode.*`）和证据 dump 写入器（`yolov5_dump.*`）。原生二进制
负责解析参数、执行目标相关的输入/前向 I/O、解码三路原始输出、按需写出证据
dump，并交给独立的 OpenCV 可视化模块渲染。发布事实由 `launcher.py` 通过
`samples.vision.yolov5.runtime.python.model_binding` 解析；原生二进制不会根据
文件名猜测布局。

<a id="supported-boards"></a>
## 支持板卡

| 板卡 | 状态 | 说明 |
| --- | --- | --- |
| X5 | supported-not-run | 有 X5 HB-DNN 源；本主机无 X5 SDK、无板卡、无已发布模型文件，未在硬件上编译或运行 |
| S100 | supported-not-run | 有 S UCP 源；无板卡、SDK 及 `yolov5x_672x672_nv12.hbm` 资产 |
| S600 | supported-not-run | 同一 S 源，使用 64 字节 BPU 对齐宏；未编译、未运行 |
| S100P | not-supported | YOLOv5 无 S100P 发布资产，`--target s100p` 被拒绝 |

每个 adapter 只为一个目标编译，生成的二进制会拒绝与其编译身份不一致的
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
每个目标只编译一个 adapter 源文件；CMake 同时定义 SoC 对齐宏（`SOC_S600` /
`SOC_S100` / `SOC_S100P`）与运行时用于拒绝不匹配 `--target` 的
`YOLOV5_TARGET_NAME`。产物为该构建目录下的 `yolov5_cpp`。

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
| `--variant` | X5 `s-v2.0`，S `x-672` | 资产变体；X5 默认为固定 C++ 源默认值 |
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

`yolov5::RuntimeOptions` 是原生入口契约；`run_native` 负责目标相关的模型初始化、
张量分配、cache 操作、同步前向与清理。

- X5：要求恰好一个 packed NV12 模型，输入为紧凑 `[1,3,640,640]`，三路输出为
  原生 F32、`NONE` 量化 NHWC 头，stride 必须恰为 8/16/32。固定 X5 源就是按紧凑
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
  标量 scale/zero-point 描述符（长度 1）被接受，因为 adapter 私有反量化器对
  其做广播；共享 `c_utils` 的 `dequantizeTensorS32` 会按 `scale_data[c]` 越界
  读取，永远不会拿到这类张量。raw dump 保留完整 `alignedByteSize` 范围，并在
  manifest 中记录 stride 与完整 scale/zero-point 数组，因此带 padding 的运行
  仍可机器比对。
- 所有权：两个 adapter 都只释放真正分配成功的资源，部分失败的分配不会变成盲目
  free。X5 通过 RAII lease 释放 task 与缓存；S 的 guard 跳过 `sysMem` 从未赋值
  的张量。
- 编译期构建身份（`YOLOV5_TARGET_NAME`）必须与 `--target` 一致，因此按某一种 S
  对齐编译的二进制不能当作另一个目标运行。

与固定源相比的**已声明**差异（保留而非静默抹平）：

- **X5 默认变体。** 固定 X5 C++ 源默认 `s-v2.0` 制品；统一 Python runtime 默认
  `n-v7.0`。原生 launcher 在既未给 `--variant` 也未给 `--asset-id` 时保留 C++
  源默认。
- **NMS。** X5 保留源 `cv::dnn::NMSBoxes` 的按类行为：score 边界为严格大于
  `--score-thres`，每类上限 `top_k = 300`。S 保留源 `nms_bboxes` 行为：等于
  `--score-thres` 保留，且无每类上限。
- **预处理。** 两个原生 adapter 均使用 letterbox；统一 Python 路径默认 stretch。
  这是有意的源兼容选择，并不表示两者数值完全一致。
- **调度。** 固定 S 源把 `priority` 强制写 0；统一 S adapter 应用调用方的
  `--priority`/`--bpu-core`，使文档参数真实生效。`--bpu-core` 是核**索引**
  （`-1` = 任意，`0..3`），并显式转换为 SDK 的 backend 位掩码
  （`HB_UCP_BPU_CORE_0..3 = 1ULL<<0..3`，`HB_UCP_BPU_CORE_ANY = 1ULL<<7`）；
  超出 `-1..3` 的索引被拒绝，原始索引绝不会直接赋给 backend 字段。X5 没有
  经验证的 HB-DNN 映射，因此非默认值被明确拒绝而不是静默忽略。
- **非有限 score。** 统一解码器丢弃非有限置信度；源 S 解码会保留它们。统一行为
  是已声明的修复。

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
- input 文件保存该次推理实际提交的输入缓冲（X5 为紧凑 NV12 负载，S 为按行
  汇聚的平面负载）；未初始化的 padding 字节有意不导出。X5 的原始与变换后
  张量同为原生 F32 头；S 的原始张量保留完整分配范围（`alignedByteSize`，含
  pixel padding）、变换后为反量化浮点，因此板端比对可以分别检查两个阶段，
  并依据 manifest 中的 stride 解释带 padding 的布局。dump 记录的是本二进制
  的产出本身，并不因此声明与固定源 runtime 数值等价——那需要板端 evaluator
  另行建立。
- 板端状态（2026-09-24，协调者证据）：整改前提交在真实 X5 8GB 上编译链接
  `rc=0`，首次 launcher 推理返回 `rc=0`；同一提交在 S100 上编译失败，原因是 S
  adapter 使用了 X5 独有的 SDK 拼写，本轮已按板端头文件证据修复。本工作树不作
  任何数值板端对照、精度或性能声明；板卡复验由协调者执行。主机测试通过仍只
  代表契约/解码器结论。
