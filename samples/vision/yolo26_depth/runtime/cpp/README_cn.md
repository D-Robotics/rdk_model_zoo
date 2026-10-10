[English](README.md) | 简体中文

# X5 C++ 深度推理

本目录运行五种已发布 X5 YOLO26 Depth BIN。
具名 `Yolo26Depth` 模型统一负责前处理、warmup 加一次计时 forward 以及已校准 log-depth 还原；
参数解析、绘图与报告写入位于 CLI 模块。
二进制支持旗标与三个位置参数两种命令形式，并写出带 NumPy 文件头的数组，方便离线评估。

<a id="overview"></a>
## C++ 推理

在 X5 上从 BGR 图片估计相对深度。`main.cpp` 显式构造 `Yolo26Depth` 并调用 `predict`；
阶段计算、张量契约与 SDK 资源管理位于 `depth.cpp`；参数解析、深度上色与产物／报告写入位于 `cli.cpp`。
启动器选择模型并运行原生程序。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── inc/
│   ├── cli.hpp  # CLI 选项、图像加载、上色与报告声明
│   └── depth.hpp  # Yolo26Depth 模型、阶段数据类型、张量契约
├── src/
│   ├── main.cpp  # 薄入口：解析参数、构造模型、predict、保存
│   ├── cli.cpp  # 参数解析、深度上色、NPY/F32/PNG/报告写入
│   └── depth.cpp  # preprocess/infer/postprocess、letterbox/NV12、SDK 句柄
├── tests/  # （位于示例 tests/）原生契约、资源与 CLI 主机测试
├── CMakeLists.txt  # RDK_TARGET 门控的构建定义
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── launcher.py  # Python 脚本
└── run.sh  # 运行示例
```

<a id="supported-boards"></a>
## 适用板卡

| 目标 | 变体 | 原生状态 |
|---|---|---|
| X5 | n/s/m/l/x，已校准 log-depth / NV12 | 已实现 |
| S100/S100P/S600 | 请使用 Python 运行时 | 无对应原生深度实现；本程序显式拒绝 S 目标 |

身份规则与仓库注册表一致：优先 boardinfo `x5`；缺失时查 socinfo `x5u/x5h/x5m`；
再缺失才匹配设备树 `D-Robotics RDK X5 V1.0`。
已存在但未知的 boardinfo/socinfo 不继续回退。构造函数的身份门在任何 SDK 调用之前执行，
自定义模型路径不能绕过检查。

<a id="dependencies"></a>
## 依赖

需要匹配的 X5 Linux SDK，包含 `dnn/hb_dnn.h`、`dnn/hb_sys.h`、`libdnn`，
以及 C++17 编译器、CMake ≥3.16、OpenCV core/imgproc/imgcodecs、pthread、rt、dl。
构建链接系统 DNN/OpenCV；真实 SDK 编译链接在目标环境完成。
测试用模拟头文件不进入正式构建的 include 路径。

启动器和下载工具还需要 Python 与 PyYAML 来读取共享清单，不依赖 `hbm_runtime`；
二进制直接调用原生 DNN。脚本不安装软件、不在推理时下载模型，也不忽略链接器未解析符号。
请在匹配的 SDK 环境中显式准备依赖。以下命令从仓库根目录执行。

<a id="build"></a>
## 模型准备与构建

```bash
bash samples/vision/yolo26_depth/model/download.sh --target x5 --variant n
bash samples/vision/yolo26_depth/runtime/cpp/run.sh --target x5 --variant n \
  --build --output /work/depth/cpp-n
```

`--build` 先验证本机身份、所选制品及输入，再在 `runtime/cpp/build/x5` 中调用 CMake，
随后运行。X5 已发布模型必须通过清单 SHA-256 校验；已有输出目录会被拒绝。
也可在已配好依赖的 SDK 环境中单独构建：

```bash
cmake -S samples/vision/yolo26_depth/runtime/cpp \
  -B samples/vision/yolo26_depth/runtime/cpp/build/x5 \
  -DRDK_TARGET=x5 -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/yolo26_depth/runtime/cpp/build/x5 --parallel 2
```

CMake 必须显式指定 `RDK_TARGET=x5`（交叉编译时拒绝 auto，因其会读取构建主机的 SoC 身份）；
可通过缓存变量 `DNN_INCLUDE_DIR`、`DNN_LIBRARY` 指定显式准备的 SDK。
编译成功后，请在目标板端完成运行与兼容性验证。

<a id="run"></a>
## 运行或只查看解析结果

```bash
bash samples/vision/yolo26_depth/runtime/cpp/run.sh --list-models
bash samples/vision/yolo26_depth/runtime/cpp/run.sh --target x5 --variant l --dry-run
bash samples/vision/yolo26_depth/runtime/cpp/run.sh --target x5 --variant l \
  --test-img samples/vision/yolo26_depth/test_data/bus.jpg \
  --warmup 3 --output /work/depth/cpp-l
```

list/dry-run 不需要模型、SDK、CMake 或板卡，不会执行构建。
普通运行使用已有二进制，只有显式 `--build` 才编译。
`run.sh` 将用户相对路径按仓库根目录解析。

也可直接调用二进制，包括三个位置参数形式：

```bash
samples/vision/yolo26_depth/runtime/cpp/build/x5/yolo26_depth \
  --model-path /work/depth/model.bin --test-img /work/depth/input.jpg \
  --output /work/depth/native-direct --target x5
# 等价的位置参数形式：
samples/vision/yolo26_depth/runtime/cpp/build/x5/yolo26_depth \
  /work/depth/model.bin /work/depth/input.jpg /work/depth/native-positional
```

直接调用只验证板卡与张量契约，不验证发布方身份。
需要清单摘要校验和 `launch-report.json` 时使用启动器。
自编译模型同时传入 `--converted-model`、`--model-path` 与 list-models 给出的精确 `--asset-id`。
此时 ID 是**张量契约参照**，不表示新文件就是已发布制品；仍要求 X5 和同样的已校准 log-depth/NV12 契约。

<a id="parameters"></a>
## 参数

| 入口 | 参数 | 含义与默认值 |
|---|---|---|
| 启动器 | `--target`、`--variant`、`--asset-id` | 默认 x5/n，精确 ID 可推断变体；只有 X5 可执行 |
| 启动器 | `--model-path`、`--converted-model` | 外部模型必须给出精确 ID；自转换模式显式区分来源 |
| 两者 | `--test-img`、`--output` | 输入图与新输出目录；启动器默认附带 bus、`outputs/yolo26_depth_cpp` |
| 两者 | `--warmup` | 一次计时前的非负 forward 次数；默认 0（Python 运行时默认 3） |
| 启动器 | `--binary` | 显式指定已编译二进制，不能与 `--build` 同用 |
| 启动器 | `--build` | 前置检查后显式配置、编译、执行 |
| 启动器 | `--dry-run`、`--list-models` | 互斥主机查看模式 |
| 二进制 | `--model-path` / `--model`、`--test-img` / `--input` | 显式模型与图像，不隐式选择模型 |
| 二进制 | `--help` | 不加载模型，只显示用法 |

二进制仅接受空格分隔的 `--key value` 形式，不支持 `--key=value`。
调度使用 SDK 默认值。
非法选项、输入或目标返回退出码 2。

<a id="interface-lifecycle"></a>
## API、阶段与资源生命周期

```cpp
#include "depth.hpp"
#include <opencv2/imgcodecs.hpp>

yolo26_depth::Yolo26Depth model("/work/depth/model.bin",
                                yolo26_depth::DepthOptions{/*warmup=*/3});
cv::Mat image = cv::imread("/work/depth/input.jpg");
auto prepared = model.preprocess(image);   // 独立 NV12 + letterbox 上下文
auto raw = model.infer(prepared);          // 3 次预热 + 1 次计时 forward
auto result = model.postprocess(raw, prepared.context);
// model.predict(image) 串联完全相同的三个阶段。
```

`infer` 先执行恰好 `DepthOptions::warmup` 次不计时 forward，再做一次计时 forward，
返回已校准原始 log-depth 及 `RunMetadata{latency_ms, warmup}`；计时只覆盖完整 forward
（内存复制、缓存操作、SDK 运行、原始输出复制），不含还原阶段。每次调用返回独立拥有的数据，
不受下一次 predict 影响。模型不可复制；并发处理时为每个工作线程创建独立实例，
不作出任何 SDK 并发承诺。可选 C++ execution-gate 构造参数仅用于主机测试注入，不暴露为 CLI 绕过选项。

| 阶段 | 契约 |
|---|---|
| preprocess | 非空 BGR CV_8UC3 → 768 线性 letterbox，填充 114 → 独立打包 NV12 字节及逐次 ImageContext |
| infer | 精确 warmup + 一次计时 forward；唯一输入/模型/输出；返回带 RunMetadata 的独立 F32 192 方形原始值，不做 exp、绘图或文件写入 |
| postprocess | 有限 192×192 F32 → exp、放大到 768、按上下文去 padding、还原原图；拒绝不匹配的上下文 |
| predict | 对一张图恰好串联 preprocess → infer → postprocess |

几何取整与 Python 的 ties-to-even 一致；极端长宽比导致某一缩放尺寸为零时明确报错。
SDK 输出必须是 F32/NONE、NHWC 或 NCHW 单通道 192 方形。
正字节 stride 或由 aligned shape 推导的 stride 都会与分配容量核对，支持读取带 padding 的输出。
输入要求紧凑 NV12 pyramid 几何；带 padding 的逻辑输入尺寸显式不支持，不会静默错误复制。

模型在成功路径恰好一次释放完成的任务句柄（释放失败会报错），guard 兜底全部异常路径；
析构仅释放已取得的资源，构造失败不泄漏。
绘图与全部产物／报告写入位于 `cli.cpp`；`main.cpp` 只解析参数、构造模型、predict 并保存。

<a id="results-interpretation"></a>
## 输出与验证边界

成功的原生执行写出：

- `log_depth.npy`：独立 float32 192×192 已校准 log-depth。
- `depth_native.npy`：float32 原图 H×W 相对深度。
- `depth_native.f32`：同样深度的小端行优先 F32 裸数据；尺寸见报告。
- `depth.png`、`overlay.png`：插值计算的 2%/98% 分位范围与反向 TURBO；原图权重 0.45，深度颜色权重 0.55。
- `report.json`：实际模型名、路径、形状、预热与计时；未另行记录时 SDK 版本为 unknown。

启动器增加 `launch-report.json`，记录精确命令、UTC 起止时间、退出码、模型/输入/二进制/
原生报告摘要及发布/自转换来源；已创建输出目录时另保存原生 stdout/stderr。
早期失败可能仅有 stderr，部分输出目录不表示成功。

`report.json` 的 `latency_ms` 覆盖一次完整 forward 的内存复制、缓存操作、SDK 调用及原始输出复制，
且在恰好 `warmup` 次不计时 forward 之后；与 HRT 的纯 BPU 延迟比较时请注意口径差异。
深度为相对值，颜色不表示米制距离；精度以数值指标为准。
