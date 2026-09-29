[English](README.md) | [简体中文](README_cn.md)

# X5 C++ 深度推理

本目录运行五种已发布 X5 YOLO26 Depth BIN。
SDK 资源管理与前处理、forward、后处理、predict 分离。
保留源原生推理能力及三个位置参数入口，并新增带 NumPy 文件头的数组，方便离线评估。

<a id="supported-boards"></a>
## 板卡支持与验证状态

| 目标 | 变体 | 原生状态 |
|---|---|---|
| X5 | n/s/m/l/x，已校准 log-depth / NV12 | 已实现；纯主机与模拟 SDK 测试通过；真实 SDK 构建和板端推理 not-run |
| S100/S100P/S600 | 请使用 Python 运行时 | 源分支无对应原生深度实现；本程序显式拒绝 S 目标 |

身份规则与仓库注册表一致：优先 boardinfo `x5`；缺失时查 socinfo `x5u/x5h/x5m`；
再缺失才匹配设备树 `D-Robotics RDK X5 V1.0`。
已存在但未知的 boardinfo/socinfo 不继续回退。启动器和实际 SDK owner 都检查本机身份，
自定义模型路径不能绕过检查。

<a id="dependencies"></a>
## 依赖

需要匹配的 X5 Linux SDK，包含 `dnn/hb_dnn.h`、`dnn/hb_sys.h`、`libdnn`，
以及 C++17 编译器、CMake ≥3.16、OpenCV core/imgproc/imgcodecs、pthread、rt、dl。
源实现使用系统 DNN/OpenCV；本轮尚未使用真实 SDK 编译链接。
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
  -B samples/vision/yolo26_depth/runtime/cpp/build/x5 -DCMAKE_BUILD_TYPE=Release
cmake --build samples/vision/yolo26_depth/runtime/cpp/build/x5 --parallel 2
```

可通过 CMake 缓存变量 `DNN_INCLUDE_DIR`、`DNN_LIBRARY` 指定显式准备的 SDK。
单独编译成功不等于板端行为或运行时兼容性已验证。

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

也可直接调用二进制，包括源三个位置参数形式：

```bash
samples/vision/yolo26_depth/runtime/cpp/build/x5/yolo26_depth \
  --model-path /work/depth/model.bin --test-img /work/depth/input.jpg \
  --output /work/depth/native-direct --target x5
# 等价的源位置参数形式：
samples/vision/yolo26_depth/runtime/cpp/build/x5/yolo26_depth \
  /work/depth/model.bin /work/depth/input.jpg /work/depth/native-legacy
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
| 两者 | `--warmup` | 一次计时前的非负 forward 次数；原生默认 0 保留源行为，Python 默认 3 |
| 启动器 | `--binary` | 显式指定已编译二进制，不能与 `--build` 同用 |
| 启动器 | `--build` | 前置检查后显式配置、编译、执行 |
| 启动器 | `--dry-run`、`--list-models` | 互斥主机查看模式 |
| 二进制 | `--model-path` / `--model`、`--test-img` / `--input` | 显式模型与图像，不隐式选择模型 |
| 二进制 | `--help` | 不加载模型，只显示用法 |

源 X5 C++ 使用 SDK 默认调度，本实现不新增未经验证的核选择或优先级参数。
非法选项、输入或目标返回退出码 2。

<a id="interface-lifecycle"></a>
## API、阶段与资源生命周期

```cpp
#include "model_runner.hpp"
#include "yolo26_depth.hpp"
#include <opencv2/imgcodecs.hpp>

yolo26_depth::ModelRunner runner("/work/depth/model.bin");
yolo26_depth::Yolo26DepthTask task(
    [&runner](const auto& nv12) { return runner.run(nv12); });
cv::Mat image = cv::imread("/work/depth/input.jpg");
auto prepared = task.pre_process(image);
auto raw = task.forward(prepared.nv12);
auto result = task.post_process(raw, prepared.context);
// task.predict(image) 串联完全相同的三个阶段。
```

`ModelRunner` 必须比任务回调存活更久。每次返回独立拥有的原始 F32 数组，结果也拥有各自的 OpenCV 数组。
不要并发使用同一个 runner；并发客户需使用不同实例，这不构成 SDK 并发能力声明。
可选 C++ execution-gate 回调仅用于主机测试注入，不暴露为 CLI 绕过选项。

| 阶段 | 契约 |
|---|---|
| pre_process | 非空 BGR CV_8UC3 → 768 线性 letterbox，填充 114 → 独立打包 NV12 字节；逐次返回上下文，调用方不应修改 |
| forward | 唯一输入/模型/输出；SDK 调用和结构检查；返回已校准原始 log-depth，不做 exp、绘图或文件写入 |
| post_process | 有限 192×192 F32 → exp、放大到 768、按上下文去 padding、还原原图 |
| predict | 串联同样三个阶段，任务逻辑不包含预热或计时 |

几何取整与 Python 的 ties-to-even 一致；极端长宽比导致某一缩放尺寸为零时明确报错。
SDK 输出必须是 F32/NONE、NHWC 或 NCHW 单通道 192 方形。
正字节 stride 或由 aligned shape 推导的 stride 都会与分配容量核对，支持读取带 padding 的输出。
输入要求紧凑 NV12 pyramid 几何；带 padding 的逻辑输入尺寸显式不支持，不会静默错误复制。

SDK owner 在正常和异常路径释放 packed model、已分配内存及每次任务句柄。
绘图位于 `image_io.cpp`，CLI/计时和序列化位于 `main.cpp`/`cli_io.cpp`。
旧组合类 `Yolo26Depth::Infer` API 改为 runner 加 task；归档源代码仍保留旧 API。

<a id="results-interpretation"></a>
## 输出与验证边界

成功的原生执行写出：

- `log_depth.npy`：独立 float32 192×192 已校准 log-depth。
- `depth_native.npy`：float32 原图 H×W 相对深度。
- `depth_native.f32`：同样深度的小端行优先 F32 裸数据，保留源能力；尺寸见报告。
- `depth.png`、`overlay.png`：插值计算的 2%/98% 分位范围与反向 TURBO；原图权重 0.45，深度颜色权重 0.55。
- `report.json`：实际模型名、路径、形状、预热与计时；未另行记录时 SDK 版本为 unknown。

启动器增加 `launch-report.json`，记录精确命令、UTC 起止时间、退出码、模型/输入/二进制/
原生报告摘要及发布/自转换来源；已创建输出目录时另保存原生 stdout/stderr。
早期失败可能仅有 stderr，部分输出目录不表示成功。

计时包含一次完整 forward 的内存复制、缓存操作、SDK 调用及原始输出复制，
不能当作源 HRT 的纯 BPU 延迟。深度是相对值，颜色不是米；图像看起来合理也不等于精度通过。

主机测试编译纯几何、张量、CLI、序列化代码，并使用刻意精简的模拟 SDK 头编译真实 owner。
覆盖十三个 SDK 调用失败点、原始输出独立所有权、错误 metadata、padding/stride、身份优先级，
以及 NumPy 读取原生文件。**真实 SDK/OpenCV 构建、完整原生图像链路、板端推理和性能均为 not-run。**
