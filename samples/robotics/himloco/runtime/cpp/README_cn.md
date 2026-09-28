# HIMLoco C++ 策略核心

[English](README.md)

<a id="supported-boards"></a>
## 支持板型与迁移状态

本目录已实现不依赖 SDK 的策略阶段。原生 SDK 适配器已实现并通过显式替身检查，可执行程序及启动脚本
仍在迁移，当前尚未提供统一板端推理命令。原实现保留在
[源 C++ 说明](../../../../../platforms/x5/samples/robotics/himloco/runtime/cpp/README_cn.md)。
已实现的运行接口见统一 [Python 入口](../python/README_cn.md)。
主机策略测试不代表 SDK 或板端测试。

| 目标 | 发布制品 | 原生状态 |
| --- | --- | --- |
| X5 | Bayes-e 融合 Go2 BIN | 纯阶段主机测试通过；SDK 替身检查通过；CLI 待迁移；板端未运行 |
| S100／S100P／S600 | 无对应策略制品 | 不支持 |

源为 X5 提交 `ac115717197920355fc390bb04299b20e6436864`。保留源阶段语义：
270 个有限 float 观测输入、12 个有限 float 动作输出，不增加归一化、激活或动作缩放。

<a id="dependencies"></a>
## 依赖

纯策略核心仅使用 C++17 和标准库，编译不需要 libdnn、gflags、OpenCV、Python、Torch、
模型下载或转换工具链。以下主机检查使用本机 Apple Clang 执行，不证明板端 SDK 兼容性。
源历史板测环境为 RDK OS 3.5.0-beta、DNN Runtime 1.24.5、HBRT 3.15.55。

实际 `sdk_runner.cc` 还需要 X5 BSP 的 `dnn/hb_dnn.h`、`dnn/hb_sys.h` 和 libdnn；
`model_preflight.cc` 复用仓库 `samples/_shared/cpp/` 的板型识别与 SHA-256。
主机替身头只用于测试，不得作为生产 SDK 头。

<a id="build"></a>
## 构建主机契约检查

在仓库根目录运行：

```bash
build_dir="$(mktemp -d)"
c++ -std=c++17 -Wall -Wextra -Werror \
  -I samples/robotics/himloco/runtime/cpp \
  samples/robotics/himloco/runtime/cpp/policy.cc \
  samples/robotics/himloco/runtime/cpp/tests/test_policy.cc \
  -o "$build_dir/test_policy"
```

生成的是使用显式数值 runner 替身的无 SDK 测试程序，不构建或模拟训练策略及量化模型。

<a id="run"></a>
## 执行主机契约检查

保持同一终端，仍在仓库根目录运行：

```bash
"$build_dir/test_policy" samples/robotics/himloco/test_data/obs_history
```

预期打印四行 `passed` 并退出 0。最后一项读取随仓库提供的 21 条源观测，确认前处理
逐字节保留 float 数据；其余检查覆盖阶段内存独立、逐次计时、非法输入输出拒绝和
注入的传输异常传播。整个检查不加载 SDK。

<a id="parameters"></a>
## 参数与数据类型

统一原生 CLI 参数尚未提供。主机测试仅接收一个位置参数：随仓库提供的
`obs_history` 目录。库接口使用 [policy.hpp](policy.hpp) 中的类型：

| 类型／字段 | 约定 |
| --- | --- |
| `PreparedInput::values` | 独立持有的 270 个有限 float32 值 |
| `RawOutputs::actions` | 独立持有的 12 个有限 float32 值 |
| `RawOutputs::latency_ms` | 本次 runner 调用的有限非负耗时 |
| `InferenceResult` | 独立持有的原样动作和对应耗时 |
| `Runner` | `std::function<RawOutputs(const std::vector<float>&)>`；必需且非空 |

观测按当前 45 维帧在前、前五帧在后的顺序排列。调用者提供与训练一致的历史、裁剪、
缩放和关节顺序，策略核心不构造或更新历史。布局和来源见
[输入说明](../../test_data/README_cn.md)。

<a id="interface-lifecycle"></a>
## 接口与资源生命周期

`HimLoco(Runner)` 保存传入的可调用对象，不打开模型、不分配 SDK 缓冲区。
公开推理接口仅包含四个阶段：

```cpp
auto prepared = task.pre_process(observation);
auto raw = task.forward(prepared);
auto result = task.post_process(raw);
// 等价的完整路径：
auto complete = task.predict(observation);
```

`pre_process` 校验并复制观测。`forward` 校验输入，调用一次 runner，再校验动作和
耗时。`post_process` 返回独立副本，不裁剪、不归一化、不应用控制器的 0.25 rad
动作缩放。`predict` 顺序组合前三个阶段。

尺寸错误、NaN／Inf、非法耗时或空 runner 抛出 `std::invalid_argument`；runner 异常
继续传播给应用。接口返回值而不修改调用者输出参数，避免预测失败时将旧结果误当成
本次成功结果。

各阶段容器独立持有 vector，后续调用不会覆盖早先的原始输出或耗时；前处理与后处理
均不调用 runner。任务不保存可变的逐次推理状态，但线程安全仍取决于 runner，
不要并发使用同一个原生 SDK runner。SDK 加载、元数据、资源生命周期、输入输出文件、
报告及板型／模型身份检查由适配器和应用负责，SDK 生命周期、元数据及身份检查已在 `SdkRunner` 中实现；文件／报告／CLI 仍待迁移。

### SDK 适配器

`SdkRunner(NativeConfig)` 在构造时先读取实际板型，要求 X5，再校验显式本地 `.bin`
路径和发布 SHA-256；通过后才加载 SDK。没有下载、环境变量绕过或自动 S 平台回退。
`NativeConfig::priority` 默认 `-1`（SDK 默认），可指定 `[0,255]`；`model_path` 必须显式提供。

```cpp
himloco::SdkRunner runner({model_path, 7});
himloco::HimLoco task([&runner](const std::vector<float>& input) {
  return runner.run(input);
});
auto result = task.predict(observation);
```

runner 必须比引用它的 task 活得更久，二者不能并发使用同一 SDK 资源。
适配器要求单模型、单输入 `obs_history`、单输出 `actions`，float32 且无需手动反量化。
校验 X5 四维有效／对齐形状、元素数与分配容量后才复制数据。输入保留源实现的紧凑提交
方式并清零剩余缓冲区；输出按对齐跨度取出 12 个逻辑值。返回数据独立持有。
每次任务退出释放 task handle，构造失败和析构均先释放张量再释放 packed model。
`input_metadata()`／`output_metadata()`、`model_name()`、`runtime_version()` 与
`priority()` 为应用提供报告信息，不在推理文件中写报告。

主机 SDK 替身检查覆盖容量、对齐、类型／量化拒绝、分配失败、推理／等待／缓存失败、
非有限输出和资源回收；生产身份检查另用板型读取替身覆盖拒绝路径。
真实 SDK 编译与运行仍未执行，CLI 尚待完成。

<a id="results-interpretation"></a>
## 结果解释

12 维动作是未缩放的策略输出，不是机器人指令。源控制器在模型外部使用
`default_joint_position + 0.25 * actions`，本例不包含实时控制环。

核心传递每份 `RawOutputs` 自带的耗时，不添加前后处理时间；原生 runner 的计时范围为 `hbDNNInfer` 加 `hbDNNWaitTaskDone`，
不含缓存维护、输入复制、输出提取和文件 I/O。[评测说明](../../evaluator/README_cn.md) 保留源历史测量及其口径，
这些不是统一 C++ 的新结果。离线数值一致不能证明闭环行为。

代码遵循仓库 [Apache-2.0 许可](../../../../../LICENSE)。
