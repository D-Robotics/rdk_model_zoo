[English](README.md) | 简体中文

# HIMLoco C++ 运行

<a id="overview"></a>
## C++ 推理

在 X5 上离线运行融合后的 HIMLoco Go2 策略。六帧观测组成含 270 个数值的输入，推理返回十二个未缩放策略动作。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── tests/  # 自动化测试
├── CMakeLists.txt  # 源码或数据文件
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── application.cc  # 源码或数据文件
├── cli_io.cc  # 源码或数据文件
├── cli_io.hpp  # 源码或数据文件
├── launcher.py  # Python 脚本
├── main.cc  # 源码或数据文件
├── model_preflight.cc  # 源码或数据文件
├── policy.cc  # 源码或数据文件
├── policy.hpp  # 源码或数据文件
├── run.sh  # 运行示例
├── sdk_runner.cc  # 源码或数据文件
└── sdk_runner.hpp  # 源码或数据文件
```

<a id="supported-boards"></a>
## 适用板卡

原生运行时包含四阶段策略、SDK 适配器、离线 CLI 和构建启动器。实现 X5 源发布的
float32 输入／输出语义，不增加归一化、动作缩放或机器人控制。

| 目标 | 制品 | 状态 |
| --- | --- | --- |
| X5 | Bayes-e 融合 Go2 BIN | 支持（构建与运行见本指南） |
| S100／S100P／S600 | 无匹配发布制品 | 不支持 |

源板测环境为 RDK OS 3.5.0-beta、DNN Runtime 1.24.5、HBRT 3.15.55；
源测量数据见[评测说明](../../evaluator/README_cn.md)。

<a id="dependencies"></a>
## 依赖

- 主机核心：C++17、CMake ≥ 3.18；直接使用 `c++` 编译核心也可以。
- 原生可执行程序：X5 BSP 的 `dnn/hb_dnn.h`、`dnn/hb_sys.h`、libdnn，以及
  `nlohmann/json.hpp`（通常由 `nlohmann-json3-dev` 提供）。不需要 gflags 或 OpenCV。
- 主机 CLI 检查（`tests/test_cpp_cli.py`）：同样的 `nlohmann/json.hpp` 由共享的
  主机依赖发现（pkg-config 或标准系统 include 路径）解析。可用
  `NLOHMANN_JSON_INCLUDE` 显式指定 include 目录；无效的 override 会使检查失败，
  缺少该头文件的主机显式跳过。不回退到任何个人目录。
- Python 启动器：Python、NumPy、PyYAML，用于制品选择与板型检查；
  不加载 Python 推理 SDK。可通过 `PYTHON` 指定解释器。

使用发布模型不需要 Torch 或量化工具链。测试替身头只用于主机检查，不能替代真实 SDK。

<a id="build"></a>
## 构建

以下命令均在仓库根目录执行。无板卡时，仅构建核心并运行主机检查：

```bash
cmake -S samples/robotics/himloco/runtime/cpp -B /tmp/himloco-host \
  -DHIMLOCO_BUILD_TESTS=ON
cmake --build /tmp/himloco-host --parallel 2
ctest --test-dir /tmp/himloco-host --output-on-failure
```

板端启动器自动执行原生构建。若需要手动构建（X5 或配置好的交叉编译环境）：

```bash
cmake -S samples/robotics/himloco/runtime/cpp -B /tmp/himloco-native \
  -DHIMLOCO_BUILD_SDK=ON -DHIMLOCO_BUILD_CLI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/himloco-native --parallel 2
```

生成 `/tmp/himloco-native/himloco_cpp`。三个 `HIMLOCO_BUILD_*` 开关默认均为 OFF；
CLI 要求 SDK 开关为 ON。非标准 SDK 路径可通过 `HIMLOCO_DNN_INCLUDE`、
`HIMLOCO_DNN_LIBRARY`、`HIMLOCO_JSON_INCLUDE` 指定；交叉编译使用 CMake toolchain 文件。
CMake 不推断板型，运行时独立读取真实板型并检查模型摘要。

<a id="run"></a>
## 运行

任意主机可以预览，不下载、不创建构建目录、不加载 SDK：

```bash
bash samples/robotics/himloco/runtime/cpp/run.sh --help
bash samples/robotics/himloco/runtime/cpp/run.sh --list-models
bash samples/robotics/himloco/runtime/cpp/run.sh --target x5 --dry-run
```

显式准备发布模型后，在 X5 执行：

```bash
bash samples/robotics/himloco/model/download_model.sh --target x5
bash samples/robotics/himloco/runtime/cpp/run.sh --target x5 \
  --output-dir outputs/himloco_cpp
```

默认预热 10 次，处理 21 条观测，写入 21 个动作 BIN 和 `report.json`。
每次使用新输出目录；启动器不隐式下载。单文件、自定义输出与调度示例：

```bash
bash samples/robotics/himloco/runtime/cpp/run.sh --target x5 \
  --input-path samples/robotics/himloco/test_data/obs_history/000003.bin \
  --output-dir outputs/himloco_cpp_single --warmup 0 --priority 7
```

手工构建后可在仓库根目录直接运行 `/tmp/himloco-native/himloco_cpp`，默认路径同下表。
显式 `--model-path` 必须同时传入准确的 `--asset-id`，且仍需匹配发布摘要。

<a id="parameters"></a>
## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--target` | 启动器 `auto`；原生 `x5` | 实际推理仅 X5；主机 dry-run 必须明确 x5 |
| `--asset-id` | 启动器自动选择；原生固定发布 ID | `x5:himloco:himloco_go2_bayese_1x270.bin`；外部路径需显式指定 |
| `--model-path` | `samples/robotics/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin` | 启动器使用仓库绝对默认路径；直接二进制相对当前目录 |
| `--input-path` | `samples/robotics/himloco/test_data/obs_history` | 单个数字命名 BIN 或目录；启动器默认绝对路径 |
| `--output-dir` | `outputs/himloco_cpp` | 相对当前工作目录，必须新建 |
| `--report` | 输出目录下 `report.json` | 必须新建；可指定外部位置 |
| `--warmup` | `10` | 首条输入预热，范围 0..1000000 |
| `--priority` | `-1` | SDK 默认；或 0..255 |
| `--build-dir` | 本目录下 `build` | 仅启动器，原生构建目录 |
| `--dry-run` | `false` | 仅启动器，输出选择与构建／执行 argv |
| `--list-models` | `false` | 仅启动器，列出唯一发布制品 |
| `--help` | `false` | 显示用法 |

`--model_path`、`--input_path`、`--output_dir` 作为参数别名；
原生解析器接受 `--key value` 和 `--key=value`，拒绝重复及未知参数。
使用启动器时，自定义相对路径仍相对调用者当前工作目录。

输入必须为 1080 字节小端 float32，数值有限，文件名数字部分为源索引，重复索引拒绝。
若输入目录上一级存在 `runtime-input-manifest.json`，校验物理约定、索引、路径和逐文件摘要；
无清单的自定义输入可执行，报告明确记录空来源清单。输入约定见[输入说明](../../test_data/README_cn.md)。

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
报告及板型／模型身份检查由适配器和应用负责，SDK 生命周期、元数据及身份检查在 `SdkRunner` 中实现；`cli_io.cc` 和 `application.cc` 负责输入、输出和报告。

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
`input_metadata`／`output_metadata`、`model_name`、`runtime_version` 与
`priority` 为应用提供报告信息，不在推理文件中写报告。

<a id="results-interpretation"></a>
## 结果解释

12 维动作是未缩放的策略输出，不是机器人指令。源控制器在模型外部使用
`default_joint_position + 0.25 * actions`，本例不包含实时控制环。

核心传递每份 `RawOutputs` 自带的耗时，不添加前后处理时间；原生 runner 的计时范围为 `hbDNNInfer` 加 `hbDNNWaitTaskDone`，
不含缓存维护、输入复制、输出提取和文件 I/O。源测量数据及其口径见
[评测说明](../../evaluator/README_cn.md)。离线数值一致不能证明闭环行为。

代码遵循仓库 [Apache-2.0 许可](../../../../../LICENSE)。

### 输出文件与失败状态

动作文件按源索引命名，例如 `000003.bin`，每个 48 字节（12 个小端 float32）。
报告记录模型／输入清单摘要、每条输入输出摘要、SDK 元数据、预热完成次数、调度和计时。
`status: completed` 表示本次全部完成；`failed` 保留错误、当前源索引及已完成记录，
不声明整批延迟。成功汇总含 min/mean/p50/p95/max，mean 大于零时附顺序 FPS。
退出码 0 表示完成或帮助，2 表示执行或参数错误；启动器构建失败也返回 2。

文件以独占方式创建，不覆盖已有结果。模型或清单在执行中发生变化会导致失败。
进程强制中断或文件系统写入失败可能留下 `running` 或不完整报告，不能据此认定成功。
