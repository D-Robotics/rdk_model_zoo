# Sample C++ Runtime 代码规范

提供 C++ runtime 的 Sample 与 Python 规范（[Runtime 代码规范](runtime-code.md)）同口径：
入口直接找到模型，模型文件内读懂输入如何变成结果。差异在于 C++ 默认以头文件公开契约、
以源文件隐藏 SDK，构建必须显式指定支持目标。本规范描述全仓库默认口径；各 Sample 按
其批准的架构交付，真实例外（共享后端、LLM 引擎）在下文对应条目标明。

## 1. 默认目录

```text
samples/<领域>/<模型>/runtime/cpp/
├── inc/
│   ├── <task>.hpp          # 模型契约：config/input/context/raw/result 类型与模型类
│   └── cli.hpp             # 参数、图像加载、渲染与结果/报告写入声明
├── src/
│   ├── <task>.cpp          # 模型：preprocess/infer/postprocess、张量契约、SDK 生命周期
│   ├── cli.cpp             # 参数解析、渲染、产物与报告 IO
│   └── main.cpp            # 薄入口：解析 → 构造具名模型 → predict → 保存
├── tests/                  # （位于 Sample tests/）编译生产源码的主机回归
├── CMakeLists.txt          # 支持目标门控的构建定义
├── README.md / README_cn.md
├── launcher.py             # 可选：清单选择与启动
└── run.sh                  # 定位入口，转发参数和退出码
```

`<task>.hpp`/`<task>.cpp` 按任务命名：`classify`、`detect`、`segment`、`depth` 等。
普通单任务 Sample 固定为上述三源文件 + 两头文件；确有独立职责的算法（复杂绑定、
多模型编排中的独立模型）才允许增加模型级文件，判断标准与 Python 规范第 6 节一致。
不设 application/dispatch 层，不为同一阶段引入转发包装。LLM 等以引擎/会话流程为
真实管线的 Sample 按其实际流程组织文件，不强行套用单任务三段式。

模型头文件默认只依赖 OpenCV 与标准库，声明本次交付实际支持的类型：配置（选项）、
Input、Context、RawOutputs、Result 五类概念与 Python 规范第 9 节同名同义。
SDK 句柄、张量与会话类型默认不进入头文件，放在源文件的私有 `Impl` 或匿名命名空间；
确有真实的共享后端边界（如跨 Sample 的 `backend.hpp`）或 LLM 引擎厂商 API 需要公开
类型时，公开实际所需的最小集合，并在头文件注释说明该边界的职责。

## 2. main.cpp：看得见的运行入口

入口依次完成：获取参数、构造具名模型、调用 `predict`、把结果交给 CLI 保存/渲染。

```cpp
const auto options = parse_options(std::vector<std::string>(argv + 1, argv + argc));
const auto image = load_image(options.image_path);
yolo26_depth::Yolo26Depth model(options.model_path,
                                yolo26_depth::DepthOptions{options.warmup});
const auto result = model.predict(image);
save_results(options, image, result, model.model_name());
```

- 模型构造与 `predict` 直接出现在入口中，不藏进 dispatch 或 application 包装。
- 入口不展开张量处理、解码、SDK 适配、计时循环或渲染数学。
- 入口级前置校验（如"输出目录必须新建"）保持在构造模型之前执行，先后语义不变。
- 模型 API 向调用方抛异常；入口只做顶层 catch，打印易读错误并返回非零退出码。

预热/重复推理等执行语义属于模型：以选项（如 `DepthOptions{warmup}`）传入，以结果
元数据（如 `RunMetadata{latency_ms, warmup}`）返回，调用计数与计时边界（计时只覆盖
一次完整 forward 还是纯 SDK 调用）在模型内声明并由测试验证。入口不自带推理计时
循环；CLI/evaluator 可以在端到端层组织多次模型调用做基准测量（如多流压测），但只能
调用模型公开方法，不得重写推理或计时语义。

## 3. 模型文件：完整的推理流程与资源生命周期

```cpp
PreparedInput Yolo26Depth::preprocess(const cv::Mat &image) const;
RawDepth Yolo26Depth::infer(const PreparedInput &prepared);
DepthResult Yolo26Depth::postprocess(const RawDepth &raw,
                                     const ImageContext &context) const;
DepthResult Yolo26Depth::predict(const cv::Mat &image);
```

- 构造函数建立可复用 Runtime：先执行该 Sample 声明的身份门（若运行时确实校验板卡，
  可注入 gate 供主机测试，默认实现读 `/sys`，在任何 SDK 调用之前执行），再校验模型
  文件与张量元数据（模型/输入/输出数量、几何、dtype、量化、stride 与容量），全部
  通过后才分配。构造失败不泄漏：已取得的句柄与缓冲由成员析构释放。
- `preprocess` 返回本次调用拥有的输入数据与 Context；`infer(prepared)` 完成上传、
  SDK 执行与原始输出拷贝，返回拥有的 RawOutputs；`postprocess(raw[, context])`
  返回拥有的 Result。可复用的物理 SDK 缓冲留在 `Impl` 内部，但任何阶段返回值都
  不被下一次调用改写；`predict` 显式串联各阶段，与逐步调用结果一致（tests 验证）。
- 保留原有数学：预处理字节、插值方式、letterbox/对齐取整、解码、默认值、dtype、
  输出集合与数值口径按批准的交付行为保留。README 只描述当前实际行为，不写迁移
  叙述；与源实现的差异记录在仓库外的评审证据中。
- 公开阶段名默认为 `preprocess`/`infer`/`postprocess`，不提供
  `pre_process`/`forward`/`post_process` 历史兼容委托；自回归 LLM 引擎等以
  生成/会话流程为真实管线的形态，按实际流程命名与拆分阶段。
- 所有 SDK 返回值都检查并转为携带操作名的异常；模型默认不打印诊断、不写文件、
  不做参数解析，确需日志或产物输出的运行时（如 LLM 引擎）在头文件注明实际行为。
- 元数据先验证后索引：任何 shape/stride/长度字段先通过契约检查再参与下标、
  乘法或 memcpy；溢出用安全乘法，动态 stride（-1）按平台规则解析后再验证。
- 任务句柄恰好释放一次：成功路径显式释放并检查返回码（失败要报出），异常路径由
  默认可构造、不可复制的 guard 兜底，绝不二次释放。

## 4. cli.cpp：使用方式与结果交付

CLI 集中维护参数解析、图像解码、结果渲染与全部产物/报告 IO。

- 选项命名与同 Sample 的 Python CLI 对齐（kebab-case；下划线别名仅当交付面确有
  提供）。实际命令面（是否支持 `--key=value`、位置参数形式、别名）以用户批准的
  交付 API 为准，并在 README 如实描述；不默认要求逐字保留旧旗标，也不混入未验证
  的调度或核心参数。
- 数值参数做完整消费校验（如 `stoi` 全量消费、非负约束），非法输入立即报错。
- 渲染（调色板、权重、归一化）与序列化（NPY/JSON/图像格式、精度、字段集）按
  交付行为原样保留；报告中的计时口径文字与模型声明的计时边界一致。
- CLI 不含模型或 SDK 逻辑，不做模型分发。

## 5. 构建：显式支持目标

- C++17；CMake ≥3.16；依赖按 Sample 实际技术栈声明（视觉类为 OpenCV 与板端
  DNN/UCP SDK，LLM 为其厂商 SDK），构建脚本不安装依赖。
- 构建必须显式指定该 Sample 实际发布的支持矩阵。缓存变量命名遵循 Sample 现状
  （`RDK_TARGET`，或既有的 `YOLO_TARGET`/`ASR_TARGET` 等），其描述文本与实际
  支持矩阵一致；`auto` 仅在原生（非交叉）编译时读取构建主机身份，交叉编译时直接
  拒绝（读取的将是构建主机身份）；SoC 字符串去空白、统一大小写后按枚举校验，
  未识别取值一律报错，不接受任意宏。仅当运行时确实读取板卡身份时才要求 `/sys`
  身份门，普通单任务 Sample 不强制新增。
- SDK 头文件/库路径可用标准缓存变量覆盖；独立编译成功不构成板端行为证明。
- 提供 per-target 构建目录的 Sample 按目标分离（如 `build/x5`、`build/s100`），
  启动器向 CMake 传递显式目标。
- 警告基线 `-Wall -Wextra`（可加 `-Werror`），与源实现一致；主机验证时若宿主
  工具链/第三方头文件触发额外告警，降级须在证据中记录为宿主兼容性说明。

## 6. Shell 与启动器

`run.sh` 定位 Python 启动器或二进制，原样转发参数并传播退出码；默认不复制参数表、
不安装依赖、不下载模型，Sample 实际提供其他行为时按实际描述。`launcher.py`
（如提供）负责清单选择、身份与制品校验、显式 `--build` 与 launch-report；
直接调用二进制时只验证板卡与张量契约、不验证发布方身份的范围也按 Sample 实际
支持的调用方式描述。

## 7. 注释与接口说明

- 模型头文件注释说明任务、构造参数、支持输入、Runtime 生命周期（谁加载、何时释放）
  与线程性（默认不承诺线程安全）。
- 阶段方法与类型注释标明 shape/dtype/布局/值域、坐标格式、是否过 softmax、
  计时口径与单位；解释算法依据与板卡差异，不写开发经过或迁移叙述。
- 注释与实现同步更新，失效说明必须删除。

## 8. 代码验收

- 入口可直接定位模型构造与 `predict`；模型源文件可读到完整阶段与 SDK 生命周期。
- 发布模型的默认参数、CLI 选项、前处理字节、输出变换与后处理数值与批准的交付
  行为一致。
- 错误板卡（适用时）、错误 shape/dtype/stride、无效输入与 SDK 异常转为异常并正确
  报告；构造失败、分配失败、任务失败均无泄漏、无二次释放。
- 重复预测与交错调用不复用上次输入的几何状态；返回数据拥有独立生命周期。
- CMake 门控按第 5 节执行；README（中英同步）描述实际交付的目录、接口与依赖，
  不含迁移叙述，不承诺未验证的依赖充分性。
- Sample tests 至少覆盖（按实际依赖裁剪）：`predict` 与显式阶段组合一致；固定
  fixture 下 `infer` 消费显式传入的 prepared 数据并返回拥有的原始输出（经伪 SDK
  或真实依赖执行）；交错调用时 Context 只描述当次输入、上次输出不被改写；坏
  stride/坏元数据被拒绝；注入失败时已持有资源与任务恰好释放一次、成功路径释放
  失败可传播并可恢复。原有数值断言保留，不得以删除测试消除回归。

## 9. 主机验证与证据

- 主机测试编译**生产源码**（模型与 CLI 源文件），配合与该 Sample 实际依赖匹配的
  替代（仓库内伪造 SDK 头——仅测试 include 路径，绝不进入正式构建——或真实
  OpenCV/第三方库）；不以纯语法检查冒充输出等价证明。真实 SDK/板端推理与模型
  编译未执行时如实记录 not-run。
- 涉及资源生命周期的测试使用带计数与故障注入的伪 SDK：逐调用点失败，断言
  分配/释放、任务创建/释放计数平衡；测试代码在重置伪 SDK 前销毁全部模型实例。
- 最终证据保存完整受检的编译命令与退出码、可执行文件先删后建、运行输出与
  退出码；失败尝试的过程证据保留，不以覆盖后的通过日志替代。
