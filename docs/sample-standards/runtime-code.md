# Sample Runtime 代码规范

Sample 应让开发者从入口直接找到模型，并在模型文件内读懂输入如何变成结果。目录和接口围绕这条阅读路径组织。

## 1. 默认目录

```text
samples/<领域>/<模型>/
├── model/                  # 编译后模型的获取、制品和板卡对应关系
├── conversion/             # 原始权重、导出和编译配方
├── runtime/
│   └── python/
│       ├── main.py         # 参数 → 构造模型 → predict → 展示结果
│       ├── cli.py          # 参数、发布模型选择、列表、结果展示与保存
│       ├── classify.py     # 模型类及完整推理流程；按任务命名
│       └── run.sh          # 定位入口，转发参数和退出码
├── evaluator/              # 数据集评估及指标
├── test_data/              # 可直接运行的输入示例
└── tests/                  # Sample 的行为与数值回归

utils/
├── py_utils/               # 跨模型复用的 Python Runtime 和工具
└── c_utils/                # 跨模型复用的 C/C++ 能力

utils/tools/              # 可独立执行的维护程序
```

模型文件按职责命名，例如 `classify.py`、`detect.py`、`segment.py`、`matching.py`、`policy.py`。普通单任务 Sample 默认采用三个 Python 文件。多任务家族按任务划分模型文件：Ultralytics YOLO 覆盖 detect / segment / pose / classify / obb 五类任务，目标形态是每任务一个模型文件加 `main.py` 与 `cli.py`，合计约 7–8 个 Python 文件；确有完整职责的后端绑定可另立一个文件并说明其角色。已存在的 `pre_process`/`forward`/`post_process` 兼容名保留为实际行为的委托，不为满足命名一致性新增包装层。不按每个函数拆文件。

## 2. main.py：看得见的运行入口

入口依次完成四件事：获取参数、构造具名模型、调用 `predict`、交付结果。

```python
args = build_parser().parse_args(argv)
selection = resolve_selection(args.target, variant=args.variant)
model = ResNetClassifier(selection.model_path, target=selection.target)
result = model.predict(args.test_img)
print_result(result, selection, args.top_k)
```

该示例展示入口结构；实际入口还须传递所选模型的输入尺寸、类别数、插值、分数策略等默认参数。完整参数由各 Sample 的 CLI 定义。

- 保留清晰的函数调用和必要的分支，不在入口展开张量处理、解码或 SDK 适配。
- 模型构造和 `predict` 调用直接出现在入口中，不藏进通用 dispatch 或整套 application 包装。
- 参数较多时在 CLI 中组织；不把大量配置代码移到 Shell。
- 异常在 CLI 边界转换为易读错误及非零退出码；模型 API 向调用方抛出实际异常。

## 3. 模型文件：完整的推理流程

具名模型类负责初始化，并在本文件内实现三个阶段及其组合：

```python
class ResNetClassifier:
    def predict(self, image):
        inputs = self.preprocess(image)
        outputs = self.infer(inputs)
        return self.postprocess(outputs)
```

- 模型类通过 `__init__` 或具名构造方法（如 `Model.from_model(...)`）接收模型路径、目标板卡及必要的模型参数，建立可复用的 Runtime。具名构造方法返回可直接调用 `predict` 的模型实例；`main.py` 不展开 runner 创建、模型加载或张量绑定。
- `preprocess` 展示该模型的输入准备步骤。
- `infer` 使用公共 Runtime 或明确的本地 SDK 适配执行模型。
- `postprocess` 展示输出解释、解码和结果构造。
- `predict` 显式串联三个阶段，返回结构化结果。
- 预测过程不承担参数解析、下载模型、安装依赖、打印或保存文件。
- 单模型路径接口允许用户传入自己的编译模型和实际张量参数。发布目录的模型枚举和制品选择属于 CLI。
- 多模型任务明确说明每个模型的输入输出和连接方式；语音等任务可使用表达实际数据流的阶段名。

模型特有的算法和默认值保留在 Sample。公共函数可以直接调用；调用方不必经过 Sample 内额外的转发模块。

## 4. cli.py：使用方式与结果交付

CLI 集中维护参数、发布模型及板卡选择、模型列表、预览、标签加载、结果格式化和保存。

- 同一选项只声明一次，`main.py` 和 `run.sh` 使用同一套参数。
- 帮助、模型列表和预览按各命令声明的依赖执行，不触发板端 SDK 加载或实际推理。
- `target=auto` 使用硬件身份识别；显式 target 对应明确的模型制品与 SDK。
- 实际执行前校验目标板卡和模型元数据。板卡切换不改变模型算法含义。
- 参数只影响其声明的行为，例如输入大小、插值、置信度、Top-K、保存路径和调度设置。

## 5. 公共 Runtime 与工具

Python 公共能力统一放在 `utils/py_utils/`，C/C++ 公共能力统一放在 `utils/c_utils/`。

- SDK 会话负责模型加载、板卡校验、执行及资源生命周期。
- 通用图片读取、标签校验、NV12 转换、张量校验和数学操作按完整职责组织。
- 公共库不依赖某个 Sample 的入口，也不隐藏整套模型前后处理。
- 多个消费者确实共用的能力再抽取；模型专用代码留在模型目录。
- 删除只转发 import、继承后原样转发构造参数的本地文件；直接引用公共实现。
- 保持轻量导入，避免查看参数时连带加载图像库、训练框架或板端 SDK。

## 6. 允许增加文件的情况

额外文件应有独立且完整的职责。例如分词器、跟踪算法、语音前端、CIF、多任务解码或复杂的物理张量绑定。阅读一个普通模型时不应为同一阶段跨越多层包装。

简单的结果展示归入 `cli.py`，纯 import 转发或参数透传不单独建文件。

增加文件时检查：

1. 是否承载完整算法或独立资源生命周期？
2. 是否需要单独阅读、测试或复用？
3. 拆分后，模型文件是否仍能说明完整的数据流？

文件数和行数用于发现阅读负担，不作为机械合并算法的指标。

## 7. Shell

`run.sh` 定位 Python 入口，原样转发参数，并传播退出码。工作目录变化不应使入口失效。需要选择解释器时提供明确的环境变量或文档约定。

Shell 不复制 Python 的参数表、模型列表和板卡选择规则，不自动安装依赖或下载模型。模型准备通过 `model/` 中的独立命令完成。

## 8. 注释与接口说明

采用 Google-style docstring，说明模块和类的职责、参数、返回值及影响调用者的异常。张量接口写明 shape、dtype、布局和值域；输出说明类别、坐标、分数或序列的语义。

行内注释解释关键算法或平台约束。代码和注释同步更新，普通转发无需逐行复述。

## 9. 阶段名与数据流契约

公开业务接口统一为 `preprocess` → `infer` → `postprocess`，由 `predict` 串联；旧名
`pre_process`/`forward`/`post_process` 保留为同一实现的薄兼容委托，静态检查器对两套
拼写同等扫描。`postprocess` 是否接收 context 由任务是否消费几何信息决定；`predict` 的
串联语义与三步显式调用的结果一致性必须在 tests 中验证。

每个任务在 docstring/类型定义中具体化五个概念：Input（业务输入）、Tensors（后端物理
输入映射）、Context（本次调用的几何/状态信息）、RawOutputs（结构/metadata 校验后的
原始输出）、Result（业务结果）。Context 必须显式存放在 `prepared` 返回值中，不允许放
入会被下一次调用覆盖的实例字段；有状态任务（跟踪、语音流、生成式）显式声明
session/reset 语义，不虚称线程安全。

多阶段任务（OCR/SAM/Paraformer）：每个 stage 具备公开的 preprocess/infer/postprocess
三步接口，`pipeline.predict` 显式编排；stage composer 与逐阶段 helper 合法，但阶段次序
在代码中直接可读，不为凑函数数把下一阶段推理藏入上一阶段 postprocess。仅有原生 C++
运行时的 sample（如 `samples/llm/*`）以其 README 声明的原生等价接口为准。

## 10. 代码验收

- 入口可直接定位模型构造与 `predict`，模型文件可读到完整阶段。
- 发布模型的默认参数、CLI 选项、前处理字节、输出变换和后处理数值保持一致。
- 自定义模型参数经过实际张量契约校验。
- 错误板卡、错误 shape/dtype、无效输入和 SDK 异常仍能正确报告。
- 重复预测不复用上次输入的几何状态，结果的生命周期清楚。
- 更新移动后的 Python 导入、测试 patch 目标、评估器与维护工具引用。
- 保留有意义的数值和行为断言，不能靠删除测试消除回归。
- 主机注入测试、真实板端推理和模型编译分别记录其实际执行结果。

Sample tests 至少覆盖：predict 与显式三步逐字段一致（含旧名委托等价）；注入固定
fixture 时 infer 只做容器校验、无解码/激活/文件读写；两种尺寸输入交错调用时
context 只描述当次输入；binding 声明之外的数值变换使测试失败。多阶段 sample 另需
覆盖零检测短路、多 crop 次序稳定与阶段错误归属。

接口和目录变化另附映射供文档维护者更新示例；README 按当前交付接口介绍使用方法。
