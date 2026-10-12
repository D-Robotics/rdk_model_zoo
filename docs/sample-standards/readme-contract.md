# README 内容契约（readme-contract）


## 交付文档口径

### 阅读顺序

以 [Runtime 代码规范](runtime-code.md) 与本契约为统一框架：模型介绍与来源在前，
当前目录结构其次，使用方法随后。Sample 根的 `directory` 紧随 `overview`；
conversion 的 `directory` 紧随 `source-model`；model 的 `directory` 紧随 `artifacts`；
evaluator 的 `directory` 紧随 `dataset`。Python/C++ runtime 以 `overview`、`directory`
开头，再分别进入环境/板卡、构建和运行说明。各层级的中英文模板同步采用此顺序。
下方内容表仍列必答内容，章节呈现顺序以本段和模板为准。

README 直接介绍当前交付物的用途、适用板卡、模型准备、编译、运行、输入输出和开发接口。
README 面向使用者，不包含开发过程、实现意图、测试执行记录或评审结论。
技术前提以具体操作表达：给出所需 SDK、工具链、校准数据、模型配置和真实可用的入口。
原有编译指南的步骤、命令、参数、版本、配置值、链接和图示须完整保留；目录调整时同步更新路径。
英文和中文遵循同一口径。下文的固定锚点继续有效；`known-gaps` 可显示为“补充准备”，
`boundaries` 可显示为“适用范围”，中英文标题按用户操作命名。

## 行文与资料完整性

- 开头简短介绍模型及任务。输入张量、输出结构、板卡组合和 API 细节写在对应章节，不集中堆入导语。
- 用具体操作说明模型行为、所需依赖和输入输出。
- 每个 Sample 单独核查各层 README；不能以统一模板替代对实际入口、文件和参数的检查。
- benchmark 数值及其板卡、模型尺寸、SDK、数据集、测试方法、单位和条件完整保留；不同条件的表格分开呈现。
- conversion 完整保留来源中的步骤、命令、版本、配置、链接和图示。目录路径可按实际位置调整，不能用概要或历史链接替代配方。
- 使用模型、板卡及工具链名称组织内容；资料来源通过链接标明。

## 1. 目的与适用范围

本规范定义各级 Sample README 的内容、双语配对和固定章节 ID。
[仓库规范](../Model_Zoo_Repository_Guidelines.md)规定目录结构；本目录的 `templates/`
提供对应文档模板。

一个 sample 的文档层级与适用性：

| 层级 | 文件 | 必需性 |
| --- | --- | --- |
| sample 根 | `samples/<domain>/<name>/README.md` + `README_cn.md` | 必需 |
| model | `model/README.md` + `README_cn.md` | 必需（有模型制品即有） |
| runtime/python | `runtime/python/README.md` + `README_cn.md` | 有 python 实现即必需 |
| runtime/cpp | `runtime/cpp/README.md` + `README_cn.md` | 有 cpp 实现即必需；**未提供 cpp 时不得在任何层级声称双语言支持** |
| conversion | `conversion/README.md` + `README_cn.md` | 有 conversion 目录即必需；无配方时按 §4.5 写明缺失项 |
| evaluator | `evaluator/README.md` + `README_cn.md` | 有 evaluator 目录即必需；无评估实现时写明边界 |

`not-applicable` 判断只能基于上述适用范围并写明理由（如：`model/` 为 manual 准备、
`evaluator/` 仅有记录文档）；不允许用 not-applicable 替代 required 检查。

## 2. 双语配对规则

- 非仓库根目录的 README 第一行固定为语言切换栏：英文使用 `English | [简体中文](README_cn.md)`，中文使用 `[English](README.md) | 简体中文`；空一行后写正文标题。
- 每个层级的 `README.md`（英文）与 `README_cn.md`（中文）**成对存在**，覆盖相同的
  固定章节 ID 集合；英文/中文标题文字可以不同，**章节 ID 必须一致**（见 §3）。
- CLI 命令、参数名、默认值、支持矩阵（target×variant×语言）、结果与限制在两种语言中
  **必须对应一致**；允许表述不同，不允许范围不同。
- 只在一种语言中存在的章节视为该层级缺失该章节。
- 导航链接不能代替必要操作步骤：可以链接子 README，但 sample 根的快速体验路径必须
  自身可跟随（含模型准备与结果确认），不能只有一串链接。

## 3. 固定章节 ID 机制

每个章节标题前放置**显式 HTML 锚点**，ID 固定且与语言无关；跨文档链接以锚点 ID 定位章节（Markdown 自动生成的标题锚点随语言变化，不可依赖）：

```markdown
<a id="quickstart"></a>
## 快速体验（QuickStart）        <!-- 中文版 -->
<a id="quickstart"></a>
## Quick Start                   <!-- 英文版 -->
```

模板（`docs/sample-standards/templates/`）给出各级的全部固定 ID；章节按上述阅读顺序排列，
新增专题放在对应使用步骤之后，不复用既有 ID。删除某章节必须满足适用性要求（§1）。

## 4. 各级内容契约

各节应包含下表列出的信息。命令、接口和结果说明应与实际程序一致。

### 4.1 sample 根（`templates/sample.{en,zh}.md`）

| ID | 章节 | 必答问题 | 内容要求 |
| --- | --- | --- | --- |
| `overview` | 算法与来源 | 这个模型做什么？算法一句话概述；官方论文/仓库/来源链接；在本仓的定位 | 官方链接可直接核对 |
| `support-matrix` | 支持矩阵 | 哪些 target（x5/s100/s100p/s600）× variant × 语言（python/cpp）提供实现和对应制品 | 列明可用组合及具体前提 |
| `prerequisites` | 环境前提 | 需要什么板卡/系统镜像/工具链版本；python 依赖；磁盘与内存约束 | 版本号具体，不写“最新版” |
| `quickstart` | 快速体验 | **一条**从模型准备到看到结果的完整路径：每条命令给出 cwd、前置文件来源、参数、输出与成功判断 | 命令可逐条复制执行；使用实际脚本和目标选择参数显式准备模型 |
| `expected-results` | 预期结果 | 正常运行后应看到什么（Top-5 列表示例、检测框数量级、输出文件路径与命名） | 与 test_data 的真实输出一致；不虚构精度数字 |
| `directory` | 目录职责 | 子目录树 + 每项一句话职责 | 与实际目录一致 |
| `entry-points` | 入口索引 | model/runtime/conversion/evaluator 各入口链接与一句话说明 | 链接有效；缺项写明理由 |
| `license` | 许可 | 模型权重与代码的许可约束；与仓库顶层 LICENSE 的关系 | 有专门许可必须显式给出 |

### 4.2 model（`templates/model.{en,zh}.md`）

| ID | 章节 | 必答问题 | 内容要求 |
| --- | --- | --- | --- |
| `artifacts` | 制品清单 | 每个制品文件 ↔ target/stage 对应表（哪个文件用于哪个 `--target`、哪个 pipeline 阶段）；制品来源（下载/手动准备/manual） | 与 manifest 行一致；来源为 manual 时写明获取途径 |
| `preparation` | 准备步骤 | 下载命令（含实际目标选择参数与 cwd）或手动准备步骤；失败时的替代途径 | 下载脚本存在且参数一致；hash 校验行为说明 |
| `accompanying-files` | 伴随文件 | 词典/labels/mvn/config 等非模型制品的作用与必需性 | 每个文件一句话职责 |
| `local-paths` | 本地路径 | 准备完成后文件应位于何处；runtime 默认参数指向哪里 | 路径与 runtime 默认值一致 |
| `formats-checksums` | 格式与校验值 | 每个制品的格式（.bin/.hbm/onnx 等）与已知 SHA-256；未知校验值写 `sha256: null (unknown)` | 校验值来源（发布记录）；null 时注明 |

### 4.3 runtime/python（`templates/runtime-python.{en,zh}.md`）

| ID | 章节 | 必答问题 | 内容要求 |
| --- | --- | --- | --- |
| `environment` | 环境 | 板端系统要求、Python 版本、依赖清单与安装命令；`hbm_runtime` 仅板端可用的提示 | 依赖与实际 import 一致 |
| `usage` | 使用 | cwd；默认命令（零额外参数）与自定义命令各一条；成功判断标准（退出码/输出） | 命令与 `main.py` 实际行为一致 |
| `parameters` | 参数 | 全部 CLI 参数表：名称/类型/默认值/说明；**默认值必须与 parser 实际值一致** | 无遗漏；kebab-case 命名 |
| `results` | 结果 | 输出字段/文件的位置、格式与含义；坐标/类别/置信度语义 | 字段名与代码返回一致 |
| `integration-example` | 集成示例 | **完整可运行**的 Python 片段：输入与配置变量全部定义、无未定义引用；优先演示具名模型类的 `predict`，按需要补充三阶段调用 | 使用真实接口，列明所需本地文件 |
| `stage-io` | 三阶段 I/O | `preprocess` 输入→输出、`infer` 张量契约、`postprocess` 输出的类型与 shape 约定（与 runtime-code.md §9 一致的摘要） | 与 docstring 一致 |
| `troubleshooting` | 故障排查 | 常见错误（模型缺失/target 不匹配/输入尺寸）与处置 | 只列真实会踩坑的点 |

### 4.4 runtime/cpp（`templates/runtime-cpp.{en,zh}.md`）

| ID | 章节 | 必答问题 | 内容要求 |
| --- | --- | --- | --- |
| `supported-boards` | 适用板卡 | 列出提供 C++ 实现的板卡、匹配制品和 SDK 要求 | 按实际组合描述；额外模型准备步骤具体可操作 |
| `dependencies` | 依赖 | 交叉编译/板端依赖（库、头文件路径、CMake 版本） | 版本具体 |
| `build` | 构建 | 完整构建命令序列（cwd、CMake 配置、make）；SoC 宏检测说明 | 命令完整可复制 |
| `run` | 运行 | cwd、默认与自定义运行命令、必需前置（模型路径） | 与解析器默认值一致 |
| `parameters` | 参数 | 全部 CLI 参数，按实际解析器的拼写和默认值填写 | 与命令帮助一致 |
| `interface-lifecycle` | 接口与生命周期 | 对外接口（config 结构/模型类）说明、资源分配/释放时序、线程边界 | 引用真实头文件符号 |
| `results-interpretation` | 结果解释 | 输出内容如何解读（文本/图像/退出码） | 与实际输出格式一致 |

### 4.5 conversion（`templates/conversion.{en,zh}.md`）

| ID | 章节 | 必答问题 | 内容要求 |
| --- | --- | --- | --- |
| `source-model` | 源模型 | 源框架/权重版本/获取方式；与官方发布的对应关系 | 版本可追溯 |
| `toolchain-targets` | 工具链与目标 | OE 工具链版本、march、每个 target 的编译配置入口 | 版本具体 |
| `export` | 导出 | ONNX 导出步骤（环境、脚本、命令、产物） | 缺配方 → 移入 `known-gaps` |
| `calibration` | 校准 | 校准数据来源与数量、量化配置、校准脚本命令 | 数据集来源明确 |
| `compile` | 编译 | 生成 .bin/.hbm 的完整命令与产物命名约定 | 命令可复制 |
| `validation` | 检查编译产物 | 模型信息查看命令、板端运行命令和输出检查方法 | 给出可操作的步骤和预期输出 |
| `artifacts` | 产物 | 产物清单与 target 对应、落盘路径 | 与 model/artifacts 一致 |
| `known-gaps` | 补充准备 | 需要用户准备的校准数据、外部配置或工具链示例，逐项给出获取及使用步骤 | 使用真实存在的脚本、配置和入口 |

### 4.6 evaluator（`templates/evaluator.{en,zh}.md`）

| ID | 章节 | 必答问题 | 内容要求 |
| --- | --- | --- | --- |
| `dataset` | 数据集 | 数据集名称/版本/规模、获取与准备步骤（cwd、命令）、目录结构 | 版本可追溯 |
| `environment` | 环境 | 板端/主机、依赖、与 runtime 的关系 | — |
| `command` | 评估命令 | cwd、完整命令、参数表（默认值与代码一致）、预期耗时 | 可复制执行 |
| `metrics` | 指标 | 每个指标的定义与测量条件（topk、IoU 阈值、数据子集） | 条件完整，结果才可比 |
| `outputs` | 输出 | 输出文件位置与格式 | — |
| `reference-results` | 参考结果 | 已发布的参考值、模型/板卡/数据集条件及来源 | 只列有来源的数值；没有参考数值时给出结果获取步骤 |
| `boundaries` | 适用范围 | 列出实现支持的数据集、指标和输入条件；外部评估工具给出具体入口 | 与实际脚本或可执行程序对应 |

## 4.7 子目录 README 的操作覆盖

每个子目录先说明用途和输入输出，再给当前目录结构，最后给完整操作步骤。主 README 提供
一条完整快速运行路径；子 README 不能仅用“参考其他模型”代替本目录操作。

| 目录 | 必须说明的操作 |
| --- | --- |
| `model/` | 模型/板卡对应关系、下载或准备命令、伴随文件、最终保存路径 |
| `conversion/` | 原始权重获取、ONNX 导出、工具链环境、校准数据准备、量化配置、编译命令、输出模型路径及使用方式 |
| `runtime/python/` | 环境与依赖、模型准备、运行命令、完整 CLI 参数、结果说明、模型类 predict 和三阶段集成示例 |
| `runtime/cpp/` | SDK/编译依赖、构建命令、模型准备、运行参数、输出与调用接口 |
| `evaluator/` | 数据集准备、依赖、运行命令、指标定义、结果文件和读取方式 |
| `test_data/` | 示例输入的格式、来源、准备方式和运行入口 |

同一模型有多种板卡/工具链配方时分小节列出，完整保留每条原始配方的命令、依赖、版本、
配置、链接和图片，仅调整已经改变的路径。

## 5. 写作要求

1. 命令注明工作目录、前置文件来源、参数、输出路径和预期结果。
2. Python 集成示例定义全部输入与配置变量；需要本地模型或数据时，在示例前给出准备步骤。
3. 每份 README 包含本目录的关键操作。相关文档链接用于补充资料。
4. 转换与评估步骤使用实际脚本、配置和参数；外部工具、权重及校准数据列出获取方式。
5. SHA-256 来自对应模型文件的发布记录；未知时写 `sha256: null (unknown)`。
6. README 用示例和简要说明介绍 API；shape、dtype、布局、值域、坐标和异常与 docstring 一致。
7. 支持矩阵列出板卡、语言、模型及依赖条件。性能和精度数值附上模型版本、环境与测量方法。
