# README 内容契约（readme-contract）

> 状态：Phase 0.5 Q1 基线（2026-09-20）。适用范围：`develop` 统一 samples 架构下的全部
> sample 文档。旧交付分支（rdk_x5 / rdk_s / rdk_x3）的存量 README 不追溯适用本契约；
> 其修订以各自 ref 的规范为准。

## 交付文档口径（2026-10-07）

### 阅读顺序（2026-10-08）

以 [Sample 交付规范](sample-delivery.md) 为当前统一框架：模型介绍与来源在前，
当前目录结构其次，使用方法随后。Sample 根的 `directory` 紧随 `overview`；
conversion 的 `directory` 紧随 `source-model`；model 的 `directory` 紧随 `artifacts`；
evaluator 的 `directory` 紧随 `dataset`。Python/C++ runtime 以 `overview`、`directory`
开头，再分别进入环境/板卡、构建和运行说明。各层级的中英文模板同步采用此顺序。
下方内容表仍列必答内容，章节呈现顺序以本段和模板为准。

README 直接介绍当前交付物的用途、适用板卡、模型准备、编译、运行、输入输出和开发接口。
迁移经过、审计过程、测试是否执行、评审结论和发布提升状态放在独立维护记录中，不写入 README。
技术前提以具体操作表达：给出所需 SDK、工具链、校准数据、模型配置和真实可用的入口。
原有编译指南的步骤、命令、参数、版本、配置值、链接和图示须完整保留；目录调整时同步更新路径。
英文和中文遵循同一口径。下文的固定锚点继续有效；`known-gaps` 可显示为“补充准备”，
`boundaries` 可显示为“适用范围”，标题无需沿用历史命名。

## 行文与资料完整性（2026-10-08）

- 开头简短介绍模型及任务。输入张量、输出结构、板卡组合和 API 细节写在对应章节，不集中堆入导语。
- 说明面向开发者的操作和模型行为，不解释作者的编排选择，不加入迁移、审计或验收叙述。
- 每个 Sample 单独核查各层 README；不能以统一模板替代对实际入口、文件和参数的检查。
- benchmark 数值及其板卡、模型尺寸、SDK、数据集、测试方法、单位和条件完整保留；不同条件的表格分开呈现。
- conversion 完整保留来源中的步骤、命令、版本、配置、链接和图示。目录路径可按实际位置调整，不能用概要或历史链接替代配方。
- 历史分支资料的来源对照和核查结果保存到维护记录中；README 使用模型、板卡及工具链名称组织内容。

## 1. 目的与适用范围

本契约定义每个 sample 各级 README **必须回答的问题**与**可验证的验收要求**。根规范
（`docs/Model_Zoo_Repository_Guidelines.md`）链接本契约，不在别处重复维护同一要求；
Q3 检查器（`scripts/tools/sample_contract/check.py`）按本契约的固定章节 ID 与规则执行自动检查；
Skills（develop/review/validate）按本契约引导流程。三层各司其职：规范定义事实与要求，
模板承载内容结构，检查器执行可机器判定的规则。

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

每个章节标题前放置**显式 HTML 锚点**，ID 固定且与语言无关；跨文档引用与 Q3 检查均
以锚点 ID 为准（Markdown 自动生成的标题锚点随语言变化，不可依赖）：

```markdown
<a id="quickstart"></a>
## 快速体验（QuickStart）        <!-- 中文版 -->
<a id="quickstart"></a>
## Quick Start                   <!-- 英文版 -->
```

模板（`docs/sample-standards/templates/`）给出各级的全部固定 ID；章节按上述阅读顺序排列，
新增专题放在对应使用步骤之后，不复用既有 ID。删除某章节必须满足适用性要求（§1）。

## 4. 各级内容契约

每行“必答问题”都必须能从 README 正文找到直接回答；表格中的“证据”指该章节应当让
读者可核对的事实来源（命令输出、校验值、评估记录等），不是要求在本文件内嵌证据。

### 4.1 sample 根（`templates/sample.{en,zh}.md`）

| ID | 章节 | 必答问题 | 证据/适用条件 |
| --- | --- | --- | --- |
| `overview` | 算法与来源 | 这个模型做什么？算法一句话概述；官方论文/仓库/来源链接；在本仓的定位 | 官方链接可直接核对 |
| `support-matrix` | 支持矩阵 | 哪些 target（x5/s100/s100p/s600）× variant × 语言（python/cpp）提供实现和对应制品 | 列明可用组合及具体前提；测试状态另记 |
| `prerequisites` | 环境前提 | 需要什么板卡/系统镜像/工具链版本；python 依赖；磁盘与内存约束 | 版本号具体，不写“最新版” |
| `quickstart` | 快速体验 | **一条**从模型准备到看到结果的完整路径：每条命令给出 cwd、前置文件来源、参数、输出与成功判断 | 命令可逐条复制执行；使用实际脚本和目标选择参数显式准备模型 |
| `expected-results` | 预期结果 | 正常运行后应看到什么（Top-5 列表示例、检测框数量级、输出文件路径与命名） | 与 test_data 的真实输出一致；不虚构精度数字 |
| `directory` | 目录职责 | 子目录树 + 每项一句话职责 | 与实际目录一致（Q3 校验本地路径） |
| `entry-points` | 入口索引 | model/runtime/conversion/evaluator 各入口链接与一句话说明 | 链接有效；缺项写明理由 |
| `license` | 许可 | 模型权重与代码的许可约束；与仓库顶层 LICENSE 的关系 | 有专门许可必须显式给出 |

### 4.2 model（`templates/model.{en,zh}.md`）

| ID | 章节 | 必答问题 | 证据/适用条件 |
| --- | --- | --- | --- |
| `artifacts` | 制品清单 | 每个制品文件 ↔ target/stage 对应表（哪个文件用于哪个 `--target`、哪个 pipeline 阶段）；制品来源（下载/手动准备/manual） | 与 manifest 行一致；来源为 manual 时写明获取途径 |
| `preparation` | 准备步骤 | 下载命令（含实际目标选择参数与 cwd）或手动准备步骤；失败时的替代途径 | 下载脚本存在且参数一致；hash 校验行为说明 |
| `accompanying-files` | 伴随文件 | 词典/labels/mvn/config 等非模型制品的作用与必需性 | 每个文件一句话职责 |
| `local-paths` | 本地路径 | 准备完成后文件应位于何处；runtime 默认参数指向哪里 | 路径与 runtime 默认值一致（Q3 校验） |
| `formats-checksums` | 格式与校验值 | 每个制品的格式（.bin/.hbm/onnx 等）与已知 SHA-256；**未知校验值写 `sha256: null (unknown)`，禁止伪造或跨制品复制** | 校验值来源（发布记录）；null 时注明 |

### 4.3 runtime/python（`templates/runtime-python.{en,zh}.md`）

| ID | 章节 | 必答问题 | 证据/适用条件 |
| --- | --- | --- | --- |
| `environment` | 环境 | 板端系统要求、Python 版本、依赖清单与安装命令；`hbm_runtime` 仅板端可用的提示 | 依赖与实际 import 一致 |
| `usage` | 使用 | cwd；默认命令（零额外参数）与自定义命令各一条；成功判断标准（退出码/输出） | 命令与 `main.py` 实际行为一致 |
| `parameters` | 参数 | 全部 CLI 参数表：名称/类型/默认值/说明；**默认值必须与 parser 实际值一致**（Q3 从 `build_parser` 核对） | 无遗漏；kebab-case 命名 |
| `results` | 结果 | 输出字段/文件的位置、格式与含义；坐标/类别/置信度语义 | 字段名与代码返回一致 |
| `integration-example` | 集成示例 | **完整可运行**的 Python 片段：输入与配置变量全部定义、无未定义引用；优先演示具名模型类的 `predict`，按需要补充三阶段调用 | 由 sample tests 在 fixture 中验证（Q2）；不得含未定义变量 |
| `stage-io` | 三阶段 I/O | `preprocess` 输入→输出、`infer` 张量契约、`postprocess` 输出的类型与 shape 约定（与 inference-contract 一致的摘要） | 与 docstring 一致 |
| `troubleshooting` | 故障排查 | 常见错误（模型缺失/target 不匹配/输入尺寸）与处置 | 只列真实会踩坑的点 |

### 4.4 runtime/cpp（`templates/runtime-cpp.{en,zh}.md`）

| ID | 章节 | 必答问题 | 证据/适用条件 |
| --- | --- | --- | --- |
| `supported-boards` | 适用板卡 | 列出提供 C++ 实现的板卡、匹配制品和 SDK 要求 | 按实际组合描述；额外模型准备步骤具体可操作 |
| `dependencies` | 依赖 | 交叉编译/板端依赖（库、头文件路径、CMake 版本） | 版本具体 |
| `build` | 构建 | 完整构建命令序列（cwd、CMake 配置、make）；SoC 宏检测说明 | 命令完整可复制 |
| `run` | 运行 | cwd、默认与自定义运行命令、必需前置（模型路径） | 与 gflags 默认值一致 |
| `parameters` | 参数 | 全部 gflags 参数表（snake_case），默认值与代码一致 | Q3 对 cpp 参数做静态核对 |
| `interface-lifecycle` | 接口与生命周期 | 对外接口（config 结构/模型类）说明、资源分配/释放时序、线程边界 | 引用真实头文件符号 |
| `results-interpretation` | 结果解释 | 输出内容如何解读（文本/图像/退出码） | 与实际输出格式一致 |

### 4.5 conversion（`templates/conversion.{en,zh}.md`）

| ID | 章节 | 必答问题 | 证据/适用条件 |
| --- | --- | --- | --- |
| `source-model` | 源模型 | 源框架/权重版本/获取方式；与官方发布的对应关系 | 版本可追溯 |
| `toolchain-targets` | 工具链与目标 | OE 工具链版本、march、每个 target 的编译配置入口 | 版本具体 |
| `export` | 导出 | ONNX 导出步骤（环境、脚本、命令、产物） | 缺配方 → 移入 `known-gaps` |
| `calibration` | 校准 | 校准数据来源与数量、量化配置、校准脚本命令 | 数据集来源明确 |
| `compile` | 编译 | 生成 .bin/.hbm 的完整命令与产物命名约定 | 命令可复制 |
| `validation` | 检查编译产物 | 模型信息查看命令、板端运行命令和输出检查方法 | 给出可操作的步骤和预期输出；执行结果另记 |
| `artifacts` | 产物 | 产物清单与 target 对应、落盘路径 | 与 model/artifacts 一致 |
| `known-gaps` | 补充准备 | 需要用户准备的校准数据、外部配置或工具链示例，逐项给出获取及使用步骤 | 使用真实存在的脚本、配置和入口 |

### 4.6 evaluator（`templates/evaluator.{en,zh}.md`）

| ID | 章节 | 必答问题 | 证据/适用条件 |
| --- | --- | --- | --- |
| `dataset` | 数据集 | 数据集名称/版本/规模、获取与准备步骤（cwd、命令）、目录结构 | 版本可追溯 |
| `environment` | 环境 | 板端/主机、依赖、与 runtime 的关系 | — |
| `command` | 评估命令 | cwd、完整命令、参数表（默认值与代码一致）、预期耗时 | 可复制执行 |
| `metrics` | 指标 | 每个指标的定义与测试条件（topk、IoU 阈值、数据子集） | 条件完整，结果才可比 |
| `outputs` | 输出 | 输出文件位置与格式 | — |
| `reference-results` | 参考结果 | 已发布的参考值、模型/板卡/数据集条件及来源 | 只列有来源的数值；没有参考数值时给出结果获取步骤 |
| `boundaries` | 适用范围 | 列出实现支持的数据集、指标和输入条件；外部评估工具给出具体入口 | 与实际脚本或可执行程序对应 |

## 5. 统一内容纪律（全层级适用）

1. **命令五要素**：cwd、前置文件来源、参数、输出位置、成功判断。缺任一要素的命令块
   视为不合格。
2. **集成示例变量必须定义**：Python 示例中出现的输入/配置变量必须在示例内定义或指向
   明确的本地文件；“用户按步骤准备真实文件”是前提，须在示例前写明前置条件，不得把
   未定义变量当示例。
3. **禁止转嫁**：“同其他模型/参考原分支/见迁移记录”不能替代关键步骤；迁移历史、
   canonical/wrapper 审计说明放 `docs/releases/unified-migration/`，客户 README 只保留
   用户需要的实际调用接口与准备条件。
4. **操作真实**：转换与评估步骤使用实际脚本、配置和参数；外部工具、权重及校准数据给出
   具体获取和准备步骤，不用不存在的脚本或虚构产物充数。
5. **hash 纪律**：未知校验值一律 `sha256: null (unknown)`，禁止猜测、禁止从同模型其他
   制品复制。
6. **API 摘要**：公开 API 的 shape、dtype、布局、值域、坐标约定及异常在 docstring 中
   精确说明（见 inference-contract），README 给可理解的摘要与使用例，两者不得矛盾。
7. **支持与测试记录**：README 的支持矩阵描述当前实现、制品和适用条件。实际主机或板端
   测试记录包含提交、模型、环境、输入和结果，放在独立验证文档中；README 不写测试状态或验收说明。

## 6. 与旧规范（rdk_x5 `docs/Model_Zoo_Repository_Guidelines.md`）的冲突记录

本契约在 develop 生效；rdk_x5 旧规范在 rdk_x5 上继续有效。已识别冲突及处置：

| # | 旧规范条款 | 本契约处置 |
| --- | --- | --- |
| C1 | sample 根 QuickStart 以 run.sh“自动下载模型/自动构建/自动运行”为默认叙事 | develop 采用**显式模型准备**（`model/download.sh --target`）＋分步命令；run.sh 若存在可作为快捷方式并列给出，但 quickstart 必须先给显式路径 |
| C2 | `model/README.md` 仅要求“写清楚模型下载方式” | 扩展为 §4.2 五章节：制品对应、伴随文件、本地路径、格式与校验值 |
| C3 | conversion/evaluator README “暂无统一规范” | 本契约 §4.5/§4.6 给出完整契约；缺配方必须显式列 `known-gaps` |
| C4 | sample 根无支持矩阵要求；runtime 章节默认“同时提供 C++ / Python” | 强制 `support-matrix`（target×variant×语言的可用组合）；语言覆盖按实际声明 |
| C5 | runtime README “默认值必须与代码一致”（仅人工约束） | 保留并升级为 Q3 自动核对（parser `build_parser` 提取） |
| C6 | 代码文档指向 `docs/source_reference/` | develop 上该目录在 Phase 1（A6）前不存在；引用必须以实际存在路径为准，模板不预设该链接 |
| C7 | 顶层 README 规范描述 rdk_x5 目录树（docs/manifests 等） | develop 布局以本契约与根规范 develop 版为准；A6 合并时按 develop 布局改写目录章节 |

## 7. 参照覆盖检验（ResNet 单模型 / OCR 多阶段）

契约设计对照两个既有试点验证可表达性（试点文档按 Q4 重写后才宣称合规）：

- **ResNet（单模型）**：`support-matrix` 表达 x5+s100 / resnet18 / python+s100-cpp；
  resnet50/152 迁入后作为 variant 行扩展，无需新层级。`stage-io` 表达单阶段分类的
  Input→Tensors→Result。
- **paddle_ocr（多阶段）**：`artifacts`（model 层）以 stage 列表达 det/rec 两组成品对应；
  `stage-io` 表达 pipeline 编排（det→crop→rec），阶段错误归属在 Q2 tests 覆盖；
  `integration-example` 需给 pipeline.predict 级示例而非仅单阶段。
- **覆盖结论**：两级场景均可由 §4 章节集合表达，无缺口；OCR 场景要求 `stage-io` 章节
  允许按阶段分小节（模板已含该形态）。

## 8. 维护

- 本契约修订与模板修订同 commit；模板是契约的实例化，不允许模板先行偏离契约。
- 新增章节 ID 先改契约再改模板；Q3 规则同步更新（规则 ID 与章节 ID 解耦）。
- 历史版本见 git 记录；重大语义变更在 `docs/adr/` 记录。
