# RDK Model Zoo 全面合并 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在统一主线合并现有平台 Sample，完成 X3 重构、对应实机基础验收、七 Skills 接入与独立发布准备。

**Architecture:** 一个 Sample 表达完整部署流程，README 和现有脚本面向用户与 Agent 共用。公共运行代码放 samples/_shared，数据准备归 datasets，发布资料和既有 Catalog 构建归 docs。X3 原生 HBM 接口子集先通过独立决策门，再推广。

**Tech Stack:** 现有 Python/C/C++ Sample、目标板卡 SDK、待选择的 X3 Python 原生绑定、现有 TypeScript Catalog publisher、Markdown、已有 YAML Manifest、GitHub Actions。

**Spec:** [完整设计与需求](../specs/2026-09-16-unified-model-zoo-spec.md)，[逐源目录迁移表](../specs/2026-09-16-unified-model-zoo-migration-map.md)。两份均须在执行前读取。

## Global Constraints

- 本文仅实施计划，当前不执行代码迁移、板测、安装、推送或发布。
- main 统一维护，develop 分批集成；历史 tag 不改写。每 Sample 独立验收。
- X3 以现有已发布能力为范围，不新增其他平台独有模型。
- Python 为默认，保留现有 C/C++；Notebook 提取内容后从新主线全部移除。
- 完整源码检出、每 Sample 标准入口；不新增全仓总命令。
- 不新增 sample.yaml、agent.yaml 或等价执行配置；不预设顶层 utils/tools/scripts。
- 根/各级必要 README 和独立 datasets 保留；普通用户不依赖 Agent、Skills、完整量化环境或 Node 运行 Sample。
- 受影响平台/任务/协议代表规格必须对应实机跑通基础 pipeline 并检查结果；其余检查引用与配置。
- 不要求完整精度/性能测试；已有资产优先，转换或协议改变则验证新资产。
- 七 Skills 按既有设计归并，工具链能力复用；源码/Skills/Catalog/资产发布身份分开。

## 执行方法与变更边界

以下任务是可独立审阅的工作包。涉及尚未读取的目标 SDK、子模块和模型时，先形成所需接口/行为审计，再在该工作包中展开可执行代码步骤；禁止把本计划中的目标 API 当成已验证的 SDK 能力。每包内部按“固定基线 → 最小修改 → 有意义的检查 → 实机/证据 → 独立审阅 → 本地提交”进行，不把整个迁移压成一次提交。

原目录前缀均相对于模型仓库。新增路径是计划目标，不表示已经存在。文档仓库和 Hub 是另外的 Git 工作区，执行时确认路径和分支，不在模型工作区内伪造它们。

## Task 0：固定全仓能力和执行基线

**Files:** 读取 platforms/registry.json、三平台 release 清单、.gitmodules、各目录 README；更新本迁移表；在 docs/releases/unified-migration/README.md 记录执行基线及后续证据索引。

**Consumes:** Spec 与当前 a282b36。**Produces:** 每个目标 Sample 的来源提交、平台能力、代表模型和迁移类别；五板环境清单。

- [ ] 读取 git status、branch、worktree、submodule gitlink 和未跟踪内容；确认无工作被误覆盖。
- [ ] 对迁移表 89 行逐一核对 README、Manifest、资产和实际代码；记录同名不同义、缺失内容、发布能力与目录归属不一致项。
- [ ] 扫描全仓已跟踪 Notebook、数据/图片、路径引用和共用函数消费者，补充迁移表覆盖；不只统计 samples。
- [ ] 经授权获取五板连接，读取 OS/SDK/Python/可用磁盘与模型目录，选择独立测试目录；不在 git 或报告中写密码。
- [ ] 保存现有 Catalog payload、元数据及构建命令作为语义比较基线，记录历史证据链接。
- [ ] 主负责人审阅迁移清单、平台矩阵与受影响范围并本地提交文档基线。

检查命令（执行时运行，不是当前结果）：

```bash
git status --short
git branch --show-current
git submodule status
git ls-files '*.ipynb'
git ls-files platforms samples
npm --prefix tools/catalog-publisher run check
```

**Gate:** 目录/资产/能力有去向；不把不存在的硬件支持纳入承诺。**回退:** 本任务只读与文档，无需动运行实现。

## Task 1：审计 X3 原生 API 与兼容子集

**Files:** 新建 samples/_shared/runtime/x3/README.md；读取当前 yolo_runtime.py、yolo_input.py、X3 YOLOv8/分类源码；保存适用 SDK 来源引用与接口对照于该 README。

**Consumes:** Task 0 的 X3 环境与资产。**Produces:** Spec 6.3 的逐成员支持表，以及内存/布局/量化/错误处理规则。

- [ ] 收集目标 SDK 实際头文件/版本/链接库及许可，逐项确认加载、枚举元数据、内存准备、执行等待、输出读取、释放能力。
- [ ] 统计现有 HBM 消费者真正使用的成员，区分必需与平台特有可选项。
- [ ] 在文档中明确 model_names、形状含义、输出 dtype/量化、输入布局、同步范围及不支持参数；无法确认项作为失败条件，不发明默认值。
- [ ] 按 Python ABI、编译器与 SDK 选择最小绑定方式，记录所选依赖和构建/安装命令；此时才能编写相应原生实现。
- [ ] 审阅是否仍是有限兼容子集；若要求复刻完整 SDK，缩小范围或走回退路线。

**Gate:** 所有首版调用有目标 SDK 依据，无静默仿真。**回退:** 保留 pyeasy_dnn 过渡实现，其他平台迁移继续。

## Task 2：实现并验证 X3 原生兼容层

**Files:** 新建 samples/_shared/runtime/x3/hbm_compat.py、native/ 下实际绑定源文件、CMakeLists.txt、tests/test_contract.py、tests/test_board_runtime.py；构建文件名称与 Task 1 选择一致。

**Consumes:** Task 1 的接口和原生调用依据。**Produces:** 有明确版本环境边界的 HB_HBMRuntime 子集和可回退决定。

- [ ] 先编写输入缺失/多余、错误 dtype/shape、关闭后调用、未支持调度等契约检查，并确认尚未实现时失败。
- [ ] 实现模型句柄、设备内存和任务生命周期；失败中途清理，输出数组首版独立持有数据。
- [ ] 实现嵌套输入/输出字典和必需元数据；不把 X3 层安装成系统 hbm_runtime，未选 X3 时不加载原生扩展。
- [ ] 在 X3 对分类与检测代表进行旧/新路径同输入对照，保留每一处差异及判断；不能仅比较最终图片存在。
- [ ] 运行有限重复推理、错误输入后再次正确推理、模型关闭后读取既有结果、释放后重复关闭等资源检查。
- [ ] 检查普通用户安装方案；提供匹配验证环境的产物方式和源构建后备说明，确认 SDK 许可。
- [ ] 负责人审阅 Go/No-Go 记录；通过才进入广泛使用，失败就固定 pyeasy_dnn 薄适配路线，不默认两套并行维护。

目标接口消费者示例（计划契约，非当前可运行脚本）：

```python
runtime = HB_HBMRuntime(model_path)
name = runtime.model_names[0]
result = runtime.run({name: {runtime.input_names[name][0]: prepared_input}})
snapshot = result[name][runtime.output_names[name][0]].copy()
runtime.close()
assert snapshot.shape == result[name][runtime.output_names[name][0]].shape
```

板测用例应实际检查 result 中数组关闭后可读取且值不被下一次运行改变，不能用上述形状断言替代完整生命周期验证。测试输入通过模型 metadata 生成正确布局，不能固定所有模型为 NV12。

建议检查：

```bash
python -m unittest discover -s samples/_shared/runtime/x3/tests -p 'test_contract.py'
python -m unittest discover -s samples/_shared/runtime/x3/tests -p 'test_board_runtime.py'
```

第二条只在匹配 X3 环境与准备好模型时执行；缺少环境不能以 skipped 当作通过。具体模型输入由测试读取显式环境路径，测试不得自动下载全部模型。

**Gate:** Spec 6.5 全部满足。**回退:** 撤销默认新后端选择，保留审计材料，不删除原稳定实现。

## Task 3：完成跨平台代表 Sample

**Files:** samples/vision/ultralytics_yolo/runtime/python/{main.py,yolo_runtime.py,yolo_input.py,yolo_assets.py,yolo_dispatch.py} 与相关任务实现、README、tests；按迁移表建立一个已发布 X3 分类目标。

**Consumes:** Task 0 资产及 Task 2 后端决策。**Produces:** 同入口跨平台实际运行的最小闭环。

- [ ] 确认 X3 YOLOv8 正确资产，不沿用含 bayese 的旧默认参数；登记类别/输出顺序/scale 信息。
- [ ] 在平台选择处延迟导入对应后端；帮助、查询和下载 dry-run 不加载任一板端 SDK。
- [ ] 对旧 X3 输出做显式角色映射与一次反量化，审计 DFL、NMS、坐标恢复和默认阈值；证明等价部分才共用。
- [ ] 保留当前 X5/S 与 YOLO26 的行为差异，不将 X3 后端工作变成重写全部后处理。
- [ ] X3 分类和检测代表上板；若公共修改影响 X5/S，按实际消费者在对应板卡复验。
- [ ] 用 Paraformer、SigLIP、HIMLoco 的现有流程检验目录/入口规则，确保不引入单图像单模型假设。
- [ ] 主负责人分别审阅结果图、文本/张量输出和验证范围，记录每个 Sample 结论后提交。

现有可复用主机检查：

```bash
python -m unittest discover -s samples/vision/ultralytics_yolo/tests
python samples/vision/ultralytics_yolo/runtime/python/main.py --help
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --family yolov8 --task detect --dry-run
```

X3 命令支持在本任务实现后才进入 README 和测试；不能用未支持参数的计划示例宣称完成。

**Gate:** 真实代表结果正确，普通用户按 README 能运行。**回退:** 仅回退该 Sample 的新入口和适配，不影响已独立验收的其他 Sample。

## Task 4：归并跨 Sample 公共代码与依赖

**Files:** 从 platforms/{x5,s}/utils 中逐项分配到 samples/_shared/、对应 Sample 或 datasets；更新使用方 import、构建引用、README 和针对性 tests。

**Consumes:** Task 0 消费者列表、Task 3 的实际共用经验。**Produces:** 无平台全局重复依赖的公共能力。

- [ ] 比较候选函数签名、默认值、颜色/布局、数值类型和返回结构；记录相同/不同，不按文件名覆盖。
- [ ] 每次只抽取一个明确能力并更新全部已知消费者；单 Sample 专用工具保留原局部位置。
- [ ] 保证在不同 cwd 按 README 命令可解析仓库公共路径，不覆盖系统模块，不新增全仓运行框架。
- [ ] 对数值相关公共函数加入边界与原实现对照；纯移动使用引用检查和既有测试，不机械新增同实现测试。
- [ ] 在所有受影响平台/任务选择代表做基础复验，记录仅检查配置的其他规格。

检查：`rg -n 'utils|py_utils' samples platforms` 定位遗留消费者，结合语义确认；不能把所有字符串命中自动替换。

**Gate:** 没有意外跨平台 SDK 依赖或未迁移消费者。**回退:** 公共模块与消费者作为同一批次回退。

## Task 5：逐批迁移全部 Sample

**Files:** 精确源/目标路径以完整迁移表 89 行为准；包含 README、model、conversion、evaluator、runtime、tests 与必要小资源。

**Consumes:** 资产基线、平台后端、已验证公共能力。**Produces:** 每个目标 Sample 一个实现维护点和独立验收记录。

批次建议：分类 → YOLO 剩余任务及独立 YOLOv5/YOLOE → 分割/OCR/深度/追踪 → 语音/编码/策略 → LLM/VLA。板卡全部可用，不按设备缺失排除任何既有平台。

对迁移表中的每个目标 Sample，必须逐项完成：

- [ ] 固定所有来源版本，列平台/任务/模型规格、语言和独有功能；确认运行时和转换流程差异。
- [ ] 移入统一目标，合并相同行为，保留有根据的不同模块与默认值；不移除已发布能力来降低工作量。
- [ ] 更新下载、代码示例、评测和转换引用，保持发布资产地址与历史来源。
- [ ] 增加必要旧脚本转发，明确 cwd 和参数，避免旧入口维护第二套实现。
- [ ] 主机检查入口/配置/引用，对协议改动做针对性数值检查。
- [ ] 对应板卡、任务、协议代表基础执行并检查实际结果；保存命令和日志/图像/文本/张量证据。
- [ ] 负责人独立验收，更新迁移表“已实现/已验证/未覆盖”，单独提交。

**Gate:** 每个 Sample 都可独立批准或退回，不以整批成功代替单项验收。**回退:** 对应提交及转发入口同步恢复。

## Task 6：处理子模块与遗留非 Sample 资料

**Files:** .gitmodules、ACT/PI0 gitlink 及目标 samples/vla/{act,pi0}，docs/tros，平台 resource、LICENSE、历史发布说明。

**Consumes:** Task 0 清点。**Produces:** 外部来源完整、引用正常且无意外更新的迁移。

- [ ] 核对 ACT/PI0 当前锁定 SHA 与上游内容，保留各自版本；更新 gitlink 路径与 .gitmodules，不同时拉取最新代码。
- [ ] 审计子模块内相对路径、模型脚本和文档的仓库假设；需要修改上游时作为独立交付，不能直接在主仓伪造改动。
- [ ] 以资产/README/实机证据确认各子平台支持，不将 S 目录扩大解释为全部 S 板卡。
- [ ] 保留适用许可证与署名，迁移被文档/运行引用的 resource 小文件，处理 TROS 说明和实际集成内容。
- [ ] 在全新检出中按文档初始化需要的子模块，验证运行入口；普通用户无需为其他 Sample 初始化所有子模块。

检查：`git submodule status`、`git diff --submodule=log`、所有受影响路径链接。**Gate:** 原锁定来源没有无意变化。**回退:** .gitmodules 与 gitlink 原子恢复。

## Task 7：数据集、Notebook 和普通用户说明

**Files:** datasets/、各 Sample test_data/ 与 README、根 README.md/README_cn.md、samples 分类 README；移除迁移表列出的 X3 Notebook。

**Consumes:** Notebook 清单和数据用途审计。**Produces:** 无 Agent、无 Notebook 的完整使用路径。

- [ ] 对重复数据集比对版本、标签、用途和脚本，统一放 datasets；不移入 docs，也不全量提交下载数据。
- [ ] 对每个 Notebook 记录独有内容去向，迁移执行代码/参数和说明，确认无依赖隐藏单元状态。
- [ ] 将基础运行输入和必要参考结果放 test_data，完整评测/标定数据通过参数指定。
- [ ] 移除已完成内容迁移的 .ipynb 和引用，扫描跟踪树及实际子模块；不删除用户未跟踪 Notebook。
- [ ] 整理根 README 快速开始、模型与数据集导航；各 Sample 提供准确 cwd、命令、结果说明与支持范围。
- [ ] 人工按 README 在干净 shell 执行代表流程，不读取 AGENTS 或安装 Skills；验证依赖只按需安装。

检查：

```bash
git ls-files '*.ipynb'
rg -n '\.ipynb' README* samples datasets docs
```

最终跟踪 Notebook 应为零；历史说明中的 ipynb 文字允许存在但不能变成失效操作链接。**回退:** 迁移提交可恢复，历史 tag 不改变。

## Task 8：集中 Manifest 与迁移 Catalog 发布项目

**Files:** docs/release/{x3,x5,s}/models.yaml、benchmarks.yaml；docs/release/schemas/、platforms.json；tools/catalog-publisher → docs/catalog-publisher；.github/workflows/model-catalog-data.yml。

**Consumes:** 已更新 Sample 路径和原 Catalog 基线。**Produces:** 来源完整、可复现的新数据包。

- [ ] 保留 ID/URL/hash/历史 source_ref，将当前 Sample 路径迁为统一目录；相同 ID 语义冲突先审阅再处理。
- [ ] 将 registry 和 schema 集中维护，保留平台分片；更新构建 sources.json 和历史来源解析，区分当前路径和历史证据。
- [ ] 整项目移动 Catalog publisher，更新 package/lock 使用路径、CI 缓存与上传 artifact，保持 Node 依赖只用于发布开发。
- [ ] 扩展来源/迁移测试：旧来源仍可解析、新路径存在、无 ID/资产丢失、Benchmark 数值与来源没有无意变化。
- [ ] 生成新 payload，语义比较差异白名单；路径/版本导致 checksum 改变是正常变化，需要新包不能复用旧校验值。

检查：

```bash
npm --prefix docs/catalog-publisher ci
npm --prefix docs/catalog-publisher run check
git diff --check
```

**Gate:** 现有行为测试及新增来源测试通过，数据差异有解释。**回退:** 发布器、源路径配置、Manifest 与数据包锁定版本一起恢复。

## Task 9：文档站接入与旧 URL

**Files:** 文档仓库现有 Catalog 锁文件、数据导入脚本、模型入口和必要重定向；模型仓库旧 catalog-redirect 的维护位置及发布引用。

**Consumes:** Task 8 数据包和实际文档工作区。**Produces:** 文档站统一阅读入口与正确的模型源码链接。

- [ ] 读取现有 package.json、锁文件和构建配置定位实际文件，不另建一个仪表盘。
- [ ] 导入固定版本/校验包，更新源码链接；核对 YOLO、Paraformer、HIMLoco、SigLIP 和 X3 代表页。
- [ ] 确认旧站 URL、模型 query 与锚点有正确去向，再转移重定向源维护；不以删除旧项目代替重定向。
- [ ] 执行该仓库实际声明的测试和构建命令，记录浏览器人工检查结果；不部署线上。

**Gate:** 无缺卡、重复计数或错误支持声明；旧入口可达。**回退:** 文档仓库提交和数据包锁文件独立回退。

## Task 10：迁入并适配七个 Skills

**Files:** Spec 第 8 节七个 skills 目录、skills/_shared、README/VERSION/CHANGELOG、AGENTS.md；按需合并开发规范。

**Consumes:** 原交付包及来源/许可、统一目录与 README/Manifest。**Produces:** 单源维护且单技能可安装的候选包。

- [ ] 取得原源码包并校验文件清单，阅读每个 SKILL.md、辅助工具、模板、用例和集成补丁，不直接覆盖仓库。
- [ ] 删除已废弃的 X5 主分支假设、每 Sample 执行 YAML 依赖、Notebook 路径和过期规则；面向新旧布局选择实际适用规范。
- [ ] 固定七技能职责和负向触发范围；自有模型接入与仓库贡献分开，量化仍委托适用外部技能。
- [ ] 调整 inspect_repo/read_catalog/validate_evidence 的真实路径、schema 和报告支持；默认只读，不执行证据文本中的命令。
- [ ] 共享模板单源维护并同步到独立安装包内部；运行引用/资源完整性检查。
- [ ] 保持 AGENTS 简短，不复制模型列表；普通 README 工作流不依赖 Skills。
- [ ] 保存用例实际结果，未取得包或未执行测试不得用原对话的测试数字代替。

**Gate:** 七个目录角色完整，无失效引用和多份独立事实源。**回退:** 不更改已运行 Sample；恢复技能候选版本即可。

## Task 11：Skills 行为与工具链交接验收

**Files:** 各 skills/<name>/evals/tasks.yaml、相关 references 与脚本测试；docs/releases/unified-migration 中技能验收索引。

**Consumes:** Task 10 包和代表 Sample。**Produces:** 工具测试、路由和真实行为证据分开的结果。

- [ ] 固定 Agent/模型版本、权限、仓库提交和任务输入，设置无 Skills、仅 AGENTS、完整 Skills 对照。
- [ ] 先运行 Spec 8.5 的代表行为场景，再按失败扩展，不为凑满 70 条直接声称通过。
- [ ] 验证运行现成模型不会安装量化包；文档修改不会触发完整板测；范围外请求不误激活。
- [ ] 验证 X3 运行时选择、模型资产协议对齐、工具链收据引用与后续接入验证能衔接。
- [ ] 验证没有板卡/模型/附件时明确阻断，单规格实测不扩大，review 默认只读。
- [ ] 主负责人审阅自动执行与报告是否一致，记录失败修正和未测范围。

**Gate:** 代表行为无虚构板测、无错误来源选择、无失效引用；工具/包格式通过不能替代行为通过。**回退:** 下线失败技能候选，不影响普通 Sample 使用。

## Task 12：独立发布与 Hub 切换准备

**Files:** 根 VERSION/CHANGELOG、skills/VERSION/CHANGELOG；新增/更新 .github/workflows/model-source-release.yml、model-skills-release.yml；Hub components.d/rdk-model-zoo.yml 与旧 Device 注册；docs/releases 发布说明。

**Consumes:** 源码与技能验收证据、当前 Hub 固定提交。**Produces:** 可审阅的发布配置和迁移材料，不执行远端发布。

- [ ] 检查 tag 占用与 Hub 当前规则，按 Spec 建议分开 model-v* 和 v*，版本文件只验证所属对象。
- [ ] 添加工作流事件隔离检查：Skills tag 不发模型 Catalog，源码 tag 不触发技能注册，两个 Latest 解析不使用全仓默认 Latest。
- [ ] 生成统一源码平台支持/验证矩阵、已知限制、迁移说明和回滚步骤。
- [ ] 在 Hub 候选中一次性移除旧同名注册并新增七目录组件，源 tag 尚未存在时只保留候选，不发起假发布。
- [ ] 在隔离工作区运行适用 Hub CI 和安装检查，确认包按注册目录自包含，原来直接安装用户有迁移说明。
- [ ] 准备源发布→Hub 切换→旧源后续版本说明的顺序与独立检查点。

**Gate:** 所有发布动作可审阅、版本隔离已测；实际打 tag/推送/发布另获授权。**回退:** Hub ref 指回已存在不可变旧版本，保持唯一注册源。

## Task 13：整体交付审查与旧入口退役准备

**Files:** 迁移表、docs/releases/unified-migration、旧 platforms 转发入口、全仓 README/CI/子模块引用。

- [ ] 对照 89 个源目录与执行中新增发现项，确保每项有目标、保留依据或审阅过的处理结果。
- [ ] 核对 R01–R16 的交付与证据，没有仅主机验证却标为板端通过的 Sample。
- [ ] 检查未跟踪/用户文件未被删除，许可、资源、数据集和 TROS 等附属内容未遗漏。
- [ ] 确认无逐 Sample 执行 YAML、无跟踪 Notebook、无自动安装整套环境的默认路径。
- [ ] 明确旧转发保留清单与退役条件；本任务不因目录整洁提前删除仍在兼容期的脚本。
- [ ] 主负责人逐 Sample 签收；对未验证范围列清单，未满足最终完成定义时不宣布全仓合并完成。

## 需求覆盖与任务依赖

| 需求 | 任务 |
| --- | --- |
| R01/R02/R03 | 0、3、5、6、13 |
| R04/R05/R07/R08/R09 | 3、4、7、10、13 |
| R06 | 0、7、13 |
| R10 | 5、9、13 |
| R11/R12/R13 | 0、2、3、4、5、11 |
| R14 | 8、9、12 |
| R15 | 1、2、3 |
| R16 | 10、11、12 |

0 → 1 → 2 → 3 → 4/5；6/7 在来源清点后可逐批并行；8 随已验收迁移更新并在 5 完成后整体核对；9 依赖 8；10 可在目录规则稳定后与 5 并行；11 依赖 10 与代表 Sample；12 依赖来源/验证结果；13 汇总全部交付。

并行任务不能同时改同一公共模块、Manifest 或发布配置。负责人负责这些交叉文件的整合与最终验收。未给出固定工期：原生 SDK 可行性、Notebook 独有内容和实际差异审计完成后，按每包范围估算，不能把目录数量直接当成开发天数。
