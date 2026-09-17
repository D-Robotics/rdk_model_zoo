# P4–P6：CI、版本、Catalog 与 Skills 集成审计

审计日期：2026-09-16。结论：已有 Catalog 发布工程与文档站消费入口可继续复用，但 P4–P6 尚未完成。以下缺口是后续批次的验收事项，不是 P1 运行回归。

本次只读检查源码、Git 对象和本地目录，仅新增本报告；没有改 CI、Manifest、Skills、文档站或 Hub，没有联网、安装、构建、运行测试、板测、创建 Tag、发布或通知。报告中的“已有保障”表示代码中存在该检查，**不表示本轮执行通过**。

## Findings

| ID | 严重性 / 轴 | 批次与要求 | 证据、影响及最小后续动作 |
| --- | --- | --- | --- |
| F01 | major / Delivery specification | P4；Spec §9.3、H19 | `M:.github/workflows/model-catalog-data.yml:4` 的 PR/push 路径只有发布器、`platforms/**` 和本 workflow；`:42` 仅运行发布器 `npm run check`。只改根 `samples/**`、`docs/release/**`、根 README/AGENTS 或未来 `skills/**` 不触发这条 CI，也没有接入已存在的 Python 主机测试。按实际消费者补触发与作业；`develop` 目前也不在 push 分支列表中，需与集成分支策略一起明确。 |
| F02 | major / Delivery specification | P4；Spec §7.1–7.4、D09 | `M:tools/catalog-publisher/sources.json:6` 仍读三个平台树；`M:platforms/registry.json:5` 仍描述平台发布线；根 `docs/release/` 当前只有目标身份信息，未收敛模型/Benchmark/Schema。Catalog 仍用旧平台路径生成当前入口，未表示新 Sample 迁移状态、具体目标基础验证与本次证据。应保留旧身份与来源，逐批建立新入口映射和必要元数据的 Sample/发布器双消费者检查；不能只搬路径或新增第二份手写资产表。 |
| F03 | major / Delivery specification | P4/P6；Spec §7.4、§7.6、§9.5、H18 | 默认来源（包括 X3）是 worktree，并把链接 `ref` 写成 `main`（`M:tools/catalog-publisher/sources.json:9,16,23`），没有自动记录本次源码提交。通用证据校验只约束 `source.ref` 字符串形式，再读当前工作树的路径/标题；tag 模式直接跳过文件证据检查（`M:tools/catalog-publisher/src/pipeline/manifest-validation.ts:68,165,193`）。局部测试验证了部分历史提交/标题，尚不能证明每条引用的 `ref:path` 配对。发布候选需固定 X5/S 源码身份和 X3 历史来源，逐条检查对应版本中的路径；保留未知外仓证据状态。 |
| F04 | blocking（P5 验收） / Delivery specification | P5；Spec §8、D08 | 模型仓库没有 `skills/`，本地已检查来源仅找到 Device 的旧单技能 `rdk-model-zoo`，未取得七技能原始包。旧技能仍规定 “Branch = board”（`V:skills/rdk-model-zoo/SKILL.md:20`），脚本携带静态分支/指标表，不能原样覆盖统一布局。先核定七技能源、许可和版本；审阅实际工具后再复用，补单目录自包含和 AG01–AG15 行为结果。旧单技能存在不等于七技能已导入。 |
| F05 | major / Delivery specification | P6；Spec §9.1–9.3、D13 | 根 `VERSION`、`skills/VERSION` 和两类发布流程均不存在；当前构建器要求 `${platform}-v${version}` 与各平台 VERSION 一致（`M:tools/catalog-publisher/src/pipeline/multiplatform-catalog.ts:47,57`）。现有 workflow 没有 Tag 触发、严格对象解析、分开并发组/附件或 Skills `make_latest: false`。新增时还需改掉“只能有一个 workflow”的旧测试假设（`M:tools/catalog-publisher/tests/release-workflow.test.ts:20`），同时保留模型仓库不部署网站的边界。不得沿用平台版本校验冒充统一模型/Skills 隔离。 |
| F06 | major / Delivery specification | P4/P6；Spec §3.3、§7.6、§9.3 | 文档站锁定的是 `x5-v1.1.3+s-v1.1.2+x3-v1.1.2` 历史包，元数据明确为本地封装（`D:catalog/catalog.lock.json:5`、`D:catalog/public/data/catalog.meta.json:23`），不是本次统一源码候选。此外页面还融合文档侧 `docs-benchmarks.json`（`D:catalog/src/main.ts:70,105`）。准备候选时须对齐两份事实的来源和展示口径，完成新包导入、路由/指标差异检查；不能宣称 People 页面已与新 Manifest 完全一致。 |
| F07 | major / Delivery specification | P6；Spec §9.4、D13 | 固定 Hub `H` 的 `components.d/rdk-device.yml:4,5,42` 仍从 Device `v1.0.0` 分发旧技能，没有 Model Zoo 七目录注册。模型仓库也没有相应发布/通知入口。需在七目录进入已验证固定 Tag 后，准备一个保持目录唯一的维护源切换候选、旧源迁移说明及回退目标；本次没有执行切换或确认远端现状。 |

## 审计锚点与范围

以下别名用于本文的 `仓库:路径:行号` 引用；外仓均只读。固定提交是证据锚点，不代表已检查远端是否有更新。

| 别名 | 本地仓库 | 固定引用及状态 |
| --- | --- | --- |
| M | `D:/20_Dev_Projects/21_RDK_MODEL_ZOO/rdk_model_zooo_mccc/_worktrees/catalog-sample-audit` | 原 HEAD **`cd74a2b241075bb21036d8d0855d0403f8e8c963`**，分支 `develop`。工作树有 P1/P2 未提交工作；不能把这些文件归入该提交。所审 `.github/`、`tools/`、`docs/adr/` 相对 HEAD 无差异。 |
| D | `D:/20_Dev_Projects/21_RDK_MODEL_ZOO/rdk_model_zooo_mccc/model_zoo_doc` | HEAD `0781cd8acd8e065d5199e3aeb01a5db4f8b91b5a`，`preview/catalog-integration`；检查时工作树干净。 |
| H | `D:/20_Dev_Projects/RDK-Skills/rdk-skills` | 按 Spec 固定的 **`131d3048d5b1b8012b1383dc70be4f8264e25918`** 用 `git show` 读取，提交在本地对象库中，亦为本地缓存 `origin/main`。该目录实际检出 HEAD 是 `9c680248e804e67c46fd8c4c092af7a136c9994d`，不能把较旧工作树规则当作 Spec 固定规则。 |
| V | `D:/20_Dev_Projects/RDK-Skills/rdk-device-skills` | HEAD `3dcd1c3d0e303d785eed5bc0880f2d2cd1a4e29b`，`fix/historical-release-dispatch`。仅见未跟踪 `tools/__pycache__/`；所引技能和通知 workflow 无本地修改。它是可读源候选，不等于 Hub `v1.0.0` 镜像字节。 |

要求以 [本期 Spec](../../superpowers/specs/2026-09-16-rdk-model-zoo-x5-s-agent-people-spec.md) §7–9、§10.3、§11.1 为准。P1 的独立主机/板端结论仍见 [P1 验证报告](2026-09-16-pilot-validation.md) 和 [P1 审查](2026-09-16-p1-review.md)；本报告未重跑，也不扩大其适用范围。未审查正在实施的 P2 算法或整仓 Sample/Notebook 内容。

## Model Zoo Standards：现有入口与保障

### 模型仓库 CI 与发布器

| 真实入口 | 现有实现与边界 |
| --- | --- |
| `.github/workflows/model-catalog-data.yml` | 唯一 workflow。PR 与 `main` push 受路径过滤，另支持手动触发；`contents: read`，完整历史检出，Node 22，`npm ci` 后运行发布器检查。非 PR 运行上传 `catalog.json` 与 `catalog.meta.json`，保留 90 天（`:17,30,39,53`）。这是 Actions 构建制品，未配置 GitHub Release 或网站部署。 |
| `tools/catalog-publisher/package.json:9` | `validate:sources` → Vitest → `catalog:build`/TypeScript → `catalog:check` 构成现有 `check`。此脚本不会调用根 Sample 的 Python 测试、C++ 构建、Notebook/双语链接检查或 Skills 验收。 |
| `tools/catalog-publisher/scripts/validate-sources.ts:22` | 从同一来源配置解析 models/benchmarks，先验证发布数据，再应用既有修订并验证规范化结果；不访问板卡。 |
| `tools/catalog-publisher/src/pipeline/manifest-validation.ts:37` | 使用各来源自己的 JSON Schema；检查有限数、models/benchmarks Tag 一致、模型和 Benchmark ID 唯一、样本及单个 `asset_filename` 关联。工作树证据检查安全相对路径及 Markdown 的真实标题。外仓证据只要求 locator，不获取外仓；数组组件关系、新契约与本次验证状态不能据此宣称已覆盖。 |
| `tools/catalog-publisher/src/sources.ts:130` | 支持 worktree 或 annotated-tag 来源，`--pin platform=tag[:tree-prefix]` 保留历史根布局；`:174` 拒绝非 annotated tag。可用这条现有路径做固定包，不必另造生成器。 |
| `tools/catalog-publisher/src/pipeline/multiplatform-catalog.ts:378` | 保留每平台来源和 Manifest 摘要，Catalog 身份由内容指纹生成（`:476`），不会要求迁移前后字节 hash 不变。仍需审阅变化原因和稳定身份差异。 |
| `tools/catalog-publisher/src/artifact.ts:32,83` 与 `scripts/build-catalog.ts:80` | 确定性 JSON 序列化；元数据记录 payload SHA-256、字节数、Catalog 版本和来源；`--check` 比较重建字节，再调用统一校验函数。完整性检查不能替代来源提交锁定或板端验证。 |
| `tools/catalog-publisher/tests/` | 有资产/变体归并、总数和来源回归、平台注册、S 修订、X3 固定 Tag 修订、X5 原始证据、包篡改/截断/版本错误等测试。`benchmark-coverage.test.ts:236` 检查部分历史 SHA 对象存在，`x3-catalog-errata.test.ts:325` 检查固定 X3 标题；这些局部保障应保留，不能说发布器完全不校验证据。 |
| `tools/catalog-redirect/index.html` 与 `README.md:9` | 旧站到文档站的重定向候选已存在，保留查询参数/fragment，并有 `redirect-artifact.test.ts`。README 明确未发布；本次没有测试线上跳转。 |

根 `AGENTS.md:19` 明确 People/Agent 共用原生入口，不能以 AGENTS 存在推导七技能已完成；`:21` 列出现有三组 Python 主机测试。`docs/release/platforms.json` 是执行目标身份来源，`platforms/registry.json` 是旧平台发布目录注册；它们目前职责不同，不能合并含义后把身份识别写成支持或实测声明。

P4 不应删除原 Catalog 工程或历史修订测试。当前旧平台路径仍保留，本文没有认定它们已经断链；要在后续切换维护入口时同时完成相应引用、版本与测试更新。

### 文档站的真实消费链（D）

现有入口是 `catalog/scripts/import-catalog-data.mjs`，先通过 `verifySnapshot` 再写入 payload、metadata 和 `catalog.lock.json`（`:45,51`）。`catalog/scripts/catalog-data.mjs:36` 检查 hash、大小、锁定版本/平台 Tag/summary，并从 payload 重算模型族和 Benchmark 条数。`catalog/package.json` 的 build 先校验快照，再编译网站；根 `package.json:20` 先构建 Catalog 再构建 Docusaurus。

`D:.github/workflows/deploy.yml:3,23` 配置 main push/手动部署，并接入 Catalog 安装、测试和站点构建；`sync-gh-pages.yml` 在部署成功后或手动触发时同步页面。这里只核实配置存在，未确认这些 workflow 在远端启用、运行成功或线上内容已更新。

锁文件当前声明：

| 字段 | 值（读取所得，未在本轮重新校验 payload） |
| --- | --- |
| Catalog 身份 | `x5-v1.1.3+s-v1.1.2+x3-v1.1.2` |
| payload SHA-256 | `6a24b6c617b2fef21b4fd67cdafa41c7057aa176fc7e661cbd254bf5e2cf678c` |
| 大小 | 9,074,093 字节 |
| generator | `rdk-x5-catalog` / `docs/catalog` |
| metadata 来源 | `x5-v1.1.3` / `e11f434dc4a94a650b7c320fb464d07e7e583bff` 的历史 Catalog，本地导入封装；明确不是已发布 metadata 附件 |

文档侧另有 `catalog/scripts/export-doc-benchmarks.mjs:68`，从本仓库附录导出第二份数据并由页面合并；这没有改变锁定 payload 的字节，但页面事实范围已经大于该 payload。P4/P6 须明确这些文档测量如何对应权威 Benchmark、保留原文/单位/条件、区分只有证据与实际可下载资产。本文没有运行数据对照，也不把它误报为已经统一的模型数据发布链。

## Delivery Specification：P4–P6 尚需闭合

### P4 全局收敛

1. 从既有 X5/S Manifest、Schema、平台注册与已迁移 Sample 的真实绑定出发收敛元数据，不更改历史 ID/URL/hash，不把本地观测摘要升级为发布摘要。新入口与历史证据分别携带正确源码引用。
2. 调整发布器与受影响测试以读取统一来源，保留 X3 的固定历史展示；对基线与候选按身份比较缺失、重复、归并、目标适用、资产 URL/hash、Benchmark 数值/单位/条件及证据。现有总数/典型案例测试是基础，不等于本次全量差异验收。
3. 将根 Sample、共享库、必要元数据、README/数据集/第一方 Notebook 与兼容入口纳入相应 CI 影响范围。公共模块变化须执行真实消费者检查。C++、外部 gitlink 和未迁移能力按清点表分别验收，不以一个 Python 测试集代替。
4. 把资源可用、源码已迁移、具体目标基础验证和精度/性能证据分别呈现。当前 `ModelRecord`/`ModelVariant`（`M:tools/catalog-publisher/src/catalog/types.ts:55` 起）描述资产/硬件/Benchmark，没有接入本次迁移/运行报告。

### P5 七技能与已有来源

| Spec 名称 | 模型仓库交付状态 | 实际发现 |
| --- | --- | --- |
| `rdk-model-zoo` | 未导入 | V 与 H 镜像有旧单技能，版本字段为 `1.0.0`；不能据此宣称兼容新布局。 |
| `rdk-model-zoo-repo` | 未导入 | 检查的本地来源中未找到目录。 |
| `rdk-model-zoo-integrate` | 未导入 | 同上。 |
| `rdk-model-zoo-develop` | 未导入 | 同上。 |
| `rdk-model-zoo-validate` | 未导入 | 同上。 |
| `rdk-model-zoo-review` | 未导入 | 现有个人技能 `rdk-model-zoo-demo-review` 名称与交付职责不同，不能冒充该成员。 |
| `rdk-model-zoo-release` | 未导入 | 检查的本地来源中未找到目录。 |

本地搜索覆盖 M，以及 `D:/20_Dev_Projects/RDK-Skills`、`D:/20_Dev_Projects/29_Skills` 的技能源码和工作树（排除依赖、`.git` 与临时 smoke 镜像）。结果不证明未检查的远端或其他位置不存在源包。

V 的旧目录包含 `SKILL.md`、`skill-card.md`、两个 references、`scripts/branch_selector.py`、`scripts/benchmark_lookup.py` 和 `evals/evals.json`。它提供候选资料，但需要以下审阅后才能迁移：

- `branch_selector.py:24` 是固定板卡/分支表；`benchmark_lookup.py:12` 声明本地参考表为来源并内置指标。它们不是读取当前目标仓库 Manifest 的 `inspect_repo.py/read_catalog.py/validate_evidence.py`，本次没有取得后三个工具的七技能源实现。
- `SKILL.md:20,112` 和 `evals/evals.json` 仍以分支分板卡为主要导航；更新统一布局时还须保留对旧工作区的只读识别。不能把旧 eval 期望与历史计数直接沿用为新验收。
- 固定 H 的 `CONTRIBUTING.md:31,79,110` 要求 `skills/<name>/`、`evals/tasks.yaml` 和单技能自包含。Spec 要求共享内容在打 Tag 前同步进各成员目录；目前 M 没有共享同步、成员版本、自包含或安装检查入口。
- 七技能应分别记录结构检查、实际辅助工具测试与 AG01–AG15 代表行为结果；单技能缺同伴时提供原生入口或交接信息。既有单技能和个人审阅技能的存在均不能替代这些结果。

### P6 版本、分发与发布准备

本轮本地 `git tag --list` 只列出历史 `s-v1.0.0/1.1.0/1.1.1/1.1.2`、`x3-v1.0.0/1.1.0/1.1.1/1.1.2`、`x5-v1.0.0/1.1.0/1.1.1/1.1.2/1.1.3`，没有 `model-v*` 或裸稳定 `v*`。没有 fetch；该结果不是远端完整版本检查，不能据此指定首发号。

| 对象 | 现状 | 发布准备缺口 |
| --- | --- | --- |
| 统一模型源码 | 仅平台 VERSION 与平台 Tag 校验 | 根 VERSION、严格 `model-vMAJOR.MINOR.PATCH` 校验、源码/支持/验证范围、固定 Catalog 及专属附件/并发组。 |
| Skills Pack / 成员 | M 无 `skills/`、Pack VERSION/CHANGELOG 或成员发布流程 | 严格裸稳定 `vMAJOR.MINOR.PATCH`、成员版本与兼容布局说明、Tag 内文件自包含、Skills 禁止修改模型 Latest/Catalog 的测试。 |
| Catalog | 可构建并上传 Actions artifact；内容身份与 hash 已有实现 | 发布候选来源固定、干净检出可重建、变化清单、持久包身份与回退包。不能把 90 天构建产物保留期当成正式长期分发策略。 |
| 文档站 | 本地候选有导入器、锁、测试和部署配置 | 从实际统一模型候选导入，确认双语/历史深链接和所有指标差异；记录实际部署状态。当前重定向仅为候选。 |
| Hub | H 固定注册仍为 Device 来源，七目录候选不存在 | 核定源 Tag 后一次变更移除旧注册并增加新源七目录，验证唯一性、镜像、单目录安装/升级和回退；旧源保留迁移说明。 |

固定 H 已有可复用的分发保障，不应重造或绕过：

- `components.d/README.md:19` 与 `.github/scripts/release_contract.py:16` 要求规范稳定 `vMAJOR.MINOR.PATCH`，拒绝分支、预发布和数字前导零。这解释了 Spec 将裸 `v*` 留给 Skills 的原因。
- `.github/scripts/generate_plugin_catalog.py:117,127,168` 检查目录和技能名唯一性；`.github/scripts/sync-components.sh:208,219` 只稀疏取得注册目录，分阶段复制并保留回退逻辑。兄弟 `_shared` 不会因为同仓就自动分发。
- `.github/workflows/component-upgrade.yml:93,130` 验证 Tag 和发布事实后生成升级候选；旧源 V 的 `.github/workflows/notify-hub-release.yml:6,34` 有正式 Release 通知入口。它们尚未接入模型仓库，本轮没有执行。
- `.github/scripts/prune-orphans.sh:12,63` 对解析失败和超过五个 orphan 目录删除设保护；维护源切换必须沿用正常审阅流程，不修改镜像内容或绕过删除保护。

发布准备还需在拟发布提交的干净检出中验证实际交付对象：不能依赖当前未提交 P1/P2 文件、发布器现有 `node_modules/dist` 或工作区同伴技能；须记录精确子模块初始化路径，并执行代码/元数据/Catalog/Hub 固定版本的组合回退演练。这些验收本轮均未执行。版本前缀、附件、Latest 与无关对象触发的负例测试也仍待建立；任何外部发布均不在本报告授权范围内。

## Passed Checks（仅本次只读事实核对）

- 用 `git rev-parse HEAD`、`git status --short --branch` 固定 M/D/V 当前状态；用 `git cat-file` 和 `git show` 读取 H 的 Spec 固定提交，无切换/获取操作。
- 用 `git ls-tree -r --name-only HEAD -- .github VERSION skills docs/release tools/catalog-publisher/scripts` 确认 M 原 HEAD 只有现有 Catalog workflow，发布器两个脚本确实受版本管理，根 VERSION/skills 不在该提交；结合当前目录检查确认本轮仍未落地。
- 用 `git diff --stat HEAD -- .github tools docs/adr` 确认受审 CI/发布工程相对 M 原 HEAD 无修改；用 `rg` 检索并读取相关入口、测试定义、版本、来源和分发规则。
- 读取 D 的锁与 metadata，记录其声明的身份和来源；没有运行 hash 校验，因此不将“锁文件存在”写成“锁校验通过”。

本轮执行状态：

| 检查 | 状态 |
| --- | --- |
| 发布器 `npm --prefix tools/catalog-publisher run check` | **not-run** |
| 新触发路径、模型/Skills 前缀隔离、Latest 及附件负例 | **未实现 / not-run** |
| 文档站 `npm --prefix catalog run check`、新包导入及路由回归 | **not-run** |
| 七技能结构、辅助工具、AG01–AG15、单目录安装/升级 | **源包待核定 / not-run** |
| Hub 同步、候选 PR、远端 Release/Tag/线上状态 | **not-run / 未联网核实** |
| 干净发布检出、子模块交付、组合回退演练 | **not-run** |

既有 P1/Catalog 基线报告中的测试数仅属于原报告的命令和快照；本轮没有新增“95 项通过”或任何 Skills 历史成绩。

## Open Questions

- 七技能原始源包、其固定提交/Tag、实际工具和许可从何处取得？旧 Device 单技能已找到，但不是完整包。
- 完整历史与拟发布变更核对后，两种对象的首发版本分别是什么？本轮不指定版本号。
- 本地固定 H/D 与远端当前状态是否一致；新 Catalog 长期包的正式消费地址和文档站/Hub 切换候选是什么？须在获准准备相应外部集成时核实。
- 文档补充测量与权威 Benchmark 的归档/引用方式如何收敛，同时保留现有页面事实和历史原文？

## Verification Level

**static-reviewed**。所有结论来自固定 Git 对象和本地文件读取；未运行上述构建、测试、安装或发布链。P1 的 host/board 证据另行引用，不能升级本报告的验证等级。

## Overall Verdict

**needs rework / P4–P6 继续实施**。保留现有发布器、历史来源、数据锁与 Hub 校验，按 F01–F07 逐项闭合。完成本报告不代表已完成版本切换、七技能导入、Catalog 发布、文档站上线或 Hub 维护源切换。
