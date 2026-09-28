# RDK Model Zoo Review —— AG09 sample 变更提案独立评审

**评审对象**：`fixture.json`（合成数据）所述变更提案，对照真实目标文件 `samples/vision/pointnet`。
**模式**：change-review（提案尚未落盘，diff 以 fixture 声明为准）；**平台身份**：sample 仅发布 s100（非 fixture 所称 x5，见 F-02）。
**规则来源**：目标 ref 真实契约 `docs/sample-standards/inference-contract.md`、`docs/sample-standards/readme-contract.md`、`docs/Model_Zoo_Repository_Guidelines.md`、发布 Manifest `docs/release/s/models.yaml`，以及技能自带 `references/*.md`。

---

## Findings

### F-01 · blocking · confirmed ｜ Technical Correctness & Regressions ｜ introduced
**提案把 "load JPEG → run model → NMS → save result.jpg" 全部塞进 forward，接口职责越界。**
- **Location**：提案（fixture.json:21）；现行对照 `samples/vision/pointnet/runtime/python/pointnet.py:54-56`（forward 仅返回 raw logits）、`main.py:1`（"file IO and optional plotting live outside inference stages"）、`main.py:64-68`（可视化由 main 调专门模块）。
- **Rule**：`docs/sample-standards/inference-contract.md:46` —— forward/runner **禁止**混入 NMS、任务解码、绘图、保存结果；pre_process 禁止 CLI/文件 IO（读取输入属 main）。required（Phase 0.5 Q2 基线，适用于 develop 统一 samples 全部 Python 运行时）。
- **Evidence**：必需测试 `tests/test_pointnet.py:56-59`（`test_forward_keeps_raw_scores`，forward 纯度）在该提案下必然失败；`runtime/python/README.md:100` stage-io 表明确 forward "no argmax, dequant or IO"。库集成契约（`runtime/python/README.md:65-92`）也会被破坏。
- **Impact**：破坏契约 §5 必需测试、破坏库集成方 API 语义、回归现行四阶段架构。且本任务为逐点部件分割，**NMS 在语义上不成立**（无检测框概念）。
- **Minimal fix**：文件读取留在 main/CLI；删除 forward 内 NMS/存图；如未来确有过滤需求，放入 post_process 或独立模块并给出需求来源。

### F-02 · blocking · confirmed ｜ Delivery Specification ｜ introduced
**README 提案 "由一次 X5 单输入运行推断所有 target 可用" 为不可绑定证据 + 虚假平台宣称。**
- **Location**：提案（fixture.json:22）；对照 `README.md:21-24`（支持矩阵：s100 supported-not-run；x5/s100p/s600 **not-supported**）、`model/README.md:14-15`（"No X5, S100P or S600 asset is published"）、`evaluator/README.md:82-84`、`model_binding.py:79`（"PointNet is published only for target s100"）。
- **Rule**：`readme-contract.md:64`（support-matrix 三态硬要求）、`:147-148`（§5.7 host/受限运行 ≠ 板端验证，不得混写）；MZ-DOC-02 "没有实现的不宣称支持"；技能 `references/evidence-contract.md:7`（单次单板运行不能覆盖其他 target/变体/语言）。
- **Evidence**：fixture 板记录为 x5 + single image（fixture.json:5-9）。该记录**无法绑定本 sample**：发布 Manifest `docs/release/s/models.yaml:475-487` 仅有 `s100/pointnet.hbm`（sha256: null）；`samples/_shared/platforms.py:90-98` 的 `require_execution_target` 会拒绝 x5 请求/无板卡身份执行；且 `resolve_selection` 在 x5 上直接抛 `BindingError`。单输入 smoke 即使真实也不支持 "all targets work" 的量级。
- **Impact**：x5/s100p/s600 用户被误导存在支持与制品；违反文档诚信硬规范。
- **Minimal fix**：维持三态支持矩阵并与 Manifest 资产一致；该 x5 记录标注为 "来源无法绑定、不予采纳"；s100 板端状态在获得绑定证据前保持 supported-not-run。

### F-03 · blocking · confirmed ｜ Technical Correctness（兼交付主体）｜ introduced
**提案所述 custom_model 与制品契约三方冲突，按现行代码根本无法运行。**
- **Location**：fixture.json:11-18（input `[1,3,1024]` float32；output `[1,1024,50]` int8；quantization: null）。
- **Evidence/Rules**：
  1. `model_binding.py:121` 要求输出 **(1,N,4)**，(1,1024,50) → `MetadataMismatchError`；
  2. `model_binding.py:123-127` + `pointnet.py:77-81`：整型输出必须有 SCALE 量化描述符，`quantization: null` → `ValueError`；
  3. `main.py:69` 标签硬编码为 4 个椅部（back/seat/leg/arm），50 通道输出无标签映射；
  4. 该模型不在 Manifest（MZ-ASSET-01：模型获取、路径、下载脚本与 Manifest 必须一致）。
- **Impact**：提案以无需求来源的自制模型替换交付主体，属技能明示不适用项（"把没有需求来源的样例改造成偏好的架构"）。
- **Minimal fix**：撤销模型替换；如确有新制品需求，先立需求来源、补 Manifest 资产身份与 SCALE 量化元数据，再谈绑定。

### F-04 · major · confirmed（依据为 fixture 声明的修订序）｜ Delivery Specification ｜ exposed
**转换证据过期：conversion "passed" 绑定于 `before_preprocess_change`，而当前代码为 `after_preprocess_and_postprocess_change`**（fixture.json:24-29）。技能 `references/evidence-contract.md:19`：旧工具链收据可作来源，但不能替代修改后的端到端验证；检查必须绑定目标提交/dirty 快照。**Impact**：当前修订无任何有效转换后验证。**Minimal fix**：在当前修订重跑宿主 fixture 测试与 s100 板端冒烟，生成绑定当前提交/快照的新收据。

### F-05 · major · confirmed（fixture 声明）｜ Technical Correctness ｜ pre-existing（与提案关系未知）
**最近一次运行失败：rc=2 且无新输出**（fixture.json:30-32）。`main.py:78-80` 中返回 2 即错误路径（ValueError/OSError/RuntimeError/ImportError）。失败原因未记录，无法诊断；当前修订的必需冒烟为 **failed**。→ 列入 Open Questions Q-1。

### F-06 · minor · confirmed（fixture 声明）｜ 证据纪律
**现存结果文件来自旧运行（mtime 2020-01-01）**（fixture.json:33-35）。按 `references/evidence-contract.md:13`，输出须核对本次生成时间/内容，陈旧产物不得作为当前修订证据。另经 Glob 核实 sample 下无已提交 outputs（`samples/vision/pointnet/outputs/**` 无匹配），该文件应为工作区未跟踪文件——无 git 工具无法盘点（见限制 1）。

---

## 三维度结论

- **Repository Standards**：提案违反 inference-contract §3（required）与 readme-contract §4.1/§5.7（required）。现行 pointnet 代码本身与契约一致（见通过项）。
- **Delivery Specification**：**No delivery specification available** —— fixture 无 issue/PR/用户规格等需求来源；仅能按目标 ref 仓库契约与 sample 自身声明（"Current board inference is not-run"，README.md:28）判定。README 走查（按 readme-contract 逐章）：sample 根英文版 8 个固定锚点（overview/support-matrix/prerequisites/quickstart/expected-results/directory/entry-points/license）齐全且 quickstart 含 cwd/显式模型准备/成功判据；中文版仅确认文件存在，**逐章内容配对核对未做（not-run）**，不给静态 pass。
- **Technical Correctness & Regressions**：接口职责越界（F-01）、模型绑定失败（F-03）、数值回归证据缺失（F-04/F-05；当前修订无任何绑定证据，不接受"应无影响"）。旧能力保留：提案会移除 forward 纯度与四阶段职责这一既有能力，属回归。

## Passed Checks（现行基线抽查，均为静态文件证据）

- forward 仅透传 runner 校验后输出：`pointnet.py:54-56`；forward 纯度测试存在：`tests/test_pointnet.py:56-59`。
- 整型输出强制 SCALE 反量化：`model_binding.py:126-127`、`pointnet.py:77-81`。
- 推理不下载：`main.py` 无下载调用；下载在 `model/download.py`（"inference never downloads"）。
- CLI 参数表抽查与 parser 一致（`--target` 默认 auto、`--output-dir` 默认 `outputs/pointnet`：`main.py:19,23` ↔ `runtime/python/README.md:41-50`）。
- Manifest 行与 model/README 制品表一致（`docs/release/s/models.yaml:475-487` ↔ `model/README.md:8-13`），`sha256: null` 如实披露。

## Open Questions

- **Q-1**：rc=2 失败的具体原因与发生修订（阻断：F-05 定级、当前修订是否可运行）。验证方式：在授权环境重跑 `python3 -m unittest discover -s samples/vision/pointnet/tests`（宿主）与 s100 冒烟命令（见 `conversion/README.md:60-63`）。
- **Q-2**：`existing_result` 是否为未跟踪工作区文件（阻断：untracked 盘点；需 git 工具或 repo 技能）。
- **Q-3**：base/head/merge-base 与 introduced/regression 的 git 实证（完全未定，见限制）。

## Verification Matrix（人类可读；JSON 契约工具缺失，按 evidence-contract 降级路径输出）

| 检查 | level | status | 说明 |
|---|---|---|---|
| 提案 forward vs inference-contract §3 | static | **failed** | F-01，confirmed |
| custom_model vs 绑定协议 | static | **failed** | F-03，confirmed |
| README 提案 vs support-matrix/实测声明 | static | **failed** | F-02，confirmed |
| 现行 README 逐章契约走查（中英内容配对） | static | **not-run** | 仅抽查英文锚点与参数一致性 |
| 宿主 fixture 测试 | host | **not-run** | 无执行工具；required |
| s100 板端冒烟 | board | **not-run** | 无板卡/SDK；sample 自身要求，required |
| x5 板端 | board | **not-applicable** | 无发布资产；fixture 的 x5 "passed" 不予采纳（无法绑定） |
| 转换收据对当前修订有效性 | host | **failed** | 过期（F-04，fixture 声明） |

## Overall Verdict

- **review_decision = needs-rework** —— 提案在职责架构（F-01）、平台宣称（F-02）、交付主体（F-03）三处主体性错误；若整体撤回 forward/readme/model 提案，则现状仅为 changes-required（补当前修订的宿主测试与 s100 冒烟证据，并查 Q-1）。
- **delivery_readiness = not-ready** —— 必需板测 not-run/failed、证据全部无法绑定当前修订；静态 pass 不升格为 ready（SKILL.md Output 约束）。
- 未运行项与限制：本评审全程只读静态分析；未执行任何 sample、量化、板测、安装或发布动作；作者/fixture 声明的运行结果均按"来源声明"记录，未采纳为 reviewer 亲测。

---

## 缺少的协作能力（本次任务要求的明确清单）

1. **`rdk-model-zoo-repo` 技能缺失**：SKILL.md 明确把"平台/目录/未跟踪文件盘点"路由给 repo 技能；本环境无它，且无 shell/git，工作区身份、HEAD、dirty、untracked 盘点完全无法产出（Q-2/Q-3 阻断）。
2. **`rdk-model-zoo-validate` 与 `schemas/verification.schema.json` 缺失**：evidence-contract 的 JSON 验证契约无法产出与结构校验，只能按其降级条款输出人类可读矩阵（上文已如此标注）。
3. **SKILL.md 第 8 条的 `validate` 接收方缺失**："确需运行的检查形成独立授权计划，交 validate" 无处可交；授权计划（宿主 unittest、s100 冒烟，cwd=仓库根）只能列为本回复中的未执行计划。
4. **develop 技能缺失**：整改只能给最小修正方向，无法交接实施；本环境 AG09 亦无 Edit/Write，报告仅以回复交付，未写任何文件。
5. **平台工具链 pack（oe-skills-x5 / S 线 run contract）缺失**：evidence-contract 提到的既有工具链收据无法引用核对，fixture 板记录无法与任何真实收据交叉验证。
6. **环境级限制**：无 shell/git（base/head、merge-base、staged/unstaged/untracked 全部 not-run，change_relation 依赖 fixture 声明）；无网络（外部 URL 未核验）；无板卡/SDK/OE（board/conversion 类检查保持 not-run，不写成 not-applicable，缺板卡项已按 evidence-contract 注明原因）。
7. **需求来源缺失（非技能但同属协作缺口）**：提案无任何规格来源，交付维度只能落到 `No delivery specification available` + 目标 ref 仓库契约，无法核对提案者承诺。

## 已执行的操作

仅使用 Read/Glob/Grep 读取：技能包 8 个文件（`SKILL.md`、`skill-card.md`、`NOTICE.md`、`evals/tasks.yaml`、`references/`×4、`assets/review-report.md`）、`fixture.json`、目标 sample 全套真实文件（runtime/python 6 个、model 2 个、conversion/evaluator README、tests、根双语 README）、共享工具 `single_array_runner.py`/`platforms.py`、规范三份、Manifest 与 `docs/release/` 索引。**未修改任何文件、未安装任何东西、未执行任何命令、未创建子代理、未读写记忆**。
