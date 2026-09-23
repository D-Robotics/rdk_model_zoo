# 外部执行计划快照（2026-09-23）

原路径：~/.claude/plans/golden-scribbling-micali.md。历史暂停指示已由同日 github-coordination 记录更新。

# 计划：基于 develop 的全分支 sample 化合并（消除 platforms/ 形态）

## 当前执行约束：外出期间静默暂缓板测（用户 2026-09-22 指示）

用户目前不在局域网板卡环境。自本说明起，在用户明确恢复板测前，板端任务静默跳过：不发起 SSH/网络可达性探测，不周期重试，不反复询问板卡或提醒同一阻塞。允许继续主机侧开发、检查及独立评审；不因板测暂缓反复停止这些可执行工作。

板测状态如实保留 `not-run（用户外出期间暂缓）`，不记 passed 或 not-applicable，不声明板端数值等价。暂缓不是永久豁免，也不自动将批次 Closed 改为 yes；独立评审分别记录主机范围结论及待补板测。恢复板测由用户明确通知，不自行根据网络探测启动。批次推进仍遵守用户授权；后续授权已允许本地主机开发与独立评审逐批推进；最新进度见文末2026-09-23记录，板测门禁仍用于最终交付。


> **执行约定：**使用 superpowers:executing-plans 逐项执行；计划内新增检查框只有取得对应证据后才能勾选。本次修订仅更新计划，不代表规范、检查器或 Skills 已实现。

**目标：**逐 sample 深度重构 X5/S 交付，使客户、开发者和 Agent 使用同一套规范、README 和真实入口。
**架构：**以 develop 为整合线，共享语义一致的实现，以制品/阶段契约隔离真实平台差异；完成全部迁移和验收前，rdk_x5、rdk_s 继续承担客户交付。
**依据：**用户已确认的 README 分层、三阶段职责、规范基线与验收门槛；既有 develop 架构文档按本计划中的范围修订使用。
**技术：**Python/C++、板端 hbm_runtime、现有 Manifest、Markdown 与现有 Skills Pack。

## 本轮补充的全局约束（2026-09-20）

- 不改成“平台代码归位即完成”；每个 sample 深度审计、重构和完整能力迁移。
- 不提前将客户新开发切到 develop。记录源基线 SHA；旧交付分支的新增 sample、已迁移 sample 修复和公共工具变更分别进入增量账本。源变化按语义适配，不覆盖已重构实现；交付前再次核对两侧最新 SHA。
- README、规范、检查器和 Skills 是不同层：仓库规范定义事实和要求，模板承载内容结构，检查器执行可验证规则，Skills 引导工作流程。禁止复制出互相冲突的规则源。
- 双语 README、职责边界、文档命令验证、同板迁移前后对照均为批次必需项；未完成不能以“代码已迁移”关闭该项。
- 不新增逐 sample 执行配置；沿用 README、脚本、Manifest 和现有 evidence JSON。

## 评审重点

1. README 标题齐全但缺输入、执行目录或结果解释：由 Q1 内容契约和 Q4 客户路径评审阻断。
2. CLI 默认值与文档不同、Python 示例含未定义变量：由 Q3 一致性检查与示例测试阻断。
3. forward 混入下载、NMS、文件输出，或 predict 另写算法：由 Q2 边界测试与 Q4 语义评审阻断。
4. 非方形/不同尺寸输入连续调用产生 context 串扰，多阶段任务为凑接口改变执行顺序：由 Q2/Q4 回归覆盖。
5. 检查器只通过正例、Skills 自检冒充独立审阅、host 冒充板测：由 Q3 负例、Q5 行为评测和逐批证据检查阻断。

## Context

仓库当前按硬件平台分分支维护（rdk_x5 主交付 37 samples、rdk_s 38 samples、rdk_x3 15 demos 归档）。`develop` 分支已确立 sample 为中心的统一架构并完成 3 个试点（ultralytics_yolo 含 yolo26、resnet **仅 resnet18 variant**、paddle_ocr），但其余内容仍以 `platforms/{x5,s,x3}/` 冻结快照形式存放（已落后源分支：rdk_s 落后 75 files，如 yoloe26_seg 整个 sample 缺失）。目标：把 rdk_x5 + rdk_s 的全部 sample 手工迁入 `samples/` 统一架构，最终删除 `platforms/`。

**已确认决策**：① 不做 git merge，逐 sample 从分支 tip 提取手工重构；② X3 保持历史归档不 sample 化；③ model_zoo_web 不合并；④ 每批迁移后用户上板（X5/S100）冒烟，修复后再进下一批。

**迁移模板**（以 `samples/vision/resnet/` 为参照）：`runtime/python/{main.py(--target), model_binding.py(平台差异), model_runner.py(懒加载 hbm_runtime), tensor_io.py, 任务模块}` + `runtime/cpp/`(保留单平台交付) + `model/{download.py, download.sh --target}` + `conversion/`(按 target mapper) + `evaluator/` + `test_data/` + `tests/` + 双语 README(_cn).md。

---

# 一、架构层设计

## 1.1 X5 与 S 推理接口的真实差异（已逐文件核实）

名义上同为 `hbm_runtime.HB_HBMRuntime`，但用法差异贯穿全链路：

| 维度 | X5（.bin, bayes-e） | S100/S100P/S600（.hbm, nash-e/m/p） |
|---|---|---|
| 输入张量 | **1 个 packed NV12**（4D `(1,3H/2,W,1)` 或 flat 1-D 均可） | **2 个 split NV12**：Y `(1,H,W,1)` + UV `(1,H/2,W/2,2)` |
| 模型声明 shape | `(1,3,H,W)` NCHW 逻辑形，H/W 取 `[2]/[3]` | NHWC 平面，H/W 取 `[1]/[2]` |
| 分类输出 | `prob` `(1,1000,1,1)` F32 | `(1,1000)` F32 |
| 量化输出 | YOLO 样例直接吃 raw（logit→sigmoid）；但 **fcos、unet 在 X5 侧也用 `output_quants` 反量化**（dequant 非 S 独有） | 15 个样例用 `output_quants` + `dequantize_outputs`（S 制品可带 int8 输出） |
| 输出访问时点 | forward 内 `run()[model]` | post 内延迟索引 |
| 默认参数 | NMS IoU 0.70、CLS resize=letterbox、插值 linear | NMS 0.45、CLS resize=stretch、插值 nearest |
| 模型获取 | sample 本地 `model/download.sh` | 系统 `/opt/hobot/model/<soc>/`；旧 run.sh 读 boardinfo 且 s100p 静默回退 s100（develop 已改为显式报错） |
| C++ | `hbDNNInfer` + `hbSysAllocCachedMem`（libdnn） | `hbDNNInferV2` + `hbUCPMallocCached`（+libhbucp；对齐 S600=64B / S100=32B）——**调用层不可统一** |

调度参数 `set_scheduling_params(priority, bpu_cores)` 是唯一完全一致的公共面，保持 pass-through。

**硬约束**（统一架构必须尊重）：制品不可互换（.bin/.hbm、每 march 一套）；输入协议属于具体制品/阶段契约；明确采用 NV12 的制品再区分 packed/split，角色按经验证的 metadata 绑定，歧义时拒绝猜测（SAM 浮点 RGB/embedding、DiffusionDrive 输入量化等不能套 NV12）；输出 rank/dtype 逐制品不同；预处理数值差异（插值/letterbox/NMS 默认值）必须 per-target 显式配置——静默改动会改变精度；板卡身份只认 boardinfo/socinfo/device-tree 且不静默回退；S100P 制品覆盖不全须在清单中显式表达；`hbm_runtime` 仅存在于板端镜像（host 侧必须懒加载）。

## 1.2 厂商先例与模式选择

调研了 Rockchip、Hailo、Qualcomm AI Hub、Vitis AI、STM32、Ultralytics、Optimum、Axera 八家多硬件 model zoo，归纳四个模式：

| 模式 | 代表 | 对 RDK 的适配 |
|---|---|---|
| **P1 model-first + 板卡 flag + sample 内 per-runtime 变体** | Rockchip rknn_model_zoo（`build-linux.sh -t rk3588`；同一 sample 内 `rknpu1/` vs `rknpu2/` 两代 runtime 源码并存） | ✅ **骨架选择**。与"sample 为中心"目标一致，且 rknpu1/rknpu2 并存先例 = 我们的 X5 `.bin`/S `.hbm` 分裂 |
| **P2 target-first 目录树** | Axera ax-samples（`examples/{ax620e,ax650,...}`） | ❌ 等于把现在的分支之痛变成目录之痛；仅适用于芯片代际真正不同的情况 |
| **P3 包 + YAML 注册表（机器可读支持矩阵）** | Hailo Model Zoo（每模型 YAML 内含 `supported_hw_arch`、精度、license） | ✅ **借其思想**：不建 pip 包，用现有 models.yaml 行（已含 sample_path/platform）+ platforms.json 作为机器可读矩阵，主机测试校验 sample↔manifest 覆盖 |
| **P4 清单驱动制品注册 + 文档卡** | STM32 manifest.json、Vitis downloader、Qualcomm AI Hub 卡片 | 已有等价物（models.yaml/benchmarks.yaml + catalog） |

**版本线佐证**：连最统一的 Hailo 也沿工具链代际线保持 v2.x/v5.x 分裂——印证 X3（旧工具链代际）单独归档、X5+S（同代 SDK，仅制品格式不同）合并的决策。

**Ultralytics 类比**：其"加载制品的扩展名决定后端"→ 我们的 `--target`/`--asset-id` 决定 binding 契约，方向一致。

## 1.3 目标分层架构（在 develop 既有代码上收敛，不另起炉灶）

```
main.py  CLI：--target auto|x5|s100|s100p|s600、--asset-id、--dry-run（无 SDK 可跑）
  │
  ├─ 平台身份  samples/_shared/platforms.py
  │     boardinfo → socinfo → device-tree；S100P 用 board_type 细分；未知即报错，无静默回退
  │
  ├─ 平台 Profile  ★H5：把 ultralytics_yolo/yolo_platform.py 的字段形状升为通用契约
  │     per-target 常量：march/文件名后缀、model_format(.bin/.hbm)、平台输入能力与默认值（实际协议由 artifact/stage binding 指定）、
  │     packed_layout、默认参数(nms_iou, cls_resize, interpolation)、URL 布局、supports_cpp
  │     落地纪律：spec §5.1——B1 在 sample 内定义，出现第二个真实消费者才升入 _shared/
  │
  ├─ 资产绑定  samples/_shared/assets.py（manifest 驱动 group:sample:file + sha256）
  │
  ├─ 模型契约  <sample>/runtime/python/model_binding.py
  │     per-sample per-target 张量契约：按 shape 识别 Y/UV/输出角色、期望 shape/dtype、
  │     输出语义声明；★H4：输出 rank 归一化规则（squeeze 批维）取代相等断言
  │
  ├─ 张量 IO  tensor_io.py：resize(stretch|letterbox) + NV12 组装 + 严格校验
  │     ★H2：packed NV12 物理布局定死一种（flat 1-D，与 yolo_input 一致），profile 显式声明
  │
  └─ Runner  model_runner.py：懒加载 HB_HBMRuntime、runtime_factory 注入缝、输出扁平化
        ★H1：binding 声明 output_transform，runner 暴露 metadata；post_process 执行 raw_f32 | dequant(output_quants) 及明确激活
        （S 制品 int8 输出现被直接拒绝，不补则 B5/B7/B8/B9 的 S 侧迁移卡住）
  │
  └─ 任务模块（pre/decode/nms 共享算法 + Profile 提供平台数值）
```

**C++ 策略**：延续现状约定——cpp 保持平台范围显式声明（README 写清 X5-only / S-only），sample 内可并存 per-target main（CMake 用 `SOC_<NAME>` 宏，符合仓库既有跨平台规范）；只共享结构，不强行统一 infer/内存调用。

**共享纪律**：sample 本地优先；两个真实消费者出现才升 `samples/_shared/`（resnet 试点已验证的 functions 不动）。

## 1.4 架构硬化清单（Phase 1.5，批量迁移前置）

| # | 硬化项 | 内容 | 验证 |
|---|---|---|---|
| H1 | 输出变换链（dequant + 激活语义） | binding 声明制品级 `output_transform: raw_f32 | dequant(output_quants)` + 激活语义（X5 YOLO raw logit→sigmoid vs S dequant 后已激活）；移植源：`rdk_s:utils/py_utils/postprocess.py:136 dequantize_outputs()`；resnet 保留 reject-int8 作为声明式选项。影响面：S 侧 B5 dinov2、B7 yolov5/bytetrack、B8 unetmobilenet/lanenet/diffusiondrive、B9 yolo 系；**X5 侧 B7 fcos、B8 unet 同样依赖 output_quants** | 用 resnet 主机测试 + S 制品元数据 fixture |
| H2 | packed NV12 布局标准化 | 定死 flat 1-D 为 canonical；**元数据声明 shape 校验不变**（X5 仍是 `(1,3,H,W)` 逻辑形），仅统一喂入张量布局；tensor_io 提供 `as_packed/as_split` 显式转换；resnet 测试同步更新 | resnet 全量主机测试 + X5 板冒烟确认 flat 与 4D 等价 |
| H3 | 元数据泛化 | `RuntimeMetadata` 支持多模型（paddle_ocr 两模型）与多输出张量；从 `model_names[0]` 单模型假设解脱 | paddle_ocr 既有主机测试回归 |
| H4 | 输出 rank 归一化 | binding 契约改为"期望 rank + squeeze 规则"，替代 `(1,1000,1,1)` 与 `(1,1000)` 双硬编码 | resnet 双 target 契约测试 |
| H5 | 平台 Profile 契约 | 以 yolo_platform.py 为参照定义 Profile 字段集（文档 + B1 落地）；明确升级到 `_shared/` 的条件 | B1 迁移时实战校验 |
| H6 | sample↔manifest 覆盖检查 | 主机测试校验：每个迁移后 sample 在 models.yaml 有对应行、sample_path 一致（Hailo 式矩阵落地，复用现有清单） | 加入 samples/_shared/tests |

H1–H4 改动收敛在 `samples/_shared/` + 3 个已迁移 sample 的测试内，一个 commit 序列完成；H5/H6 随 B1 落地。

## Phase 0 — 准备与验收

1. **安装 Skills**：从固定的 `rdk_x5` 源 SHA 提取 7 个 `skills/<name>/` 完整目录到 `~/.claude/skills/`，先确认无同名旧副本，保留用户修改；不能只拷 SKILL.md。记录源完整 SHA、安装位置及逐文件比对结果。
2. **安装验证分两项**：
   - 文件与工具可用性：从已安装位置执行 `python3 ~/.claude/skills/rdk-model-zoo-repo/scripts/inspect_repo.py --repo <REPO_ROOT>`，记录命令、退出码和结果。切换 develop 后尚无根 `skills/`，不能使用仓库相对路径冒充安装验证。
   - 会话发现与触发：重载实际执行迁移的 Claude 会话，确认七个 Skill 出现在可发现列表，并以只读 repo/review 请求确认路由到正确 Skill；记录会话、技能名、加载来源和实际响应。文件存在或脚本成功不等于会话验证成功；没有记录标为 `not-run`，不得声称已验证。
3. **切换工作树到 develop**：记录完整 HEAD 和工作区状态，保留未跟踪 CLAUDE.md。该文件的必要纠偏见下文，不能等最终收尾才消除与 develop 的冲突。
4. **建批次记录约定**：沿用 `docs/releases/unified-migration/`，使用 `2026-09-XX-bN-<name>-review.md` 和 `evidence/*.json`；在 `x5-s-migration-map.md` 追加独立的本轮 B1–B11 进度区。历史 P0 表及其状态保持不变。
5. **修正启动记录**：同步修改 `docs/releases/unified-migration/2026-09-20-batch-migration-kickoff.md` 中对 F 的错误解释，链接本轮进度区并给出一致的完成条件，不能只修计划而保留仓库内冲突说明。

### 台账语义与本轮进度

历史 `S/F/H` 是 P0 清点口径：S=源码/清单静态核对，F=旧新函数映射未核定，H=板端验证未执行；不是顺序状态机，禁止把 F 改解释为 mapping verified，也不翻转历史 P0 列。

本轮进度区按以下固定列记录，供人工审阅和 Q3 读取。一个 sample 若目标、变体或语言的完成状态不同，必须拆行，不能用一块板的通过覆盖整行其他目标。

| Batch | Sample / source SHA | Target / variant / language | Mapping | Refactor | Docs | Host | Board | Review | Closed | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

- Mapping、Refactor、Docs：`pending / in-progress / done / not-applicable`。
- Host、Board、Review：`not-run / passed / failed / not-applicable`。
- Closed：`yes / no`；默认 no，迁移代码存在不等于 closed。
- 每项记录 required 与证据链接，可在 Evidence 链接的批次报告按维度展开；not-applicable 必须引用适用范围和理由，不能替代 required 检查。
- Closed=yes 只在所有 required 维度完成/通过且没有未解决阻断项时成立；required 的 not-run/failed/pending/in-progress 一律不能关闭。可选未运行项仍披露，不扩大支持或实测声明。
- Evidence 绑定实际代码 SHA、源 SHA、目标、制品与输入身份、命令/cwd、结果和评审来源；阶段没有执行时不生成虚构成功记录。
- Q3 检查器的纳入范围按 Refactor=in-progress/done 的新进度行确定，而不是只扫描 Closed=yes；因此未完成文档的已迁移代码同样受检查。历史 P0 状态不得用于推断新代码已验收。

### Phase 0 评审后补项与进入 Phase 0.5 的条件

- [ ] 修正启动记录与计划的状态冲突，建立上述本轮进度区；暂未迁移的行如实 pending/not-run。
- [ ] 补充 Claude 会话发现/触发证据；安装文件一致和脚本可运行分别记录，不冒充完整验证。
- [ ] 开始 Phase 0.5 前对 CLAUDE.md 做最小纠偏：说明当前 develop 是整合线；去除硬件只按分支选、无条件双语言、默认回退 s100 等冲突要求；将不存在的 docs/catalog、根 skills 校验路径改为真实入口或明确尚未迁入；优先链接实际 AGENTS、当前规范和本计划。保留与当前事实一致的用户内容，不整文件覆盖；最终结构稳定后的全面整理仍在收尾执行。
- [ ] 记录 inspect_repo.py 的已知边界：当前安装版本对 develop 返回 branch_role=unknown、manifest_candidates=[]，不能据此声称没有 Manifest；人工核查 `platforms/{x5,s}/docs/release/` 并记录。支持统一布局的脚本改造纳入 Q5，不为 Phase 0 提前宣称完成。

Phase 0 的通过范围是准备工作，不代表规范基线、模型迁移或板测通过。本轮实际发现的状态冲突必须修正后才能关闭 Phase 0；会话验证缺失时保持对应项 not-run。

## Phase 0.5 — 规范基线（必须在 B1 前完成）

本阶段与 Phase 1/1.5 协调：Q1/Q2 先定规则；Q3/Q5 在 Phase 1 引入相应设施和 Skills 后落地；Q4 在 Phase 1.5 修改后重新验收。Phase 1.5 可以与基线一起迭代，但 Q1–Q5 未通过不能进入 B1。下列均为未来交付路径，当前计划修订不会自动创建它们。

### Q1 — README 内容契约与模板

**文件：**修订 `docs/Model_Zoo_Repository_Guidelines.md`；新增 `docs/sample-standards/readme-contract.md`、`docs/sample-standards/templates/{sample,model,runtime-python,runtime-cpp,conversion,evaluator}.{en,zh}.md`；根规范链接专门契约，避免重复维护要求。

- [ ] 为下表每种 README 编写双语模板，列明每章必须回答的问题、命令环境、证据和适用条件；模板不是只有标题的骨架。
- [ ] 定义固定章节 ID，英文/中文标题可不同，CLI、支持范围、结果和限制必须对应。各级内容各有归属，导航链接不能代替必要操作步骤。
- [ ] 用现有 ResNet 和 OCR 检验契约能覆盖单模型与多阶段任务；记录与旧规范冲突并同步修订，例如旧规范的自动下载示例不能覆盖现在的显式模型准备约定。

| README | 内容验收要求 |
|---|---|
| sample 根 | 算法与来源、支持/实测矩阵（target×variant×语言）、环境前提、一条完整快速体验路径、真实预期结果、目录职责、模型/运行/转换/评估入口、必要许可说明 |
| model | 制品与 target/stage 的对应、下载或手动准备、词典等伴随文件、本地路径、格式和已知校验值；未知 hash 不伪造 |
| runtime/python | 环境、cwd、默认/自定义命令、完整参数与实际默认值、结果字段/路径、完整可运行的 Python 集成示例、三阶段 I/O、必要故障排查 |
| runtime/cpp | 适用板卡、依赖、完整构建/运行命令、参数、接口和资源生命周期、结果解释；未提供 C++ 时不得声称双语言支持 |
| conversion | 源模型/权重版本、工具链与目标、导出、校准、编译、产物、转换后验证；缺配方时列出缺失项及当前可复现范围 |
| evaluator | 数据集版本/准备、环境/cwd、评估命令、指标定义和测试条件、输出位置、参考结果与来源；没有评估实现时明确边界 |

统一内容纪律：命令给出 cwd、前置文件来源、参数、输出及成功判断；Python 集成示例中的输入/配置必须定义。用户按步骤准备真实文件属于前提，不可把未定义变量当示例。禁止“同其他模型/参考原分支”替代关键步骤；禁止以通用命令伪装已验证转换流程；迁移历史、canonical/wrapper 审计说明放迁移记录，客户 README 只保留必要兼容说明。无能力时如实写限制，不用空目录或占位文档充数。公开 API 的 shape、dtype、布局、值域、坐标约定及异常在 docstring 中精确说明，README 给可理解的摘要和使用例。

### Q2 — 推理职责与接口契约

**文件：**新增 `docs/sample-standards/inference-contract.md`；根规范引用；调整 `samples/vision/resnet/runtime/python/` 和 `samples/vision/paddle_ocr/runtime/python/` 及其 tests。

- [ ] 单模型公开业务接口统一 `pre_process`、`forward`、`post_process`、`predict`；允许初始化、必要调度/资源生命周期接口和委托 predict 的 `__call__`。不以整个文件只能四个函数作为规则。
- [ ] 三阶段数据流采用以下概念契约：每个任务在 docstring/类型定义中具体化 Input/Tensors/Context/RawOutputs/Result，不要求全仓新增通用基类。旧接口差异只在明确兼容适配中保留。

```python
prepared = model.pre_process(inputs)  # tensors + 本次调用的 context
outputs = model.forward(prepared.tensors)
result = model.post_process(outputs, prepared.context)
# model.predict(inputs) 必须串联以上步骤，返回同一 Result 契约
```

| 模块/阶段 | 职责 | 禁止混入 |
|---|---|---|
| pre_process | 业务输入验证、数值/布局变换、构造本次调用上下文 | CLI、下载、数据集遍历 |
| forward / runner | 匹配后端加载/调用、结构与 metadata 校验、明确的输出容器适配 | NMS、任务解码、反量化/激活业务变换、绘图、保存结果、评估 |
| post_process | 按制品契约反量化/激活、解码、坐标还原、业务结果 | 模型下载、再次推理、报告/文件输出 |
| predict | 串联阶段、返回业务结果 | 第二套前后处理实现 |
| main | CLI、输入读取、调用、结果展示/保存 | 另写模型算法 |
| binding | target/stage/asset 的张量与语义契约 | CLI、文件下载、任务编排 |

NMS、CTC、几何操作等相关辅助函数允许存在；复杂或复用逻辑拆成职责明确的本地模块，禁止把杂项塞进泛化 utils，也不为满足函数数目限制把大段代码塞进一个方法。可视化可由 main 调用专门模块。runner 可以暴露量化 metadata，但 H1 数值变换由 post_process 或其明确委托模块执行。

- [ ] context 显式保存尺寸、scale/padding 等本次输入信息，不放入会被下一次调用覆盖的实例字段；有状态任务如跟踪、语音流明确 session/state/reset 语义，不虚称线程安全。
- [ ] OCR/SAM 等采用 stage 的三阶段接口及 pipeline.predict 编排；检测→裁剪→识别顺序必须可读，禁止为凑三函数把下一阶段推理藏入 post_process。LLM/流式任务可声明生成/stream/reset 接口及理由，不能机械套单图模型。
- [ ] 在对应 sample tests 中验证：predict 与显式三步结果一致、forward 不执行解码/文件操作、两种尺寸输入交错处理时 context 不串扰、原始输出不被未声明变换；OCR 验证零检测、多裁剪及阶段错误归属。异常输入按 sample 协议验证，不宣称 SDK 并发可用。

### Q3 — 自动化检查（含反例）

**文件：**新增 `tools/sample_contract/check.py`、`tools/sample_contract/tests/test_check.py`、`tools/sample_contract/tests/fixtures/` 与 `.github/workflows/sample-contract.yml`。检查器只读取源码/文档和指定 fixture，不下载、加载 SDK 或执行任意 README 代码块。

- [ ] 先构造失败 fixture：缺必要章节、失效本地链接、CLI 默认值漂移、forward 调用下载/保存、双语参数不一致；每种对应明确规则 ID 和 path:line，正例与反例都断言结果。
- [ ] 实现命令 `python3 tools/sample_contract/check.py --sample samples/vision/resnet`：检查适用 README、章节/链接/本地路径、公开接口、AST 可判定的越界调用；动态行为无法证明时交语义评审，不报告自动通过。
- [ ] CLI 参数与默认值从实际 parser 获取；只对可信目标代码在主机测试进程中调用无副作用的 build_parser，不运行 sample。静态模式不执行代码；完整 Python 示例由对应 sample tests 在明确 fixture 中验证。
- [ ] README 板端命令与 evidence JSON 记录 cwd、完整命令、code SHA、模型 hash、target/runtime、输入 hash 和实际输出。主机测试只证明主机边界；所有文档代码块按说明/主机/板端/转换分类审阅，不能自动全执行。
- [ ] CI 对已迁移 sample 执行检查；首次建立迁移范围解析时读取本轮进度区的 Refactor 列（in-progress/done 均纳入），不读取历史 P0 的 S/F/H，不新建逐 sample 执行配置。公共规则修改重跑全部已迁移 sample。检查器测试命令：`python3 -m unittest discover -s tools/sample_contract/tests -v`。

不得通过减少文档内容、删除原有能力或放宽规则来消除检查错误。合法任务例外在接口契约及批次报告明确，评审确认后加入带理由的测试用例；禁止宽泛目录豁免。

### Q4 — 参照样例与双视角验收

- [ ] ResNet 作为简单分类参照，OCR 作为多阶段参照；按 Q1/Q2 修正代码、各级双语 README 和集成示例。现有试点并非天然合规。
- [ ] 同板、同制品、同输入和参数记录迁移前后结果；预处理数值、raw output、任务结果分别确定实际可用容差。X5 与 S 相互比较不能替代同板前后比较。
- [ ] 对每个参照 sample 做客户阅读路径评审（按 README 准备到结果）和 Agent 开发路径评审（定位接口、参数、制品、转换/评估入口）；评审者读取实际文件与命令证据，不能仅看作者摘要。
- [ ] 已有功能未迁移、必需文档不可操作、职责违规或必需板测缺证据，均不能标为 ready。合法未支持能力单列；不能拿 not-run 关闭必需项。

### Q5 — Skills 评估与补强（不新增第八个 Skill）

现有 `rdk-model-zoo-develop`、`rdk-model-zoo-review`、`rdk-model-zoo-validate` 已覆盖实现、独立审阅、验收。缺口是细化约束与检查证据，不是入口数量。仓库规范为唯一权威；Skills 读取目标 ref 的规范，不能把 X5 维护源规则无条件施加到旧版/X3。

- [ ] 补强 `skills/rdk-model-zoo-develop/SKILL.md` 与 `assets/development-checklist.md`：改动前列 README 内容契约和文件职责，改动后逐条关联文件/检查证据，不允许“文档已更新”一行代替。
- [ ] 补强 `skills/rdk-model-zoo-review/SKILL.md` 与 `assets/review-report.md`：分别审 README 可操作性、接口职责、旧能力保留和数值回归；缺章节内容、越界业务逻辑等必需整改不可静态 pass。作者自检不能冒充独立 review。
- [ ] 补强 `skills/rdk-model-zoo-validate/SKILL.md`：验证 README 的真实命令与 API 例子，区分 host/板端/转换，绑定代码和制品证据。执行范围沿用现有授权，不把读取技能当成执行板测的授权。
- [ ] 在上述三个技能的 `evals/tasks.yaml` 增加行为场景：标题齐全但不能运行、forward 混入下载/NMS、CLI 与文档漂移、OCR 合法多阶段、未验证板卡冒充通过；分别断言应阻断/应接受的行为，检查含义而非关键词。
- [ ] 共享内容只编辑 `skills/_shared/`，按 pack.json 的引用登记生成；不人工修改 references 副本。运行同步检查、Pack 校验、现有单元测试和新增行为评测；静态校验不等于行为评测通过。
- [ ] 扩展 `skills/rdk-model-zoo-repo/scripts/inspect_repo.py` 对统一布局的事实发现：迁移期 platforms 下清单与搬迁后 docs/release/{x5,s} 均有 fixture；分支角色依据仓库明确事实报告，不猜硬件。测试未知布局不误报确定结论，并保留历史 ref 行为；改造后重新验证已安装副本。
- [ ] 源码技能修订与已安装副本分开记录版本/摘要；确认无用户改动后更新执行端所需副本并重新验证发现/内容一致。安装、发布、Hub 切换不是本计划修订已完成的动作。

**基线出口：**Q1–Q5 全部完成、两类参照通过、检查器能拒绝反例、技能门禁有行为证据，才能进入 B1。第三个试点 ultralytics_yolo 不在本轮参照内，其 Q1–Q5 合规在 B9 收编时按同一门槛验收（收编前仅维持现状，不提前重写）。

## Phase 1 — 仓库级设施就位（ develop 上 1~2 个 commit，全部从分支 tip 提取）

| # | 内容 | 来源 → 去向 | 要点 |
|---|---|---|---|
| A1 | `utils/` | rdk_x5:utils/ + rdk_s:utils/（含 S 独有 c_utils）→ `utils/` | 先逐文件 diff 两侧 py_utils 再合并；这是过渡兼容层（ADR-0002），**不**并入 `samples/_shared/`（spec §5.1：两个真实消费者才准入共享） |
| A2 | `datasets/` | 两侧 datasets/ → `datasets/` | 同名目录先 hash 比对，禁止按名覆盖 |
| A3 | `skills/` | rdk_x5:skills/ → `skills/` | 直接复制；独立版本线（ADR-0006） |
| A4 | **Manifests + assets.py 改造** | develop `platforms/{x5,s}/docs/release/*.yaml` → `docs/release/{x5,s}/`；schema 取 rdk_x5:docs/manifests/schemas/ | 先建立旧 ID/资产/源路径/新路径映射：现有清单仍含 resnet18/50/152、paddleocr、ultralytics_yolo26 旧路径，且 S tip 清单缺 yoloe26_seg 等已有内容；逐项核对并补漏，保留旧 ID、资产事实与历史 Benchmark 来源。**同一 commit** 内改 `samples/_shared/assets.py` 的 `_models()` 路径与 `Asset.source_path`（现为 `platforms/{group}/docs/release/models.yaml`，assets.py:35/43）→ `docs/release/{group}/models.yaml`，并更新 `samples/_shared/tests/test_assets.py`、`tools/catalog-publisher/sources.json` |
| A5 | Workflows | 保留 develop 的 model-catalog-data.yml，移植必要数据校验，新增 sample-contract.yml | 不复制 Pages workflow（依赖 develop 不存在的 docs/catalog，且违背 ADR-0001）；更新清单路径并运行 catalog-publisher 完整检查 |
| A6 | 根/文档设施 | rdk_x5 的 CHANGELOG.md、VERSION、Model_Zoo_Repository_Guidelines.md、tros/、source_reference/；rdk_s 的 tros/、source_reference/、API guides | 双侧同名目录合并并注明来源 |
| A7 | 台账补遗 | rdk_s tip 新增内容 | 把快照后新增（yoloe26_seg、minicpm5 evaluator 等）补入迁移台账行，保证"每能力有去向" |

Phase 1 不做：platforms/ 删除、registry 合并、根 README 重写、任何 X3 内容。

## Phase 1.5 — 架构硬化（见上文 1.4 清单 H1–H6）

批量迁移的前置门槛：H1 反量化、H2 packed NV12 标准化、H3 元数据泛化、H4 输出 rank 归一化在一个 commit 序列内完成并跑通 3 个已迁移 sample 的全部主机测试；H5 Profile 契约与 H6 覆盖检查随 B1 首批落地。不做超前的框架化——硬化只补已证实存在的缺口。

## B1 独立评审与执行暂停点（2026-09-21）

审阅 HEAD：`c218e8622a2cb2025b6807ce2d36ef5c3df3d500`。独立 reviewer：Codex。
报告：`docs/releases/unified-migration/2026-09-21-b1-independent-review.md`；主机证据：同目录 `evidence/2026-09-21-b1-independent-review.json`。
结论：**changes-required / not-ready；B1 未关闭，不进入 B2。** 代码与大部分板测已完成不等于批次交付完成；下面不是已修复列表。

- [x] B1-R1（716bdca 独立确认全部关闭；历史：独立复审 dd60911：文件迁入通过，R1a/R1b README 待修正；整改 `e43718b`，resnet 52 tests OK；ResNet50 维持指针式不虚构）：迁入 ResNet152 校准脚本、YAML、x86 参考实现，并补齐 ResNet50/152 变体转换文档和逐能力映射；保留无法证明的源配方限制。
- [x] B1-R2（整改 `c1bb46b`，24 tests OK；板端复测 `dd60911`：s100 正例复现基线、s100p 4/4 拒绝）：MobileNetV2 C++ launcher 识别 S100+S100P board_type 别名，补 shell 身份正反例，禁止静默使用 S100 制品。
- [x] B1-R3（整改 `7cfc186`；X5 RGB 校准列为缺失前提，未虚构配方）：V3/V4 校准与编译 README 按 target/variant 明确 ONNX、RGB/BGR、校准目录及数值约定；脚本默认 224 与 medium 256、X5 RGB 和 deploy 图缺口必须具体说明，不能靠重命名假装流程连通。
- [x] B1-R4（整改 `7cfc186`，v4 20 tests OK，medium 256/128 为回归锚点）：修正 V4 中英文转换验收的 medium S 张量形状（Y 256×256、UV 128×128×2），增加文档关键 shape 与 binding 一致性验证。
- [x] B1-R5（整改 `97dc639`+`25a8187`；台账拆 20 行，自评与独立评审分离）：同步根/evaluator/C++ 双语 README 到实际板测证据，拆分台账不同目标/变体/语言状态；作者自评不是独立 Review=passed。
- [x] B1-R6（整改 `ea68ea6`；CI 同命令 rc=0、无基线 rc=1、修一条锚定问题 rc=1 三态验证）：已批准延至 B9 的 84 条 YOLO 文档欠账在 checker/CI 显式限界；新增违规仍失败，不可 continue-on-error 或整体排除 sample，B9 必须移除延期。
- [x] X5 8GB（`5ec7e1d`：192.168.3.208 复测 5/5，evidence 已回填）：由用户安排原执行者补测；独立 reviewer 本次不连接板卡。同步完整命令、代码/制品/输入身份与结果。
- [x] 已于 dd60911 完成整改独立复审：331 tests 通过，CI checker 通过；R2–R6 closed，R1 partial/open。见 `docs/releases/unified-migration/2026-09-21-b1-independent-rereview.md`。
- [x] R1a/R1b 双语文档已修复（B1 收尾提交：校准命令改 conversion cwd 相对路径并补记三步衔接；删除 scale"一致"声明，明确 mean 同/scale 异为保留源配方差异、未经 OE 重建确认；转换文件与 SHA 不变；resnet 52 OK、checker 0 violations）：见整改记录"复审遗留整改"节。
- [x] B1 独立复核 R1a/R1b 修复（716bdca 已通过，Closed=yes）；原待办：（仅文档，无需板测）；通过后才 Closed=yes。**修复记录为待独立确认，不以作者自检代替。**
- [x] 用户决定（2026-09-21，如实记录）：S600 MobileNetV2 C++ 不补测、保持 not-run、不作为 B2 阻断项（本就在 B1 既定门槛外）；R1a/R1b 修正完成后授权直接开启 B2，无需等待 B1 下一次独立复审。B1 各行 Closed=no / Review=changes-required 记录状态不变；B1 独立复核在后续完成。

现有板测证据边界：X5 4GB Python 5/5；S100、S600 Python 各 7/7；S100 MobileNetV2 C++ 构建运行通过；S600 C++ 未测且原批次不要求；S100P 仅两项拒绝负例，不能称模型正向推理已通过。独立 reviewer 本次执行 312 项主机测试全过、五个 B1 sample checker 无违规，但全量 migration checker exit 1 / 84 violations。验收记录不把这些结果相互替代。

后续批次新增检查重点：全能力文件映射覆盖 conversion/evaluator；README 与每个实际 variant 契约一致；shell 与 Python 身份规则一致；板后修复必须同步客户文档；独立评审每批执行，收尾抽查是附加检查。无需为此新增 Skill。

## Phase 2 — 迁移批次表（B1–B11，每批 3~8 个 sample）

每批板端冒烟通过后才进下一批（板测是瓶颈，估计每批 1~2 天，总计 2~4 周）。

| 批 | 内容 → 目标路径 | 来源 | 注意点 |
|---|---|---|---|
| B1 | mobilenetv1/v2/v3/v4 → `samples/vision/mobilenetv{1..4}`；**S 的 resnet50/resnet152 → `samples/vision/resnet` 新增 variant**（现 `SUPPORTED_VARIANTS=("resnet18",)`，model_binding.py:20，需扩展） | x5+s | 最简单重合；mobilenetv2 的 runtime/cpp 仅 S 侧（X5 侧 python-only）；**本批落地 H5 Profile 契约与 H6 覆盖检查** |
| B2 | efficientnet(x5+s)、efficientformer(v2)、efficientvit → 同名 | x5+s（efficientvit 仅 x5，已核实） | |
| B3 | convnext、edgenext、fasternet、fastvit | x5 | 纯移植，无合并审计 |
| B4 | repghost、repvgg、repvit、mobileone、resnext、vargconvnet、googlenet、hgnetv2 | x5 | 部分 conversion/evaluator 仅 README——如实记录，不造假交付 |
| B5 | clip(x5)、siglip/dinov2/vit/3dresnet(s) → 同名各自独立 | x5+s | clip 与 siglip **不合并**（spec §4.2）；3dresnet 是视频动作识别，独立输入路径；dinov2 依赖 H1 反量化 |
| B6 | efficient_sam、mobile_sam(x5+s) → 同名 | x5+s | 转换最重（Conv 8~12）；先审计 prompt/decoder 协议再决定共享代码 |
| B7 | yolov5(x5+s)、fcos、yoloworld、lprnet、modnet(x5)、bytetrack(s) | x5+s | **依赖 H1 反量化**（S 侧 YOLO 系制品带 int8 输出）；bytetrack 依赖 yolov5 检测器，批内排序并记录绑定；modnet 资产为 manual |
| B8 | unet(x5) 与 unetmobilenet(s) **各自独立**、pp_liteseg(x5)、yolo26_depth(x5+s)、depth_anything_v2、lanenet、pointnet、diffusiondrive(s) | x5+s | yolo26_depth 是最大转换移植（S Conv 29）；不并入 ultralytics 家族；unetmobilenet/lanenet/diffusiondrive 依赖 H1 反量化 |
| B9 | **YOLO 收编**：s 的 yolo11/yolo11_pose/yolo11_seg/yolov13 → `ultralytics_yolo` 新 family/variant 模块；x5 yoloe + s yoloe11_seg/yoloe26_seg → `samples/vision/yoloe` | x5+s | 仅在输出协议证明一致时合并，否则 sample 内独立模块（旧 Manifest ID 保持可解析）；**不得整目录覆盖已迁移的 ultralytics_yolo**，只 lift 模块；yoloe26_seg 取自 s tip（快照里没有） |
| B10 | himloco(x5) → `samples/robotics/himloco`；asr/kws/paraformer(s) → `samples/speech/` | x5+s | 首次音频 I/O，tensor_io 扩展留在 sample 内；paraformer 资产按组件记录 |
| B11 | gemma4-e2b、minicpm5-2b → `samples/llm/`；vla act/pi0 → `samples/vla/` | s | LLM 入口在 runtime/cpp；minicpm5 取 tip 的 legacy evaluator；vla 更新 `.gitmodules` 路径至 `samples/vla/` 并保持 pinned SHA（326ea043/a32de276），不 vendor 内容 |

X3 不在批次内（收尾时归档）。

## 每批固定流程

1. **预检**：固定源 SHA；`git ls-tree -r <branch> -- samples/...` 清点；先写旧→新函数映射表（x5、s 分开列）进批次评审文件再动手。
2. **提取**：`git restore --source=<branch> -- <path>`；已迁移的 3 个 sample 禁止整目录恢复。
3. **按模板重构**：统一 runtime/python 结构；`utils.py_utils` 引用优先 sample 本地化（第二个消费者出现才进 `_shared/`）；模型 URL 改为资产引用 `x5:<id>:<file>` / `s:<id>:<file>`；cpp 原样保留。
4. **主机与文档验证**：运行 sample 实际 unittest/pytest 及 `samples/_shared/tests`、Q3 检查器；验证无 SDK 入口、完整 API 示例、参数默认值和双语一致性；证据 JSON 存 `evidence/`。
5. **Manifest**：核对既有行 `sample_path`/`download_scripts` 匹配；新增产物及源分支已有但漏登的资产均需补录并注明来源；历史发布事实与迁移后的可执行路径分别说明。
6. **记录**：更新本轮进度区的 Mapping/Refactor/Docs/Host/Board/Review 及证据，按 Phase 0 的 required 规则判断 Closed；历史 P0 表不变。同步批次评审文件与 development-checklist。
7. **提交**：conventional scoped commit（如 `feat(mobilenet): migrate X5/S mobilenetv1-v4 to unified samples`）；文件归位与行为修复分开提交。
8. **板端冒烟**（用户在 X5 8GB/4GB + S100 执行）：图像类 `bash samples/<p>/model/download.sh --target x5` → `python3 samples/<p>/runtime/python/main.py --target x5 <img>`，S100 同理 `--target s100`；有 cpp 的加跑 `runtime/cpp/run.sh`；**非图像输入 sample（3dresnet 视频序列、B10 语音/观测序列、B11 文本/子模块）冒烟命令按各自 sample README 与 test_data 执行**。命令采用该 sample README 的实际命令，不机械套用此示意。先记录同板源实现基线，再按 Q4 做迁移前后对照；修复后重跑受影响项。
9. **交付评审**：按 Q1–Q5 完成客户/Agent 双视角审阅，逐项列出代码位置、文档位置和证据；必需项未过，不进入下一批。独立 reviewer 必须有独立审阅记录，作者自检不替代。

## 收尾阶段（B11 板测通过后）

1. 删除 `platforms/x5/`、`platforms/s/`（前提：能力台账按 sample×target×variant×语言逐项核对，Q1–Q5 门禁及必需板测通过，最终源增量核对完成，grep 无存活引用；未测平台明确限制且不能标全平台 ready，历史文档豁免）。**先新增 ADR**（顺延编号）记录对 ADR-0002 过渡兼容窗口的正式关闭：链接迁移台账与板测证据，声明旧入口自本 ADR 起仅文档兼容（不再可执行）。
2. X3 归档：`git mv platforms/x3 archive/x3`，更新 README/CONTEXT 链接；不新增 X3 任务。
3. registry 收敛——两类信息分开，不混入一个文件：`platforms/registry.json` 的发布线行（branch/tag/manifest_directory）改为 `docs/release/registry.json`（manifest_directory 改指新位置）；SoC 目标身份仍留在 `docs/release/platforms.json`（`platforms.py` 消费，不动）。
4. `samples/_shared/assets.py` 收敛为 `docs/release/{group}/models.yaml` 单一路径。
5. 文档统一：根 README(_cn)、Guidelines 合并版、CHANGELOG/VERSION 版本线约定（ADR-0006）、samples/README 矩阵扩到全量、台账标记为历史文档。
6. **`utils/` 引用清零评估**：grep 全仓 `utils.py_utils`/`c_utils` 引用；若迁移后仅剩历史文档引用，按 ADR 决定保留为兼容层或归档（S 侧 cpp 的 c_utils 头文件依赖是主要保留理由，需逐 sample 确认已 vendor 或改路径）。
7. CLAUDE.md 按最终目录和入口全面整理；Phase 0.5 前的最小纠偏已完成，此处不是首次修正过期规则。

## 验证方式

- **Phase 1.5 后**：3 个已迁移 sample 全量主机测试回归（H1–H4 不破坏既有行为）；X5 板冒烟确认 flat packed NV12 与原 4D 等价（H2）。
- **每批**：主机 pytest + 无 SDK 入口检查（结构/导入/参数），证据入 `evidence/`；板端冒烟由用户执行、我闭环修复。
- **规范基线后及每批**：Q3 正反例检查、README 命令/API 与双语一致性、Q4 职责与数值回归、Q5 行为评测；新增/变更规则需回归已迁移 sample。
- **Phase 1 后**：`skills` 校验三件套（`sync_references.py`、`validate_pack.py`、`unittest discover -s skills/tests`）+ `test_assets.py` 证明 manifests 搬家无破坏。
- **收尾**：`inspect_repo.py` 全仓盘点 + `rdk-model-zoo-review` sample-audit 抽查 + 本轮进度区按维度全量闭环核对（历史 P0 的 S/F/H 保持原义）（板测未覆盖项如实 not-run，S600 待 SSH 恢复）。

## 风险与对策

- **快照过期**：一律从分支 tip 提取，台账记录每文件源 SHA。
- **同名≠同协议**：先函数映射表后动手；未证明一致前保持独立模块。
- **assets.py/manifest 耦合**：A4 单 commit 原子搬迁（复制 + 改路径 + 改测试）。
- **utils 两侧分叉**：A1 先 diff 合并再开批。
- **cpp 主机不可验**：板测门槛，Python 通过不代表 cpp 通过。
- **激活语义静默漂移**：X5 raw logit（手动 sigmoid）与 S dequant 后已激活值语义不同，合并任务模块时若不按 H1 的 `output_transform` 声明区分，会出现"能跑但精度错"——固定输入双板对照是唯一可靠的检出手段。
- **过渡兼容（ADR-0002）**：旧入口按 resnet/paddle_ocr 先例保留 compat 层，并注明可执行兼容还是仅文档兼容。

## 明示不做

X3 sample 化、notebook 复活；model_zoo_web 合并；vendor act/pi0 子模块内容；默认分支切换（需另行授权）；S600 复测（SSH 未恢复）；全量数据集精度/性能回归；与迁移无关的 Skills 全面重设计（Q5 的定向补强与行为评测必须完成）；引入新云服务/CLI。

## 关键文件

- `samples/_shared/assets.py`（A4 manifest 路径改造核心）
- `samples/_shared/{platforms.py, image.py}` + `samples/vision/resnet/runtime/python/{model_binding, model_runner, tensor_io}.py`（Phase 1.5 硬化 H1–H4 的落点）
- `samples/vision/ultralytics_yolo/runtime/python/yolo_platform.py`（H5 平台 Profile 契约的参照实现）
- `docs/releases/unified-migration/x5-s-migration-map.md`（保留历史 P0；追加本轮进度区并逐维度记录）
- `platforms/{x5,s}/docs/release/models.yaml`（manifest 来源 → `docs/release/{x5,s}/`）
- `samples/vision/resnet/`（每批重构的参照模板）
- `.gitmodules`（B11 vla 子模块路径）
- `~/.claude/skills/`（Phase 0 安装目标）

## 厂商调研来源（要点索引）

Rockchip rknn_model_zoo（P1 骨架：model-first + `-t <soc>` flag + sample 内 rknpu1/rknpu2 两代 runtime 并存）；Hailo Model Zoo（P3：模型 YAML 内嵌 `supported_hw_arch` 机器可读矩阵；v2.x/v5.x 沿工具链代际分裂——印证 X3 归档）；Qualcomm AI Hub（卡片=targets×runtime×precision 矩阵）；Vitis AI（按 DPU 配置编译制品注册表）；STM32 modelzoo+services（manifest.json + per-target config 而非目录树）；Ultralytics（制品扩展名选后端 ≈ 我们的 --target）；Optimum（硬件插件包）；Axera ax-samples（P2 target-first 反例）。

## B1 最新复审状态补充（2026-09-21，dd60911）

本节更新前文历史测试状态：X5 8GB 已完成 5/5；R2 S100 正例和 S100P launcher 4 例补测已记录，S100P 仍无正向推理支持，S600 C++ 仍 not-run。独立主机测试 331 项通过，CI 同命令 0 violations / 84 精确豁免（B9 移除）。结论 changes-required / not-ready，剩余 R1a/R1b 文档修正；B1 Closed=no，B2 pending。

## 最新独立评审：B1 关闭、B2 整改（2026-09-21，716bdca）

本节取代前文历史“B1 未关闭/B2 pending”的当前状态。B1 R1a/R1b 已独立确认，全部 R1–R6 closed，Closed=yes；S600 MobileNetV2 C++ 依用户决定保持 not-run。B2 作者实现和板测已完成，独立评审 changes-required / not-ready；433 tests 与 CI checker 通过，Closed=no，不进入 B3。

- [x] B2-R1：EfficientNet 按 target 选择默认变体（x5 b2 / S lite0）；统一运行与下载入口、help/双语文档，增加遗漏参数/auto/错误组合回归，并验证受影响 S 默认入口。（整改提交见 B2 整改 commit；`SampleBindingTable.default_variant` 映射 + `default_variant_for`，download.py 同一默认并以测试钉住与契约表一致；efficientnet 25→28 OK；s100/s600 overlay 复验省略变体默认入口 rc=0 且与 lite0 记录逐位一致，显式路径不变；board evidence `post_review_remediation_b2r1` 节）
- [x] B2-R2：四 sample 根/evaluator 双语 README 同步真实板测状态与证据；区分 softmax 后 Top-K 容差、raw output 与精确平局，保留未测边界。（16 份 README + efficientnet runtime/model 双语默认变体说明；checker 4/4 零违规、CI rc=0）
- [x] B2-R3：修正板测摘要/报告/台账：28 次对照=26 次严格 ID 一致+2 次精确平局；跨实现分数并非逐字节相同。原始五板记录和 harness 已由 reviewer 保留到 evidence/2026-09-21-b2-review-inputs。（evidence `post_review_corrections` 节、评审 §6.5/§6.7、台账 Board 列同步更正；原始 record 未改动）
- [x] B2 整改后独立复审已通过（21c833a），Closed=yes，可安排 B3；仅文档修复不重复全量板测。

独立报告：`docs/releases/unified-migration/2026-09-21-b2-independent-review.md`；独立证据同名 JSON。

## B2 整改独立复审（0a6deaa）

R2/R3 独立确认关闭；R1 代码修复通过，436 tests 与 CI checker 通过，作者板端成功摘要尚缺版本/部署哈希及原始输出绑定。B2 changes-required / not-ready / Closed=no，B3 pending。此前 R1 勾选仅代表作者整改，独立关闭仍以本节为准。

- [x] B2-R1-E：保存已有 S100/S600 默认入口复验记录（准确 cwd/argv、时间、板身份、部署修复文件 SHA/可验证版本、模型/输入身份、rc、完整输出）。原 bundle 哈希不覆盖 overlay 后文件；原记录可用则不重跑，无可追溯记录时只复测两板默认入口。（原复测未存完整输出，按指示只复测两板默认入口并留证：evidence/2026-09-21-b2-r1e-default-entry/ 两板 JSON record + capture 脚本；UTC 起止时间、板身份、精确 argv/cwd、rc、完整 stdout/stderr；四个部署文件 SHA-256 板上计算且与当时 HEAD 逐一相等；制品/输入 digest 与五板记录同一；s100/s600 默认入口完整 Top-5 与原显式 lite0 记录逐 rank 一致，s600 lite2 对照复现。复审后新捕获，非回填。同轮采纳 N1（单资产不隐式默认，efficientvit 26→27）与 N2（evaluator 28 标注）。）
- [x] B2-R2/R3：README 主要状态/口径和 28=26+2 计数修正独立确认通过。
- [x] 仅针对补齐证据独立确认后关闭 B2（21c833a 已完成）；不要求全矩阵重跑。

非阻断建议见独立复审：B2-N1 缺省映射与单资产回退语义不一致（现有配置不触发）；B2-N2 evaluator 25/28 测试数标注。

## B2 最终独立关闭（2026-09-21，21c833a）

B2-R1-E 已确认：两板四文件部署哈希与当前代码相同，模型/输入/标签身份可追溯，默认入口 rc=0，完整 Top-5 在 CLI 打印精度下与旧记录一致。N1/N2 已确认。B2 pass / ready（既定范围）/ Closed=yes；本段取代前文历史 changes-required 状态。独立针对性 tests 126 全过、CI 通过；原 436 独立回归及作者本轮 437 全量回归分开记录。B3 pending，可由用户安排启动，本次不执行。

报告：`docs/releases/unified-migration/2026-09-21-b2-independent-closure.md`。

## B3 独立评审与远程板测方向（2026-09-22，f552c52）

独立评审由 Codex 负责，已完成本轮代码/主机侧审查：545 tests 与 CI 通过，changes-required / not-ready，Closed=no。
- [ ] B3-R1：FasterNet/FastViT parser 默认 base 非法；FastViT asset_reference 默认 s 也非法。统一 binding/API/CLI 默认并测真实入口。
- [ ] B3-R2：修正两 sample 多层双语 README 非法下载变体；FastViT evaluator 图片路径、成功判据同步。
- [ ] B3-R3：恢复 FastViT evaluator 双语真实源基准表，移除误套 FasterNet 数据。
- [ ] B3-R4：四 evaluator 未板测却声称 recorded smoke，改为拟采用规则/明确 B2 先例。

用户最新方向优先于此前完全暂停策略：允许主任务统筹办公室 xgs-hp-ubuntu 的 Git/gh 预检与远程板测路径；Mac 仍不探测局域网。先固定 Git SHA 对齐远程 checkout，不复制凭据；远程可用且版本/材料就绪后再执行板测。当前专用 projectless 任务交接被应用拒绝，需配置 Git 项目；既有远程任务收到只读 Git/gh/仓库预检请求。Board 仍 not-run，不凭预检改为通过；不推进 B4。

## 最新执行授权：主机开发连续推进（2026-09-22）

用户要求 Codex 接手 B4–B11 基础开发、合并及 README/代码评审，把实际板端复验和校准留给用户。此前“未板测不进入下一批”的排期门禁在此调整：每批主机验证和评审完成后可以继续，Board=not-run、Closed=no 与交付 not-ready 如实保留；不得因此宣称客户版本可发布。远程 API/设备授权继续暂停，不尝试连接。完整规则与持续板端队列见仓库 `docs/releases/unified-migration/2026-09-22-host-development-and-board-handoff.md`。

B3 R1–R4 原整改已复核，本次 208 主机 tests 与 CI checker 通过。额外三份中文 evaluator 的无关 L1 精度文案已由 Codex 清理，属于作者修正。详见 `2026-09-22-b3-host-recheck.md`。B4 从源清点开始，B5–B11 仍待实现；整体目标尚未完成。

### B4 RepGhost 主机落地

RepGhost 五变体统一实现、下载/契约、十份双语 README 已完成，七项主机回归含源数值对照通过。转换材料与历史基线保留；板测与独立评审 not-run，Closed=no。精确板测操作在 sample evaluator，完整验证证据写入 `2026-09-22-b4-repghost-host.json`。B4 其余七 sample 未实现，下一项 RepVGG；B5–B11 与整体收尾仍待完成。B3 runtime prose 残留另已补正，不把作者修正伪称独立验收。

### B4 第二组主机进度

RepVGG 6 变体、RepViT 3 变体、MobileOne 5 变体已完成统一运行/下载/绑定与双语五级 README，各 9 项主机测试通过（含逐源数值对照、metadata 拒绝、文档命令解析）。转换配方与原始八列 benchmark 保留并逐模型披露缺口。Board/独立 Review=not-run、Closed=no；B4 当前 4/8 个 sample 主机侧完成，下一步 ResNeXt、VargConvNet、GoogLeNet、HGNetV2。整体 B5–B11 与收尾尚未执行，不改变原目标。

### B4 八个样例主机实现齐备

ResNeXt/VargConvNet/GoogLeNet/HGNetV2 已加入统一分类运行与双语文档，HGNetV2 原五导出脚本/五配置保留、实际数据集 eval.py 接入统一 runner。最后一组 42 项主机测试通过，全部 B4 为八 sample/27 变体。独立复核正在进行，Board/Closed 不升级。VargConvNet/GoogLeNet 无源转换配方、HGNetV2 校准 JPEG/float32 衔接未证实均保留明确前提待办。下一步独立复核整改与 B5 清点；B5–B11/全仓收尾仍未完成。


## 用户最新执行约束（2026-09-22，本地继续）

不再使用远程电脑，全部开发、评审和主机验证在本地推进；板端环境验证暂时跳过，保持 Board=not-run、Closed=no，不再发起远程授权、任务或板卡连接。B1/B2 既有有效证据保留。


## 本地进度：B4 独立主机复核与 B5 ViT

B4 八样例均完成独立主机评审，无主机阻断；Board=not-run、Closed=no，保留 P2 逐样例 predict/context 增强建议。README 残留的完整 SHA、中文 target choices 已同步补齐，报告见 `2026-09-22-b4-independent-host-review.md`。

B5 五样例固定源清点已完成，ViT S100 int8/int16 统一实现与十份双语 README 完成、13 主机测试通过；复用现有分类核心，CIFAR-10/224/NV12双平面/nearest/default resize=0 有源对照。独立评审进行中。其余 CLIP、SigLIP、DINOv2、3DResNet 待迁移，不宣称 B5 完成。全目标仍包括 B5 余项、B6–B11 和全仓收尾；用户最终只承担真实板端复验/校准。全部本地，无远程/板卡访问、无push或发布。


ViT 独立本地主机复核完成，13 tests/4 manifest tests/checker0 已独立确认；全量641项回归通过（清单修正后受影响范围复跑），证据保留原始失败与补正。下一步 B5 CLIP/SigLIP/DINOv2/3DResNet；Board仍not-run、Closed=no，不扩大交付声明。

## 最新主机进度（2026-09-23）

- [x] B4每sample新增真实变体predict/显式阶段和A/B/A context覆盖：八套84项通过，原P2已收尾。
- [x] B5五样例主机实现、五级双语README及独立评审完成：ViT13 / SigLIP19 / DINOv2 18 / CLIP15 / 3DResNet16，共81项。全部原findings已独立确认关闭，review=pass(host)。
- [x] 全量715项通过后，3D文档整改新增2项并复跑受影响16项；当前覆盖717项。规范28samples/0violations/84原有B9精确豁免。证据与独立报告位于仓库docs/releases/unified-migration/2026-09-23-b5-*。
- [x] 只读B6源清点：EfficientSAM与MobileSAM的X5/S各自源对象/副本逐字节一致；prompt、boxshape、数值变换和编译native dtype/shape未知项已记入2026-09-23-b6-sam-source-audit.md。
- [ ] B6实现：先逐stage声明metadata允许协议，再决定两个真实消费者可共用的SAM模块；不能把X5箱形状、固定点提示或阈值差异抹平。
- [ ] B6–B11和全仓基础合并收尾仍未完成，目标持续。

全部只在本地工作，不使用远程电脑、不连接板卡。B3/B4/B5 Board=not-run、Closed=no、delivery=not-ready；既有B1/B2真实历史证据不回退。没有提交/push/发布。一次reviewer误触发本机模型下载已由主任务核实并清理，已留事件证据，不能宣称整轮零下载；不构成板测。最终留给用户的仍为实际板端复验、校准和源材料缺口补足，不能用模拟结果冒充。

## 2026-09-23 B6 本地实施中

B5 主机收尾与独立评审已通过；B6 源清点时的“B5 未闭环”仅为历史排期状态。B6 双 SAM 正在按既定规范实现共享编码/解码阶段、精确 target 制品绑定、完整转换材料合并与双语文档。每 stage 公开 pre/forward/post，pipeline 明确编排。当前资产/metadata 10 项主机测试通过，阶段数值测试正在补充，尚未独立验收或关闭本批。Board=not-run、Closed=no；不连接远程电脑、不下载模型、不提交或发布。后续 B7–B11 与全仓收尾持续待办。

## 2026-09-23 B6 主机完成与独立确认

- [x] EfficientSAM / MobileSAM：双模型六阶段共享实现，四 target 精确绑定、默认与错误边界，完整转换源能力及 20 份客户双语 README。
- [x] 分组件交叉独立评审通过，全部本轮 findings 关闭；报告 `2026-09-23-b6-independent-host-review.md`，96 文件工作树快照。
- [x] 初轮全量 781 tests；新增两项边界后复跑 shared101 / EfficientSAM19 / MobileSAM17，通过，当前覆盖783；规范30samples/0violations/84既有B9豁免。
- [x] B7 六样例只读清点：136 固定源文件、资产与特殊协议审计，见 `2026-09-23-b7-source-audit.md`。
- [ ] B7 实现：YOLOv5先建立X5/S明确物理协议与输出变换，ByteTrack依赖该detector；FCOS、YOLOWorld、LPRNet、MODNet保持任务差异并完整重写五级双语README。
- [ ] B8–B11、全仓整合与最终本地主机评审继续待办；整体目标未完成。

B6 Board=not-run、Closed=no、delivery=not-ready。依用户指示全部本地、不用远程、不探测板卡；实际复验和校准排入交接队列。B6 未下载真实模型，无提交/push/发布。这里的host通过不替代真实SDK、模型、OE或数据集验证。


## 2026-09-23 用户主动暂停与 Claude Code 交接

用户要求做到这里，暂时停止并交给之前的 Claude Code。Codex goal=paused，全部三个并行任务已中断；不再继续实现或运行验证。B6 主机已完成，B7 仍在整改且未独立闭环；B8 只有250固定源文件只读清点，B9–B11及全仓收尾未完成。剩余本地6大步=B7收尾+B8+B9+B10+B11+全仓整合。

接手必须先读仓库 `docs/releases/unified-migration/2026-09-23-claude-code-handoff.md` 及其文件hash快照。HEAD仍16c5d04，B3–B7大量未提交/untracked成果须保留，不能clean/reset。LPRNet/MODNet存在刚写入但未复跑的中途整改；YOLOv5 C++、YOLOWorld及新README仍有待项。Board=not-run、Closed=no、delivery=not-ready，用户最新本地/跳过板端约束继续有效。
