# Phase 0.5 — Q4 参照样例双视角验收评审（2026-09-21）

对应计划：Phase 0.5 Q4「参照样例与双视角验收」。上一份报告：
[Q3 检查器评审](2026-09-20-phase05-q3-review.md)。证据：
[evidence/2026-09-21-phase05-q4-evidence.json](evidence/2026-09-21-phase05-q4-evidence.json)。

## 交付内容

- **resnet（单模型参照）**：6 级 × 双语共 12 个 README 全部按 Q1 模板重写
  （8/5/7/7/8/7 固定锚点），支持矩阵、快速体验、参数表、阶段 I/O、
  集成示例、转换缺口与评估边界均落到实际事实。
- **paddle_ocr（多阶段参照）**：同样 12 个 README 重写；两阶段
  （DB 检测 → 裁剪 → CRNN+CTC 识别）的阶段 I/O 表成为仓库多阶段任务的
  文档范本；runtime/python 参数表 15 行与 `build_parser()` 逐项机器对齐
  （含互斥模式组 `--list-models`/`--dry-run`/`--prepare`）。
- 按 Q1 契约 §5.3，迁移历史/源映射审计表从客户 README 移除，只保留
  必要兼容说明（旧入口转发一行）。
- 评审中发现并修复一处文档缺口：模型路径参数缺省时的 `model/<filename>`
  默认查找未记载（详见下文）。

## 验证结果（全部本机实跑，非推断）

| 检查 | 结果 |
| --- | --- |
| Q3 检查器 resnet | 0 violations，2 处可见 policy skip（legacy.py 兼容垫片、main.py CLI 层），exit 0 |
| Q3 检查器 paddle_ocr | 0 violations，1 处可见 policy skip（main.py CLI 层），exit 0 |
| 主机测试 | resnet 39 OK / paddle_ocr 43 OK / ultralytics_yolo 59 OK / _shared 17 OK / 检查器 23 OK |
| `--list-models`（两 sample） | exit 0，打印的完整引用与 model README 一致 |
| `--dry-run`（两 sample） | exit 0，静态契约（packed/split NV12、输出名/形状/dtype/语义）与阶段 I/O 表一致 |
| 评估器实跑 | paddle_ocr evaluator 用合成记录跑通：precision/recall/F1=1.0、exact_rate=1.0、status=measured、IoU=0.938125，stdout 与 `--output` 文件一致 |
| 集成示例导入 | 6 个模块 importlib 导入成功（paddle_ocr 4 + resnet 2） |
| 迁移范围检查 | 3 samples，84 violations——全部是 ultralytics_yolo R-README-SECTIONS |

关于 84 项违例：这是计划核准的中间态。计划明示第三个试点
ultralytics_yolo「不在本轮参照内，其 Q1–Q5 合规在 B9 收编时按同一门槛
验收（收编前仅维持现状，不提前重写）」，且禁止用放宽规则换取绿灯。
CI 在 B9 前对该 sample 保持红是诚实的状态。

## 双视角评审记录

**客户阅读路径**（按 README 从准备走到结果，逐步执行/核对）：
两个 sample 的快速体验命令均标注 cwd、输入来源与成功判据；host 可执行
步骤全部实跑通过；示例无未定义变量（`image` 均由 `cv2.imread` 定义）；
本地链接经 R-README-LINKS 零失效。发现的唯一缺口：路径参数缺省时绑定
默认解析到 `model/<filename>`，原稿只写了显式路径用法——已在
paddle_ocr model/runtime 与 resnet runtime 的参数说明（en+zh 共 6 处）
补上，复检仍 0 violations。

**Agent 开发路径**（只凭 README 定位接口、参数、制品、转换/评估入口，
再对照代码核实）：参数表经检查器 import 模式与真实 `build_parser()`
锁定（en/zh 双语一致）；阶段职责经 R-STAGE-PURITY AST 扫描零违例；
`--dry-run` 打印的绑定契约与文档阶段 I/O 表逐字段一致；转换/评估入口
链接可达且命令完整。

## 同板前后对照记录（容差按维度）

原则：同板、同制品字节、同输入、同参数，canonical 对旧入口。

| 维度 | 容差 | 状态 |
| --- | --- | --- |
| 预处理数值（NV12/RGB 张量） | 精确相等（同一确定性算法、同字节） | passed（X5 双板 + S100，含默认与保持长宽比路径，2026-09-17） |
| raw 输出（F32 张量） | 精确相等 | passed（X5 双板 + S100，2026-09-17） |
| 任务结果（类别 ID+raw 分数 / 框+文本） | 精确相等，标签排版在 raw 分数之后比较 | passed（X5 双板 + S100，2026-09-17） |
| C++ 输出 | resnet Top-5 文本相等；paddle_ocr 渲染像素相等 | passed（S100，2026-09-17） |
| ONNX 导出冒烟（resnet 主机） | 契约 1e-4，实测最大绝对差 1.79e-06 | passed（主机；仅图检查，非 BPU 编译） |

not-run（如实保留，不用他板替代）：S600 复验（SSH 未恢复）、S100P
（无已批准 resnet18 资产/无审计 OCR 模型对）、Q4 之后的板端复冒烟
——本次提交只改文档、未动运行时/转换源码，2026-09-17 板端证据仍绑定
在未变更的运行时归档上；下一道板端门禁是用户随 B 批执行的冒烟。

源证据归档：runtime `77c9532f…`（resnet）、`7e75debc…`（OCR）、
S100 C++ `ba1d7401…`，均见
[2026-09-17 集成评审](2026-09-17-integration-review.md)。

## 结论与边界

- Q4 对两个参照 sample 达成：README 合规（0 violations）、参数机器锁定、
  双视角评审通过、前后对照与容差成文、缺口如实标注。
- 边界：CI 未在真实 runner 上执行（本地用同一命令验证）；板端陈述全部
  锚定 2026-09-17 证据而非新板测；ultralytics_yolo 文档留待 B9。
- 进度区更新：resnet、paddle_ocr 两行 Docs → done，Review → passed，
  证据列挂本报告。

**下一步：**Q5——Skills 定向补强与行为评测（不新增第八个 Skill），
随后进入 Phase 1 仓库级设施。
