# Model Zoo 仓库规范（develop 统一架构版）

> 状态：Phase 0.5 Q1 基线（2026-09-20）。本文件是 `develop` 分支的规范入口。
> 交付分支 `rdk_x5` / `rdk_s` 上的同名文件在各自 ref 上继续有效；Phase 1（A6）将
> 其编码、注释与任务规范章节并入本文件并按 develop 目录结构改写目录章节。

## 定位与规范层级

`develop` 采用 sample 为中心的统一架构：X5 与 S（S100/S100P/S600）的 sample 统一存放
于 `samples/`，硬件通过每个 sample 的 `--target` 参数选择，平台身份只认板卡事实
（`/sys/class/boardinfo/soc_name` → socinfo → device-tree），未知即报错、不静默回退。

规范的单一权威分层如下，各层不重复维护同一规则：

| 层 | 文件 | 管什么 |
| --- | --- | --- |
| 协作约束 | [`AGENTS.md`](../AGENTS.md) | 分支纪律、权限边界（只读默认/下载/板测/发布须授权）、冲突报告 |
| README 契约 | [`docs/sample-standards/readme-contract.md`](sample-standards/readme-contract.md) | 各级 README 必答问题、固定章节 ID、双语配对、统一内容纪律；与旧规范的冲突处置记录在其 §6 |
| 推理契约 | [`docs/sample-standards/inference-contract.md`](sample-standards/inference-contract.md) | pre/forward/post/predict 职责边界、阶段数据流、多阶段编排、必需测试 |
| 迁移记录 | [`docs/releases/unified-migration/`](releases/unified-migration/) | 台账、批次评审、证据；迁移历史不放客户 README |
| 架构决策 | [`docs/adr/`](adr/) | ADR-0001 源仓/文档站分离、ADR-0002 过渡兼容、ADR-0006 统一版本线等 |

自动检查（Q3 落地）执行 README 契约中可机器判定的规则；不可机器判定的交语义评审，
不报告自动通过。

## 目录组织（develop 现状）

```text
.
├── CLAUDE.md                          # Claude Code 工作指引（随迁移纠偏）
├── AGENTS.md                          # 协作约束
├── datasets/                          # （Phase 1 A2 落位）示例数据与数据集脚本
├── docs/
│   ├── adr/                           # 架构决策记录
│   ├── release/                       # SoC 目标身份等发布数据
│   ├── releases/unified-migration/    # 迁移台账、批次评审与证据
│   ├── sample-standards/              # README/推理契约与模板（本规范引用）
│   │   └── templates/                 # 六级 README 双语模板
│   └── superpowers/                   # spec 与历史 plan
├── platforms/{x5,s,x3}/               # 迁移期冻结快照（来源分支为准，收尾删除；x3 归档）
├── samples/                           # 统一 sample（迁移目标形态）
│   ├── _shared/                       # 平台身份/资产绑定/NV12 等已证双消费者共享设施
│   └── vision/…                       # 每样本：model/ runtime/ conversion/ evaluator/ test_data/ tests/
├── tools/                             # catalog-publisher、sample_contract（Q3 落地）等
├── utils/                             # （Phase 1 A1 落位）过渡兼容层（ADR-0002）
└── skills/                            # （Phase 1 A3 落位）Skills 源树（rdk_x5 维护线）
```

迁移期清单位置：`platforms/{x5,s}/docs/release/*.yaml`；Phase 1（A4）原子搬迁至
`docs/release/{x5,s}/` 并同步改 `samples/_shared/assets.py`。

## Sample 布局与语言覆盖

每个 sample 遵循 `model/ + runtime/{python,cpp} + conversion/ + evaluator/ + test_data/
+ tests/ + 双语 README(_cn).md` 的固定布局，详见 README 契约 §1 的必需性规则与
`docs/sample-standards/templates/`。语言覆盖按 sample 实际交付声明（支持矩阵三态），
**不默认双语言**；未提供 C++ 时不得声称双语言支持。

模型制品不入库（`.gitignore` 拦截 `.onnx/.bin/.hbm/.pt` 等），通过
`model/download.sh --target …` 显式准备并做哈希校验；未知校验值按契约写
`sha256: null (unknown)`，禁止伪造。

## 版本与发布

发布流程文档 `docs/RELEASE.md` 与在线目录 `docs/catalog/` 当前在 `rdk_x5`
（ADR-0001：源仓与文档站分离）；统一源的版本线约定遵循 ADR-0006，Phase 1 起在本仓
落实。历史 ADR 与迁移台账中的发布事实优先于本节的一般性描述。
