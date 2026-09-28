# X5/S 非板端交付：最终独立对齐

评审者：Codex。实现者：Claude Code + GLM。复验基线为
`cc382d1bfc913d7d3fd3b8e6fd6d919f5b5c0616` 加本轮留有摘要的工作区整改包；
最终版本由包含本报告的 Git 提交确定。结论：**接受 H0–H9 的当前非板端范围**。
Git 同步以实际 push 及远端分支哈希复核为准，不能由本报告文本证明。

范围按 [当前计划](../../superpowers/plans/2026-09-26-host-completion.md) 顶部用户裁定解释：
板测暂缓，量化 README 继承可信源方案，不重新下载权重、导出、校准、编译或验证量化精度。
本结论不代表真实 SDK/ABI、模型精度、机器人控制、完整板型矩阵或正式发布通过。

## 需求与证据对齐

| 项目 | 最终非板端结论与独立证据 |
|---|---|
| H0 集成 | [集成审查](2026-09-28-integration-independent-review.md) 已确认既有修复分支的补丁覆盖；本轮没有重复合并旧分支。本轮完整相关主机回归通过。 |
| H1 文档 | 51 个 native sample、536 份适用层级指南齐备；逐源语义依据包括 [B1/B2](2026-09-28-readme-depth-b1b2-independent-review.md)、[分类族](2026-09-28-readme-depth-classifiers-independent-review.md)、[特殊模型](2026-09-28-readme-depth-special-independent-review.md)、[双语整改](2026-09-28-readme-pair-rereview.md)、[源图片](2026-09-28-source-readme-image-independent-review.md)、[入口](2026-09-28-entry-docs-independent-review.md) 和下列批次审查。最终改动由 [文档复审](2026-09-29-document-remediation-independent-review.md) 接受。 |
| H2 Ultralytics | [职责、原生与跨层检查](2026-09-28-ultralytics-final-independent-review.md) 已接受；本轮 test_data 指南 H2-DATA-R1 关闭。保留各原生程序实际支持差异，不能外推缺失 OBB/S-v10 原生路径。 |
| H3 B7 | [整批代码复审](2026-09-28-b7-batch-independent-review.md) 的 117 份代码/测试摘要无漂移；B7-DOC-R1/R2/R3 已关闭。六样例非板端整批接受，历史 smoke 与数值对照分开，当前提交未重新板测。 |
| H4 B8 | 延续 [B8 整批验收](2026-09-28-b8-batch-independent-review.md)；本轮全部对应主机套件通过。 |
| H5 B9 | [范围与源内容](2026-09-28-b9-batch-independent-review.md)、[YOLOE 运行时及转换/evaluator 文档](2026-09-28-yoloe-independent-review.md) 加本轮 YOLOE-DOC-R2 关闭，满足整批非板端验收。四个独立 S YOLO ID 退役，独立 YOLOE/World/Depth 保留；S YOLOE 浮点发布制品缺口没有被主机夹具掩盖。 |
| H6 B10 | 延续 [B10 整批验收](2026-09-28-b10-batch-independent-review.md)；Paraformer 使用已有依赖环境重跑 82 项无跳过，其余语音/机器人主机套件通过。 |
| H7 B11 | [Gemma Text](2026-09-28-gemma-text-stages-independent-review.md) 两个原始 sanitizer 反例关闭，30 项主机及 19 项原生通过；[MiniCPM 核心](2026-09-28-minicpm-core-independent-review.md) 20 项及此前四份编译示例保留；[VLA 父仓集成](2026-09-28-vla-independent-review.md) 的两个独立固定 gitlink 再次核实。B11 非板端整批接受，完整板端/模型验收仍开放。 |
| H8 共享、目录与 Skills | [datasets](2026-09-28-datasets-independent-review.md)、[Catalog](2026-09-28-catalog-independent-review.md)、[版本跟进](2026-09-28-catalog-version-followup-review.md)、[上游 Skills 增量](2026-09-28-x5-skills-increment-review.md)、[Agent 入口](2026-09-28-agent-entry-independent-review.md) 及本轮共享导航/清单发现/发布引用整改已接受。七 Skills 的最小显式加载行为按 [独立行为评审](2026-09-29-skills-behavior-independent-review.md) 限定接受；三个首轮回答错误仍标为失败维度，不宣称完整准确率或生产安装验证。 |
| H9 验证与收尾 | 下表主机、规范、链接、Catalog 检查通过；无 SDK 的 help/list/dry-run 与库例子沿用对应样例的真实执行证据，源码未改动部分不反复重跑。最终源码摘要与 Git 变更检查绑定本次验收；仅同步现有集成分支。 |

H1 是历次逐源内容审查与最终变更复核的汇合，并非声称本轮重新逐字阅读全部
599 份 README。链接/标题数量不能代替语义验收。原发现、初次失败、作者报告和
历史时间点状态均保留，当前结论只覆盖明确的非板端范围。

## 本轮独立执行

证据位于 [2026-09-29-independent-closeout](evidence/2026-09-29-independent-closeout/)。

| 检查 | 结果与边界 |
|---|---|
| Python 主机测试 | 1,631 项通过：53 组 1,523 项，加 Gemma 30、Skills 63、YOLOE evaluator/export synthetic 15；环境修正后的重跑只计一次。 |
| 单独复跑的 CTest | Gemma 19（ASan/UBSan）、YOLOE 11、Ultralytics 12，共 42 项通过；与 Python 包装测试存在重叠，不能相加称作互不重叠覆盖。 |
| Catalog publisher | Node 22.23.2 下 check 通过，130 项/18 文件测试、类型/manifest 校验及可复现构建；catalog-v1.0.0-bcb085e3cb057c77。历史 dataset 缺失警告保留。 |
| Sample contract | 51 样例、0 violations、0 exemptions；有明确策略 skip，不宣称零 skip。 |
| README 导航 | 599 文件、3,365 本地 inline 引用，无失链/未解析锚点；不验证远端 URL、引用式链接及 submodule 内部。 |
| Skills | 63 项主机测试，7 Skills/84 eval 定义结构与引用同步通过；14 初始真实 Agent 会话及 3 次显式纠正分别留存。 |
| 原始反例 | Gemma mask 边界及短 hidden 输入两份原始驱动在 sanitizer 下正确拒绝，无内存错误；不以新测试替换失败证据。 |

默认 Python 为 3.14.7。首轮 YOLOE 因该环境缺 ONNX 有 9 个错误，Paraformer
因缺可选依赖跳过 13 项；原始日志不删除。随后显式复用现成 YOLOE PYTHONPATH
和 Python 3.12.14 Paraformer 环境，分别 44/82 全通过、无跳过。没有安装新依赖，
没有实跑真实权重量化配方。API/转换工具的合成输入单元测试仍属于普通主机回归。

## 最终文档跟进与版本边界

独立复核 Gemma 根 README 双语对及 samples 索引双语对的四行状态更新：
准确区分已实现/主机接受与 vendor ABI/live model/board/quantization 未验证；
仅新增审查入口与状态描述，没有改变命令、历史图表或实现。客户文档所说的
aggregate B11 open 指完整验收；本报告关闭的是用户此次约定的非板端维度。

远端观察仍为 develop `e0759d5a`、rdk_x5 `d1b24f65`、rdk_s `380e1a2b`。
本地 integration 与 develop 为 176/4 各自独有提交，未声称最新 develop 已合并。
两份 VLA gitlink 核定为 ACT `326ea043be204de25223d95c7d918efe8672dc66`，
PI0 为 `a32de276bc1681a2b1531012de111eaa1c16acb6`，见 vla-pins.json。
其他 worktree 的未提交工作不属于本轮包，不改写或清理。

主迁移账本的 Board/Closed 保留完整交付口径，因此 Closed=no 与本报告的
非板端完成并不矛盾。历史 MODNet 制品缺失、S100P 404、MiniCPM 历史精度失败、
真实 SDK/ABI 与完整模型执行未知均保留。未发布模型、Skills、Catalog/Pages，
未打 tag、未改默认分支、未合并 develop，也未将历史源图/性能表改写成本轮结果。
