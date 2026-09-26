# X5/S 非板端完整交付与 README 修复计划

权威范围：用户2026-09-26要求Codex全面接手，完成除板端测试外全部工作，特别对齐原分支根README、sample及子目录README质量，以Ultralytics YOLO为参照。原X5/S Spec仍控制架构。历史报告保留。

## 执行裁定

- Ruling: Codex直接实现、主机验证与GitHub同步；不再依赖Claude Code/GLM额度。成本：后续交接须读新台账而非重启旧任务。
- Ruling: 板端运行是独立未完成维度，不再阻断B8–B11主机迁移；缺模型、OE工具链和数据集的事实单列，不能虚构通过。成本：真实SDK差异可能在后续板测暴露，必须保留精确可重测提交。
- Ruling: 每份README以原源内容与当前代码双向核查；标题/锚点/字数达标不能替代可用性。历史性能表保留并提供直接入口，不声称迁移后重测。
- 使用既有隔离工作树rdk-b7-board-integration。用户已授权每轮commit/push；不发布release或更改默认分支。

## 完成清单

- [ ] H0 汇合未合入的已审修复；修正B3工具依赖隔离、原生audit与两侧日志归档，完成主机回归。
- [ ] H1 README覆盖审计：根、sample索引、平台说明、每个sample及model/runtime/python/runtime/cpp/conversion/evaluator；建立源能力→新位置映射，修复失链与默认命令矛盾。
  - 2026-09-26 总入口、36 个 Sample 索引及平台注册说明已更新，362 个本地链接通过；原平台正文保留。逐 Sample 深度内容审核仍继续，H1 不整体关闭。
- [ ] H2 Ultralytics YOLO全部层级文档与代码规范示范：可复制最短流程、全任务命令、完整API输入变量、参数默认/输出/模型/转换/评估/历史指标/故障说明、中英一致。
  - 2026-09-26 文档子项已完成：Ultralytics 根/model/runtime/python/runtime/cpp/conversion/evaluator 双语改写；36 samples / 0 violations / 0 exemptions，原 84 条基线与 CI 旗标已删除。H2 的实现职责审计仍待完成。
- [ ] H3 B7主机整改集成；客户文档跟随实际代码与历史板证据，保持板测缺口。
- [ ] H4 B8全部样例完整源能力迁移与测试、双语README。
- [ ] H5 B9独立YOLO样例收编及YOLOE迁移；README 基线及 workflow 豁免旗标已于 2026-09-26 提前清零；系列收编和 YOLOE 迁移仍 pending。
- [ ] H6 B10语音/机器人样例迁移；保留执行边界，不触发实机控制。
- [ ] H7 B11大模型和VLA来源/gitlink/资源集成，不用空壳冒充上游能力。
- [ ] H8 全仓共享职责、datasets、旧路径兼容、manifest/catalog、七skills原包来源与文档/Agent导航、上游增量核对。
- [ ] H9 全部相关主机测试、无SDKhelp/list/dry-run、双语命令与本地链接核验、独立整体评审、GitHub同步；最终报告列出板测及外部环境未验证项。

## 文档验收

根README提供维护范围、完整任务导航、系统/制品区别、获取与启动、结构、贡献/许可/历史版本。sample文档说明能力/限制/发布组合/依赖/最短运行/输入输出/源码阅读/库集成/转换与评估/故障。子目录给出实际任务所需步骤、工作目录、参数、默认值、输出和成功判据；不能只让用户运行--help或打开历史分支寻找必需流程。源中存在的基准、图示、参考链接和特殊变体不能无说明丢失。中英使用同命令与支持范围。

## 进度记录

2026-09-26：确认develop仍为3b6f5aa，既有集成分支保留SAM及README修复；已合并最新develop报告。发现根README仍自述三试点，Ultralytics model/evaluator文档相比源严重压缩；先补evaluator可操作流程，不宣称H2整体完成。

2026-09-26：完成 Ultralytics model/evaluator 双语操作说明及 Python 示例纠错；78 项主机测试通过，文档检查 36 samples / 0 violations / 60 exemptions。详见 [质量记录](../../releases/unified-migration/2026-09-26-readme-quality-review.md)。H2 其余层级与全部后续批次继续，未关闭。

2026-09-26：H0 的 B3 依赖隔离及原生 audit/双侧日志归档已修复，分别通过 30 和 79 项主机测试；代码已汇入集成分支。详见 [工具整改](../../releases/unified-migration/2026-09-26-tool-remediation.md)。全部修复分支归并与最终主机回归仍待核对，H0/H3 不提前整体勾选。

2026-09-26：B8 源码核对发现共享反量化对逐通道 scale 丢弃单个非零 zero-point；已修正广播，266 项主机测试通过。见 [数值修正](../../releases/unified-migration/2026-09-26-quantization-offset-review.md)。这是明确披露的源代码缺陷修正，不声明板端等价；H4 的八类样例迁移仍未完成。

2026-09-26：B8 PointNet 主机迁移完成（四阶段、显式下载、严格目标/metadata、五层双语 README）；326 项相关主机测试通过，规范范围 37 samples / 0 violations / 0 exemptions。见 [PointNet 记录](../../releases/unified-migration/2026-09-26-b8-pointnet-review.md)。独立评审与板测未运行，其余七类 B8 样例继续。

2026-09-26：B8 UNet 五骨干主机迁移完成，保留转换/评估能力并将 X5 evaluator 接入统一三阶段；README 保留原始基准/MIT 内容并补齐操作与边界。373 项相关主机测试通过，规范范围 38/0/0 exemptions。见 [UNet 记录](../../releases/unified-migration/2026-09-26-b8-unet-review.md)。其余六类 B8、H0–H9 全范围继续，板测/独立评审未运行。

2026-09-26：H8 子项已统一 PointNet/UNet 的单输入单输出 raw runner，保留各 sample 物理张量与语义契约；178 项受影响主机测试通过。PP-LiteSeg 原文档输出语义、默认图片、导出前提及构建/校准问题已核对并记录，[详见](../../releases/unified-migration/2026-09-26-array-runner-review.md)。这些 PP 项是下一步迁移待修项，未标成已完成。

2026-09-26：B8 PP-LiteSeg（846c519）已完成主机重构及五层双语说明，修正类别图/图片/转换门禁；UNetMobileNet 已完成 Python/C++ 阶段职责分离和六层双语说明，使用共享 split-input runner 并修正逐通道量化排序、原生资源失败处理。真实 SDK 构建及板测仍未执行，独立验收未关闭。见 [PP-LiteSeg](../../releases/unified-migration/2026-09-26-b8-ppliteseg-review.md) 和 [UNetMobileNet](../../releases/unified-migration/2026-09-26-b8-unetmobilenet-review.md)。B8 尚余 YOLO26 Depth、Depth Anything V2、LaneNet、DiffusionDrive；全部 H0–H9 范围保持不变。
