# X5/S 非板端完整交付与 README 修复计划

> 2026-09-28 最新分工：用户已改为 Claude Code + GLM 实现，Codex 只做方向、独立评审、整改派发和 GitHub 同步；覆盖下文此前“Codex直接实现”的规则。MiniCPM 实现包请先阅读 [reviewer 基线三项复现](../../releases/unified-migration/2026-09-28-minicpm-reviewer-baseline.md)。不得绕过此分工由 Codex 自行修产品代码。


## 2026-09-28 最新用户裁定：量化 README 只做重构优化

适用于全部 Sample，而非仅 Paraformer：现有 README 中的量化方案由用户确认真实可用，
按可信源内容继承，只优化结构、表述、双语一致性、相对路径和导航，不重新证明方案可行性。
不再为了 README 验收实际执行权重下载、导出、校准、OE/Mapper 编译、HMCT 仿真或量化精度验证，
也不安装／配置工具链或调用远程机器来完成这些验证。

这些实跑项目从本次完成条件中移除，缺少工具链、数据集或量化实测结果不再阻塞交付，
不得在后续轮次重新把它们列为必须补齐的工作。历史已完成的证据原样保留；继承的源结果
保留原有来源与版本，不改写成本轮实测。代码重构所需的普通主机单元／回归测试继续；
板端验证仍按用户此前决定跳过。其他迁移和 README 质量要求保持不变。

本裁定覆盖下文及历史报告中“实际 OE/HMCT/量化验证待完成”等旧的验收要求。

## 2026-09-27 用户范围修订（2026-09-28 落地）

常规 YOLO 只维护统一 Ultralytics YOLO / YOLO26，以及已有 YOLOv5s sample。
S 独立 yolo11、yolo11_pose、yolo11_seg、yolov13_imoonlab 属重复实现，取消收编，
从活动制品/目录数据和客户入口退役；不再要求其独立 Python/C++、量化流程或转换配方迁移。
优先保留模型直接输出浮点结果的制品，不维护 Python 后处理中手动反量化的重复路径。
用户另明确：YOLOE、YOLO-World、YOLO26 Depth 的独立能力继续保留。
历史平台快照、固定源提交和既有评审证据保留追溯，不作为活动支持清单；下文旧收编要求
按本修订解释。其他非板端工作仍完整执行，板端验证暂缓。

权威范围：用户2026-09-26要求Codex全面接手，完成除板端测试外全部工作，特别对齐原分支根README、sample及子目录README质量，以Ultralytics YOLO为参照。原X5/S Spec仍控制架构。历史报告保留。

## 执行裁定

- Ruling: Codex直接实现、主机验证与GitHub同步；不再依赖Claude Code/GLM额度。成本：后续交接须读新台账而非重启旧任务。
- Ruling: 板端运行是独立未完成维度，不再阻断B8–B11主机迁移；缺模型、OE工具链和数据集的事实单列，不能虚构通过。成本：真实SDK差异可能在后续板测暴露，必须保留精确可重测提交。
- Ruling: 每份README以原源内容与当前代码双向核查；标题/锚点/字数达标不能替代可用性。历史性能表保留并提供直接入口，不声称迁移后重测。
- 使用既有隔离工作树rdk-b7-board-integration。用户已授权每轮commit/push；不发布release或更改默认分支。

## 完成清单

- [x] H0 汇合未合入的已审修复；修正B3工具依赖隔离、原生audit与两侧日志归档，完成主机回归。 已由2026-09-28独立集成评审确认，见下方记录。
- [ ] H1 README覆盖审计：根、sample索引、平台说明、每个sample及model/runtime/python/runtime/cpp/conversion/evaluator；建立源能力→新位置映射，修复失链与默认命令矛盾。
  - 2026-09-26 总入口、36 个 Sample 索引及平台注册说明已更新，362 个本地链接通过；原平台正文保留。逐 Sample 深度内容审核仍继续，H1 不整体关闭。
- [ ] H2 Ultralytics YOLO全部层级文档与代码规范示范：可复制最短流程、全任务命令、完整API输入变量、参数默认/输出/模型/转换/评估/历史指标/故障说明、中英一致。
  - 2026-09-26 文档子项已完成：Ultralytics 根/model/runtime/python/runtime/cpp/conversion/evaluator 双语改写；36 samples / 0 violations / 0 exemptions，原 84 条基线与 CI 旗标已删除。H2 的实现职责审计仍待完成。
- [ ] H3 B7主机整改集成；客户文档跟随实际代码与历史板证据，保持板测缺口。
- [x] H4 B8全部样例完整源能力迁移与测试、双语README（2026-09-28 非板端整批独立验收通过；板端仍 not-run）。
- [ ] H5 B9统一YOLO整理、重复独立系列退役及YOLOE迁移；README 基线及 workflow 豁免旗标已于 2026-09-26 提前清零；重复系列按用户新要求退役；YOLOE Python/C++、导出/转换准备/evaluator 及双语说明已完成主机实现，整体验收仍 pending。
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

YOLO26 Depth source audit: [contract findings](../../releases/unified-migration/2026-09-26-b8-yolo26-depth-source-review.md). Source audit only; implementation, six-level bilingual documentation and host acceptance remain pending. Preserve all H0–H9 scope.

2026-09-26：YOLO26 Depth 已完成 20 制品 Python 与 X5 C++ 主机迁移、六层双语 README 和离线转换/评测流程，主机结果见 [迁移记录](../../releases/unified-migration/2026-09-26-b8-yolo26-depth-review.md)。板端及真实 SDK/OE 仍 not-run、独立评审未关闭；B8 后续为 Depth Anything V2、LaneNet、DiffusionDrive，全部 H0–H9 范围继续。

2026-09-26：Depth Anything V2 源审计确认逐像素 RGB z-score 与注释不符、恒定图除零、可选 letterbox 未裁填充、S100P 无独立制品；见 [审计](../../releases/unified-migration/2026-09-26-b8-depth-anything-source-review.md)。统一实现仍在进行，尚未更新完成计数或验收状态。

2026-09-26：Depth Anything V2 已完成主机迁移、五层双语 README 和历史图表保留，见 [记录](../../releases/unified-migration/2026-09-26-b8-depth-anything-review.md)。S100P 缺制品仍拒绝，板测 not-run，独立评审未关闭。B8 继续 LaneNet、DiffusionDrive，H0–H9 全范围不变。

2026-09-26：LaneNet 源审计完成：无聚类、Python/C++ 着色差异、S64 二值输出、第三输出声明未证实、转换脚本/准确率图缺失均已记录；共享 S64 拼写归一化通过回归。见 [审计](../../releases/unified-migration/2026-09-26-b8-lanenet-source-review.md)。LaneNet 样例迁移仍 pending，B8 与 H0–H9 未关闭。

2026-09-26：LaneNet Python/C++ 主机迁移与六层双语 README 已完成；22 项 sample 测试通过，原生核心/SDK 资源管理由主机伪 SDK 验证，完整原生 SDK/OpenCV 构建、OE 与板测均 not-run。独立整体评审仍待执行，Closed=no。见 [记录](../../releases/unified-migration/2026-09-26-b8-lanenet-review.md)。B8 继续 DiffusionDrive，H0–H9 全范围保持开放。

2026-09-26：DiffusionDrive 主机迁移完成，保留四输入/输出、五案例运行、严格离线评估及六层双语 README；23 项 sample 测试通过。见 [记录](../../releases/unified-migration/2026-09-26-b8-diffusiondrive-review.md)。B8 全部样例已有主机实现，独立整体评审仍待执行，板端/真实 SDK/OE 未验证，B8 与 H0–H9 不提前关闭。后续继续 B9 收编及 YOLOE。

2026-09-27：B9 四个独立 YOLO 源目录的 78 个文件已逐字节对齐固定 S pin；十个原始制品身份已接入统一准备/路由，并补充双语路径、阈值和边界说明。反量化职责、姿态 logits/概率差异、显式 context、C++ 和完整转换/评测说明仍需迁移，不能据入口接通关闭 B9/H2。见 [进行中记录](../../releases/unified-migration/2026-09-27-b9-source-consolidation-review.md)。

2026-09-27：DFL/YOLO26 检测已将反量化移出 forward，补齐 SDK 逐通道量化 metadata，并以 PreparedDetection 显式携带几何；旧 stage 访问保留无状态适配。详见 [阶段职责记录](../../releases/unified-migration/2026-09-27-yolo-stage-purity-review.md)。上一轮 B9 台账名称导致的 R-SCOPE 检查失败及错误通过汇总已如实更正。其余 Ultralytics 任务、原生能力、YOLOE 与 H0–H9 全范围仍继续，不提前关闭。

2026-09-27：DFL 分割改用共用 raw runner 与显式几何，补回 S YOLO11 反量化并修正边界 mask 负切片；双语库示例实际在主机夹具执行。见 [分割记录](../../releases/unified-migration/2026-09-27-yolo-segmentation-review.md)。姿态/分类/OBB/YOLO26 分割、原生能力、YOLOE 和 H0–H9 剩余事项仍开放；板端/真实 SDK 未验证。

2026-09-28：Ultralytics YOLOv8/11/26 分类接入共用 raw runner、严格单输出 metadata 绑定与独立数值后处理；双语 README 新增完整分类 API 示例和 CLI/库 resize 差异。固定 X5/S 源前后处理对照见[分类记录](../../releases/unified-migration/2026-09-28-yolo-classification-review.md)。仅主机验证，真实 SDK/板测 not-run；其余任务职责、原生代码及 H0–H9 全范围继续。

2026-09-28：S YOLOv10 复用 DFL 检测三阶段并固定 no-NMS 契约，删除重复加载/图像处理/位置式解码；X5 NMS 分派保留。补齐双语 API 示例、源代码对照、实际取整几何修正及阈值边界测试，见[v10 记录](../../releases/unified-migration/2026-09-28-yolo-v10-review.md)。板端和独立整体评审 not-run，H2/H0–H9 继续。

2026-09-28：YOLO26 姿态已复用共用 pose 三阶段，单独绑定直接 LTRB/关键点偏移公式；保留 X5 字典列表与 S 四元组旧接口，补齐双语 API 示例。固定源解码、置信度范围、实际取整几何、输出生命周期及两侧旧适配器测试见[姿态记录](../../releases/unified-migration/2026-09-28-yolo26-pose-review.md)。YOLO26 分割/OBB、原生能力及 H0–H9 剩余内容继续，板测/独立整体评审 not-run。

2026-09-28：YOLO26 分割复用共用三阶段和 LTRB 绑定，保留先插值概率再二值化的 mask 算法，与 DFL mask 路径明确区分；X5 整图 mask 适配器补齐显式 transform。双语示例、固定源对照和输出生命周期见[分割记录](../../releases/unified-migration/2026-09-28-yolo26-segmentation-review.md)。OBB、原生、YOLOE、B10/B11/H8 及整体独立评审继续，板测/真实 SDK not-run。

2026-09-28：YOLO26 OBB 改用共用 runner、严格浮点角色绑定和显式图片几何，数值/NMS 下沉到独立解码模块，删除最后已无调用的 Yolo26Runtime。保留 X5/S 旋转 NMS、角度和裁剪差异；双语 README 新增实际执行示例并披露非等比缩放近似及整数几何修正。408 项主机测试、128 本地链接和 16 双语示例通过，见[旋转框记录](../../releases/unified-migration/2026-09-28-yolo26-obb-review.md)。板端/真实 SDK/整体独立评审 not-run，原生、YOLOE、B10/B11/H8 及 H0–H9 继续。

2026-09-28：Ultralytics C++ 分类修正固定类别轴和连续读取假设，严格绑定 1000 类浮点输出与物理步长，新增异常路径资源所有者；标签和数学移出主入口。双语文档明确 C++ letterbox 与 Python 默认值差异。7 个 C++ 主机测试在 ASan/UBSan 下通过，并修复由 sanitizer 发现的旧 DFL 测试夹具越界；408 项主机回归及文档检查通过，见[C++ 分类记录](../../releases/unified-migration/2026-09-28-yolo-cpp-classification-review.md)。真实 SDK 编译/板测/精度 not-run；其他原生任务、YOLOE、B10/B11/H8 与整体独立评审继续，H0–H9 未关闭。

2026-09-28：C++ 姿态/分割取消固定输出索引与连续内存假设，共用严格 NHWC 角色绑定、步长读取、有限值检查与输出资源所有者；DFL 数学合入现有公共解码，保存失败明确报错。双语 README 修正分割三联图空间、NMS 和 Python ROI mask 的区别。10 个 sanitizer 原生主机测试、408 项主机回归及文档检查通过，见[绑定记录](../../releases/unified-migration/2026-09-28-yolo-cpp-heads-review.md)。完整原生阶段拆分、其他审计及 H0–H9 继续；真实 SDK 构建/板测/精度和独立整体评审 not-run。

2026-09-28：YOLOE 源能力审计完成，补回 S26 30 个原始文件并固定发布清单/sidecar 证据及 10 个 HBM 预期哈希；源回归 4 项、publisher 121 项及生成检查通过。S11 中间图浮点信息不能代替最终 HBM 精度，S26 已发布制品均声明量化，不能直接套浮点-only Ultralytics 入口；优先保留反量化输出节点的转换路线，未构建/发布的浮点制品不得声称可运行。见[YOLOE 核定](../../releases/unified-migration/2026-09-28-yoloe-source-review.md)。Canonical YOLOE 实现与完整双语文档仍 pending，B9/H5/H8 及 H0–H9 均未关闭。

2026-09-28：YOLOE-26 PF 数值模块已分离，保留 round/114、Top-K 无 NMS 和 logits 掩码顺序；9 项固定源/边界测试及 417 项主机回归通过，共享说明补齐中英协议和边界。见[内部模块记录](../../releases/unified-migration/2026-09-28-yoloe26-kernels-review.md)。尚无 canonical YOLOE 入口，浮点制品/转换与完整客户文档仍 pending；板端/SDK/OE/独立评审 not-run，H0–H9 保持开放。

2026-09-28：Canonical YOLOE Python 入口、14 制品精确选择、显式下载及根/model/runtime/test_data 八份双语 README 已建立。保留 X5 整图 mask、S11 ROI/morph 默认值和 E26 Top-K 协议；仅明确指定、哈希绑定的独立浮点 S 转换文件可进入 metadata 验证。见[Python 记录](../../releases/unified-migration/2026-09-28-yoloe-python-review.md)。YOLOE 已加入规范检查范围；转换/评测/C++ 及对应完整 README 尚待收编，板端/真实 SDK/OE/独立评审 not-run。B9 和 H0–H9 继续开放。

2026-09-28：YOLOE 统一转换准备已提供 ONNX/词表校验、14 种目标配置、X5 raw 与 S NPY 校准、可选编译完整日志及真实状态分离；两份完整转换 README 与根/model/runtime 导航同步。445 项主机测试、双语准备命令、45-sample 规范检查通过，见[转换准备记录](../../releases/unified-migration/2026-09-28-yoloe-conversion-review.md)。源导出器统一收编、evaluator、C++ 与对应客户文档仍需完成；真实权重/OE/板端/数据集/独立评审 not-run，H0–H9 保持开放。

2026-09-28：YOLOE 统一导出器已收编 E11/E26，八种真实权重均完成主机 ONNX 导出与全输出对照，14 组真实图转换准备通过；修正 11l 第二 attention 配置，保留 E11m 优化执行差异与 E26 m/l/x 非精确平局的排序差异证据。451 项主机测试与 45-sample 规范检查通过，详见[导出记录](../../releases/unified-migration/2026-09-28-yoloe-export-review.md)。未宣称优化执行、数据集精度、OE 或板端通过；evaluator、C++、完整剩余文档、整体独立评审与 H0–H9 继续。

2026-09-28：YOLOE 统一 evaluator 和完整双语说明已实现，保留历史性能表及测量条件，新增严格类别映射、框/掩码 COCO 计分、输入身份和失败留证。14 组真实 ONNX 主机预测、461 项主机测试及 45-sample 规范检查通过；初次文档链接失败和修正均留存。发现 E26 当前 PT 文件哈希与历史 sidecar 不同，未宣称同权重复现或数据集 AP。见[评估记录](../../releases/unified-migration/2026-09-28-yoloe-evaluation-review.md)。原生 C++、剩余迁移与全分支独立验收继续，板端/SDK/OE not-run，H0–H9 未关闭。

2026-09-28：YOLOE 原生浮点角色绑定及 E26 候选解码已分离，复用 Ultralytics 物理张量读取；两个 C++ 主机测试在 ASan/UBSan 下通过，真实 E26n 单/多标签候选与 Python 类别顺序完全一致，框/分数/系数本次差值为 0。原生双语说明明确当前只提供内部模块；模板初次失败及修正留证。见[原生模块记录](../../releases/unified-migration/2026-09-28-yoloe-cpp-kernels-review.md)。E11、掩码、SDK 资源、身份门禁与完整 C++ 入口仍待集成，H0–H9 和整体评审继续。

2026-09-28：YOLOE C++ E11 DFL/NMS 候选模块已加入，复用 Ultralytics DFL/sigmoid 并与 E26 共用结果类型。保留原生 IoU 等值边界，明确披露与 Python 差异和确定平局排序。三个 ASan/UBSan 原生测试、真实 E11 s/m/l 及 E26 对照、29 项 YOLOE 回归和规范/文档检查通过。见[E11 原生记录](../../releases/unified-migration/2026-09-28-yoloe-cpp-e11-review.md)。掩码、几何、SDK 管理和完整原生入口继续；全部 H0–H9 范围及独立评审保持开放，板端 not-run。

2026-09-28：YOLOE 原生 E11/E26 图片几何及 E26 ROI 掩码恢复已实现；任务目录实际构建 OpenCV 4.14.0，五个 C++ 测试通过，真实 E26n 的 213 个 ROI 形状/像素与 Python 完全一致。修复初次发现的退化 ROI 0×58 被丢成 0×0 问题并保留失败证据；双语 README 实际构建/测试命令已执行。见[几何/掩码记录](../../releases/unified-migration/2026-09-28-yoloe-cpp-masks-review.md)。E11 掩码、NV12、SDK 生命周期和完整原生入口继续，H0–H9/整体独立评审均开放；板端/真实 SDK/OE not-run。

2026-09-28：YOLOE 原生 E11 ROI 掩码已补齐，38 个真实候选在开/关形态学下均与 Python 像素一致，E26 213 掩码回归一致。修复共用 Python DFL ROI 的零面积框补成 1 像素及 Lanczos 过冲为 2 两项契约缺陷；源对照明确保留前景、归一二值表示，失败证据留存。463 项 Python 测试、五个原生测试及规范/文档检查通过，见[E11 掩码记录](../../releases/unified-migration/2026-09-28-yoloe-cpp-e11-masks-review.md)。NV12/SDK 管理、身份门禁、完整入口、其余迁移与全分支评审继续；H0–H9 未关闭。

2026-09-28：YOLOE C++ 三阶段与 predict 已组成可独立构建的库，复用 Ultralytics NV12 拆平面工具，显式持有输入、几何和输出并拒绝跨实例阶段混用。两种 README 构建方式的六个原生测试均通过；六组完整 Y/UV 字节与 Python 一致，双语 API 示例实际编译，29 项 YOLOE 回归与 45 sample 规范检查通过。见[原生阶段库记录](../../releases/unified-migration/2026-09-28-yoloe-cpp-stages-review.md)。SDK 后端/身份门禁/CLI、其余迁移及全分支独立评审继续，H0–H9 未关闭；板端 not-run。

2026-09-28：为接入 YOLOE 原生后端，修正 Ultralytics 共用 NV12 动态容量与 SDK 任务失败释放问题；两项旧代码缺陷已复现并留证。新增长度精确的 Y/UV 直接上传，12 个原生测试及 463 项 Python 回归通过，双语 README 同步输入约束和主机验证边界。见[共用原生 I/O 记录](../../releases/unified-migration/2026-09-28-yolo-native-io-review.md)。YOLOE SDK 后端/CLI 及 H0–H9 其余事项继续，真实 SDK/板端与独立整体评审未执行。

2026-09-28：YOLOE 原生 SDK 适配器已接入共用模型/输入/输出/任务管理，十个浮点输出按语义角色绑定，不新增手动反量化；必需预检回调先于 SDK 执行。两种文档构建方式均通过八项 YOLOE 原生测试，共用层 12 项及 463 项 Python 回归通过；双语完整 API 示例编译通过。见[SDK 适配器记录](../../releases/unified-migration/2026-09-28-yoloe-sdk-runner-review.md)。统一预检策略、CLI 和完整可执行入口仍待完成；真实 SDK/板端/OE 未验证，H0–H9 与独立整体评审保持开放。

2026-09-28：YOLOE 原生预检工厂已提供本机身份、预期模型摘要及固定词表核验；共用 SHA 从 YOLOv5 提取并委托复用，修复目录被当作可哈希文件的边界问题。平台注册表与 Python 识别优先级逐项一致；两种构建方式各九项原生测试、共用层 12 项及 544 项 Python 回归通过，双语 SDK 示例实际使用预检工厂并编译验证。见[原生预检记录](../../releases/unified-migration/2026-09-28-yoloe-native-preflight-review.md)。统一发布制品选择/CLI/输出仍待完成；H0–H9 继续，真实 SDK、板端、OE 与全分支独立评审未执行。

2026-09-28：YOLOE 原生发布选择、可执行入口及结果留证已实现，完整双语 README 同步模型前提、全部参数、构建/运行、输出及失败判据。557 项 Python 回归、两种文档构建各 11 项原生测试、共用层 12 项和 18 个主机 CLI 检查通过；完整入口夹具明确标为 host-fixture，不能充当 SDK/板测证据。见[原生入口记录](../../releases/unified-migration/2026-09-28-yoloe-native-cli-review.md)。S 浮点制品、真实 SDK/OE/板端与独立整体验收未验证；继续 B10/B11/H8 及 H0–H9 剩余工作。

2026-09-28：B10 四样例 132 个源文件已与固定提交逐字节核验，HIMLoco 21 份输入完备；复现 ASR 非 CTC 折叠行为及 Paraformer 零触发 CIF 越界，确认 KWS 采样时长/目标支持文档矛盾。见[B10 源审计](../../releases/unified-migration/2026-09-28-b10-source-review.md)。统一实现、完整双语文档与各项验收仍待进行；H0–H9 保持开放。

2026-09-28：B10 KWS 统一 Python 三阶段、共享 SDK runner、显式下载和离线指标已实现，六层双语 README 补齐；真实 PaddleAudio 三类输入与源前端逐值一致，13 项 KWS / 432 项相关 Python 回归、121 项 publisher 及 46-sample 规范检查通过。见[KWS 记录](../../releases/unified-migration/2026-09-28-b10-kws-review.md)。源未提供权重/转换配方的缺口保留；真实 SDK/板端分数与独立验收未验证。继续 ASR、Paraformer、HIMLoco、B11/H8 及 H0–H9 剩余事项。

2026-09-28：ASR 完整 Python 流程、共享文本指标与原生 CTC/归一化核心已实现，七层双语 README 已补齐并执行示例。15 项 ASR 测试、七块真实音频源前端逐值对照及 C++ sanitizer 核心测试通过；CTC 修正与 legacy 对照明确区分，末块及失败留证边界写入文档。见[ASR 阶段记录](../../releases/unified-migration/2026-09-28-b10-asr-core-review.md)。原生音频/SDK/CLI 仍待集成，不能关闭 ASR；Paraformer/HIMLoco、B11/H8 与 H0–H9 全范围继续，板端/真实 SDK/OE/语料评测和独立整体评审未执行。

2026-09-28：ASR 原生音频读取/重采样/前处理与三阶段任务已实现，真实 libsndfile/libsamplerate 七块源音频对照通过；三个原生测试在 Release+ASan/UBSan 下通过，双语 C++ 完整示例实际编译运行。文档明确独立窗口与 Python Fourier 差异，见[原生音频记录](../../releases/unified-migration/2026-09-28-b10-asr-native-audio-review.md)。SDK 适配/原生 CLI 仍未完成，ASR 与 H0–H9 不关闭；其余样例迁移及独立整体验收继续。

2026-09-28：ASR UCP SDK 适配和身份/模型/词表预检已实现，复用共用资源与同步任务管理；失败注入复现并修复共用输出所有者在错误伴随非空分配时的泄漏。五项 ASR 原生、12 项共用原生、11 项 YOLOE 原生及文档示例通过，见[SDK 记录](../../releases/unified-migration/2026-09-28-b10-asr-sdk-review.md)。真实 SDK 开启配置在本机按预期因缺头文件拒绝，未宣称 ABI/模型通过；原生 CLI/词表加载/结果报告继续，ASR 与 H0–H9 保持开放。

2026-09-28：ASR 原生 CLI、固定词表 JSON 解析、完整文件结果/失败报告及显式启动器已集成，七层双语文档对齐实际入口。七项原生测试、16 项原生/启动器用例、21 项 ASR Python 测试与双语示例通过；主机替身始终标记 host-fixture，公开启动器拒绝将其当作 SDK 成功。见[完整入口记录](../../releases/unified-migration/2026-09-28-b10-asr-native-cli-review.md)。ASR 实现已完成主机集成，真实 SDK/ABI/模型与独立整体验收未核定；继续 Paraformer/HIMLoco、B11/H8 与 H0–H9 剩余工作。

2026-09-28：Paraformer CPU CIF 已独立提取并修复无 token 时的空数组访问；7 项主机测试、24 组固定源数值对照与双语 README 示例通过。见[记录](../../releases/unified-migration/2026-09-28-b10-paraformer-cif-review.md)。尚未接入完整运行/转换流程，台账 Refactor 保持 pending；真实前端、三模型运行、C++、转换/评测及完整各层 README 继续，B10/H0–H9 未关闭。

2026-09-28：Paraformer 应用层三模型编排与独立文本解码已实现；空 CIF 输出显式跳过 decoder，未混入任务 forward。13 项主机测试、20 组固定源文本对照和 6 个双语文档命令通过；8,404 项发布词表已实测并留存摘要。见[流程记录](../../releases/unified-migration/2026-09-28-b10-paraformer-pipeline-review.md)。完整 SDK/前端/转换/原生流程仍未接入；H0–H9 和独立验收保持开放。

2026-09-28：Paraformer S100 三模型精确选择、共享 SDK runner 绑定及实际调度委托已实现，错误组合先于 SDK 创建拒绝；新增六文件模型包准备、源前端文件固定摘要与双语模型说明。23 项 Sample、156 项共享回归及双语示例通过，初次共享回归缺下载入口的失败与修复留存。见[记录](../../releases/unified-migration/2026-09-28-b10-paraformer-binding-review.md)。真实前端、完整音频/原生/转换评测继续；真实 SDK、HBM 推理、板测未执行，H0–H9 未关闭。

2026-09-28：Paraformer 真实 FunASR 前端已迁移，7 组真实输入与源特征逐字节一致；修复全局 CPU 随机状态污染并验证异常恢复，保留超长输入前 400 帧并显式报告截断。26 项 Sample/156 项共享测试通过，Python/model/test_data 双语说明和源音频/清单同步。见[真实前端记录](../../releases/unified-migration/2026-09-28-b10-paraformer-frontend-review.md)。完整音频 CLI、C++、转换/评测和剩余文档继续；真实 SDK/板端与整体验收未执行，H0–H9 保持开放。

2026-09-28：Paraformer Python 完整音频/清单 CLI 和结果留证已接入，真实主机 10 用例通过；生成特征不再改写用户清单，缺音频/重复 ID/已有输出显式拒绝，33 Sample/156 共享测试通过。根与运行双语 README 补齐完整路径、全部参数、结果/失败语义及板测边界；5 组不同 CLI 文档命令实际执行。见[CLI 记录](../../releases/unified-migration/2026-09-28-b10-paraformer-cli-review.md)。C++、转换/评测与剩余文档继续，B10/H0–H9 不关闭。

2026-09-28：Paraformer C++ CPU CIF/文本模块与三模型应用编排已实现，27 组新原生/旧原生/统一 Python 字节对照及 20 组文本对照通过；双语 README 的 Release+ASan/UBSan 构建、两项原生测试及完整 API 示例实际执行，33 项 Python 回归通过。见[原生模块记录](../../releases/unified-migration/2026-09-28-b10-paraformer-native-core-review.md)。原生 SDK/清单加载/CLI、转换评测仍待迁移；真实 SDK/板端未验证，B10/H0–H9 及整体独立验收继续开放。

2026-09-28：Paraformer 原生 S100 SDK 适配器已按三模型物理名称／形状／类型／字节步长绑定，复用共用资源管理；首次集成暴露共享调用只接受一／两输入，已提取通用多输入调用并保留原图像约束。三项原生测试和双语构建/API 示例通过；真实 SDK 配置因本机缺依赖明确拒绝。见[SDK 记录](../../releases/unified-migration/2026-09-28-b10-paraformer-sdk-review.md)。具体身份/制品预检、清单/CLI、转换评测及全部剩余迁移继续，B10/H0–H9 未关闭。

2026-09-28：Paraformer 原生生产预检工厂已校验完整三模型组、实际本机身份及固定词表，任何阶段文件错误先于 SDK 加载拒绝；四项原生测试、双语实际构建/API 编译及真实本机身份拒绝检查通过。见[预检记录](../../releases/unified-migration/2026-09-28-b10-paraformer-preflight-review.md)。原生清单/CLI、转换评测及其余 H0–H9 工作继续；真实 SDK/板端未验证，整体验收不提前关闭。

2026-09-28：Paraformer 原生准备清单/NPY 读取库已接入真实 Python 前端产物，按同一字节缓冲区核验摘要；32 组实际 NumPy/异常用例和五项原生测试通过，两份真实音频特征数组与 Python 逐字节一致。双语三组 shell 命令与两个完整 API 示例已执行/编译，见[读取器记录](../../releases/unified-migration/2026-09-28-b10-paraformer-feature-io-review.md)。完整原生 CLI/结果留证、转换评测及其余 H0–H9 工作继续，未做 SDK/板端验证，整体目标保持开放。

2026-09-28：Paraformer 原生完整 CLI/三模型编排/成功失败报告及公开启动器已接通，实际本机身份门禁、测试替身拒绝和结果严格核验均已实现。14 组完整应用主机用例、6 项原生测试、39 项 Python 测试通过；5 组双语主机命令、真实 FunASR 预处理及两个 API 示例已执行/编译。见[原生入口记录](../../releases/unified-migration/2026-09-28-b10-paraformer-native-cli-review.md)。真实 SDK/模型/板端未验证；Paraformer 转换评测、HIMLoco、B11/H8 与完整 H0–H9 工作继续，独立整体验收不提前关闭。

2026-09-28：Paraformer 转换图工具已统一，拒绝残缺图和未经证明的 Range 固化，保留共享常量；两项源缺陷已实际复现。48 项 Sample 测试（含 9 项真实 ONNX/ORT 小图检查）、双语 API 示例和 47-sample 门禁通过。见[图处理记录](../../releases/unified-migration/2026-09-28-b10-paraformer-graph-ops-review.md)。完整权重导出/切图/校准/OE 编排、专用评测与剩余迁移继续，Paraformer 与 H0–H9 未关闭，板端 not-run。

2026-09-28：Paraformer 真实权重三阶段 FP32 导出已实现，直接复用上游阶段并保留固定 400/100 部署语义，不再靠全图内部编号切图及单次探测固化。16 项真实导出检查、56 项无跳过 Sample 测试和双语 API/链接检查通过；两条完整 Torch/ORT 示例 token/text 一致，但参考转录存在识别错误。历史固定宽度 ONNX 与新图在 0/1/17/100 token 对照中完全一致；任意随机隐藏向量的 Torch/ORT 大差异及首次失败均披露，不能扩大为全输入域数值等价。见[导出记录](../../releases/unified-migration/2026-09-28-b10-paraformer-export-review.md)。Ruling: 用显式阶段导出替代内部名称切图，成本是必须分别证明固定部署语义而不能声称原始变长模型等价。校准/OE 编排、evaluator 与剩余 H0–H9 继续，板端未运行，整体验收未关闭。

2026-09-28：Paraformer 真实音频校准、三份 nash-e 配置及显式编译编排已实现，复用统一前端和无屏蔽 CPU CIF，拒绝空集、非法音频、被修改/追加的校准输入，保留失败日志。65 项 Sample 测试无跳过通过；12 份真实中间数组与旧校准脚本逐字节一致，2 份 speech 与已有前端证据一致；实际 8 kHz 输入拒绝和 OE 缺失门禁通过。双语转换/root/Python 说明同步，见[校准记录](../../releases/unified-migration/2026-09-28-b10-paraformer-calibration-review.md)。本机无 hb_compile/Docker，真实 OE/SDK/板端未执行；下一步专用 evaluator，Paraformer 与完整 H0–H9 不关闭。

2026-09-28：Paraformer 专用 evaluator 已实现，复用三阶段流程、CIF、词表解码和共享 CER；75 项 Sample 测试通过。两个真实特征的 FP32 转写／编辑距离与源脚本一致，4/28 = 14.2857% CER，不作为数据集精度；HMCT 仅适配测试，实际 OE/HMCT/板端未执行。根、Python、转换和 evaluator 双语说明已同步。见[评测记录](../../releases/unified-migration/2026-09-28-b10-paraformer-evaluator-review.md)。整套 Sample 验收及 H0–H9 继续。

2026-09-28：Paraformer 14 份双语 README 做整套规范核对，修复直接检查的 78 项问题，并统一 evaluator／conversion 输出目录写法；保留 72 个示例，132 个本地文件链接通过。直接 Sample 检查 0 violations / 1 CLI policy skip / 0 exemptions。人工审查仍发现多阶段公开 pre/forward/post 接口及错误阶段归属未完成，下一步修复代码与 API 示例，暂不提升台账状态。见[文档与 API 缺口记录](../../releases/unified-migration/2026-09-28-b10-paraformer-readme-review.md)。

2026-09-28：Paraformer 三模型公开 pre_process/forward/post_process 已落地，CPU CIF 仍显式编排，补齐 StageError 阶段／操作归属和原始异常链。82 项测试通过；两条真实 FP32 的全部文本、token 和 CER 与原记录相同；双语显式阶段示例实际执行。台账提升至 in-progress 纳入迁移检查，整体验收／独立复审仍不关闭。见[阶段整改记录](../../releases/unified-migration/2026-09-28-b10-paraformer-stages-review.md)。

2026-09-28：开始 HIMLoco 迁移，52 源文件与 X5 固定提交逐字节一致；离线 Python 四阶段核心及双语 API 文档完成初版，6 项测试、21 条真实源观测前处理对照、双语可执行示例通过。SDK/CLI/C++/评测及其余 README 继续，台账仍 pending，不以核心片段宣称完整迁移。量化方案仅按源文档重构，不重新验证。见[源清点与核心记录](../../releases/unified-migration/2026-09-28-b10-himloco-core-review.md)。

2026-09-28：HIMLoco 补齐准确模型绑定、共享 SDK runner、离线 CLI 与显式下载，21 源输入及清单原样迁入；根/model/Python/test_data 双语说明已提供。15 Sample 测试及6项真实无 SDK CLI 检查通过，直接 Sample 门禁零违规，台账进入 in-progress。C++/评测/转换文档仍待完成，量化方案不实跑验证。见[Python 入口记录](../../releases/unified-migration/2026-09-28-b10-himloco-cli-review.md)。

2026-09-28：HIMLoco conversion/evaluator 七个源脚本原样迁入，四份双语 README 补齐根目录命令、实际参数、输出、历史来源及制品身份边界；根入口同步。15 主机测试与 Sample 文档契约零违规，未执行量化方案。C++ 与整套重构验收继续，见[文档迁移记录](../../releases/unified-migration/2026-09-28-b10-himloco-docs-review.md)。

2026-09-28：HIMLoco C++ 纯四阶段核心迁入，SDK 与文件职责分离，返回值持有数据和逐次耗时；新增原生主机测试先失败后通过，21 条源输入逐字节检查及 ASan/UBSan 通过。16 项 Sample 测试、文档契约零违规；C++ 双语说明及根入口同步。SDK／CLI 及整套独立验收仍待完成，见[原生核心记录](../../releases/unified-migration/2026-09-28-b10-himloco-cpp-core-review.md)。

2026-09-28：HIMLoco 原生 SDK 适配器实现，复用共享板型／SHA 校验、补齐张量容量与 RAII 失败清理；主机替身与生产 preflight 拒绝路径分别测试，ASan/UBSan 和19项 Sample 测试通过，双语 C++/根说明同步。真实 SDK 编译/运行未测；CLI、构建/启动及文件报告继续。见[SDK 记录](../../releases/unified-migration/2026-09-28-b10-himloco-sdk-review.md)。

2026-09-28：HIMLoco 原生 CLI/文件报告/CMake/启动器实现，保留源索引与 float 输出，独占结果写入、清单摘要校验、失败部分报告；启动器复用制品与板型门禁，预览不构建/下载。23 项主机测试、CMake/CTest、迁移门禁通过，完整双语原生/根指南同步。真实 SDK/板端未测，整套质量复核与其余 H0–H9 仍继续，见[原生入口记录](../../releases/unified-migration/2026-09-28-b10-himloco-native-cli-review.md)。

2026-09-28：HIMLoco 按源能力完成作者审计，Mapping/Refactor/Docs 记 done、Host 仅覆盖已留证23项/原生主机检查；Review=not-run、Closed=no不变。补齐根/索引遗漏的 Paraformer 与 HIMLoco，双语索引与49个实际Sample根完全一致，14份HIMLoco README陈旧说明已修正；相对链接核对见[作者审计](../../releases/unified-migration/2026-09-28-b10-himloco-author-audit.md)。不关闭H6/H0–H9，后续转B11并保留全仓独立评审。

2026-09-28：B11 固定源清点：Gemma67文件原样；MiniCPM62文件中13缺失、9份README落后，已恢复为S pin原字节，补回S100/S100P完整PPL失败结论及历史结果，不执行量化/评测。ACT/Pi0两个不同gitlink与URL核定，尚未初始化或搬迁。B11仅Mapping记done，Refactor/Docs仍pending，见[源清点](../../releases/unified-migration/2026-09-28-b11-source-audit.md)。

2026-09-28：VLA 两个gitlink按原SHA迁至samples/vla，.gitmodules同步并实际初始化，完整上游代码无改动；双语总览/ACT/Pi0指南及旧路径导航完成，保留板型/LeRobot差异、源历史数字和控制边界。Refactor对固定外部源码记not-applicable，父仓库集成由2项专用检查覆盖，非豁免原生sample规则；独立评审/板端仍未执行。见[VLA集成记录](../../releases/unified-migration/2026-09-28-b11-vla-integration.md)。Gemma/MiniCPM与H0–H9余项继续。

2026-09-28：Gemma67源文件迁入统一目录，58文件原字节保留；五入口显式准备/构建/启动拆分及三层双语说明完成初版。7项主机编排测试、42本地文件链接通过。新纳入迁移门禁后50samples/70violations（均Gemma缺标准README锚点）/51skips/0exemptions，未掩盖未完成项；核心职责、制品准备、全层文档继续，见[启动迁移记录](../../releases/unified-migration/2026-09-28-b11-gemma-launcher-review.md)。不执行量化或板测，H0–H9保持开放。

2026-09-28：Gemma14份双语README组织与导航补齐，保留原参数/图示/历史值，澄清模型目标及哈希来源、golden判据并修正命令参数与工作目录；21份转换脚本/完整教程保持原字节。迁移门禁50samples/0violations/51skips/0exemptions，四项主机预览通过；见[文档记录](../../releases/unified-migration/2026-09-28-b11-gemma-readme-review.md)。核心职责/显式模型准备及全分支验收继续，不执行量化/板测，不关闭H0–H9。

2026-09-28：Gemma Vision 读图移至应用层，纯前处理/显式runner/后处理与Predict组合接入main/demo；4张源图片与固定源C++前处理逐字节一致，Release及ASan/UBSan各2项CTest通过，7项启动器回归通过，门禁50/0/51skips/0exemptions。双语运行README补完整接口示例。SDK张量/资源生命周期、Text/KV、模型准备继续，见[Vision职责记录](../../releases/unified-migration/2026-09-28-b11-gemma-vision-review.md)；不执行量化/板测，H0–H9开放。

2026-09-28：Gemma复现并修复buffer/task/packed-model三类失败泄漏；全量和选择性推理共用task所有权，保留S600编译核数与V3入口。S100/S600主机72场景及ASan/UBSan通过，Gemma11项单测通过，门禁50/0/51skips/0exemptions。双语失败说明同步，见[资源记录](../../releases/unified-migration/2026-09-28-b11-gemma-resource-review.md)。Vision严格张量/stride、TextKV、模型准备与全范围验收继续；量化/板测不执行，H0–H9开放。

2026-09-28：Gemma Vision严格F16输入/F16-F32输出、语义矩阵、stride/容量校验与带padding读写接入生产engine；移除未知类型float回退，数值存储与诊断移出推理文件。修复前6a71bdaa末轴padding错误已复现留证；94资源/传输场景及ASan/UBSan、3项原生CTest、12项Sample单测通过，门禁50/0/51skips/0exemptions。双语API更新，见[张量记录](../../releases/unified-migration/2026-09-28-b11-gemma-tensor-review.md)。Text/KV、显式模型准备、MiniCPM及H0–H9继续，不做量化/板测。

2026-09-28：Gemma KV修复K长度清零V导致越界、失败重分配破坏旧别名、尾部padding参与滚动等问题；事务分配、分层Reset、追加前全层校验及位置提交、前缀保留契约完成。9项原生ASan/UBSan CTest与13项Sample单测通过，门禁50/0/51skips/0exemptions；双语状态/所有权说明同步，见[KV记录](../../releases/unified-migration/2026-09-28-b11-gemma-kv-review.md)。Text SDK/状态职责、模型准备及H0–H9继续，不运行量化或板测。

2026-09-28：Gemma Text构造泄漏在2c4da3fd原源码复现；张量所有权拆为move-only ModelIo，逐槽追踪KV借用，构造失败/正常退出先释放子图再释放packed model。301个获取失败点、6类非法描述符及所有权检查通过，11项ASan/UBSan CTest、14项Sample单测与50-sample门禁零违规；双语C++指南同步，见[Text所有权记录](../../releases/unified-migration/2026-09-28-b11-gemma-text-ownership-review.md)。完整Text张量契约/职责、模型准备与H0–H9继续，量化及板测不执行。

2026-09-28：Gemma模型准备取消隐式S100P回退，要求显式GEMMA4_SOC；新增无网络/无写入预览与参数错误处理，拒绝发布空下载。原脚本5项失败经假wget复现，整改后6项准备测试/20项Sample测试通过，50-sample门禁零违规；根/model/C++六份README同步完整参数、独立目录、S100与断点续传限制。见[准备记录](../../releases/unified-migration/2026-09-28-b11-gemma-preparation-review.md)。未下载模型或验证量化；Text/第三方/MiniCPM/H0–H9继续。

2026-09-28：Gemma第三方准备改为显式稳定Rust/Git前置检查、固定提交与递归子模块检查、保留现有修改及临时目录准备；取消自动Rust安装与删除未知目录，Cargo.lock补丁改为可移植写法。8项本地Git/真实gitlink夹具测试、28项Sample测试与50-sample门禁通过，四份双语README同步。见[依赖准备记录](../../releases/unified-migration/2026-09-28-b11-gemma-dependencies-review.md)。未远程下载依赖/模型、编译量化或板测；Text/MiniCPM/H0–H9继续。

2026-09-28：MiniCPM62源文件核对固定pin后迁入，S100/S100P旧SDK与S600新SDK保持独立实现；统一启动编排拆开模型准备/构建/运行，保留原生参数及legacy超时，主机预览无副作用、实际运行先板型检查。8项编排测试通过，八份双语入口指南同步历史证据边界；纳入门禁后51samples/70violations（MiniCPM缺README章节标记）/51skips/0exemptions，如实保持未完成。见[入口记录](../../releases/unified-migration/2026-09-28-b11-minicpm-entry-review.md)。核心/全层README/清单与H0–H9继续，不执行量化或板测。

2026-09-28：MiniCPM20份双语README覆盖全部层级，根支持矩阵/目录导航、模型参数与校验、C++构建/运行/会话/结果、转换阶段、评测判据与test_data中文指南已补齐；115本地链接存在、8份转换/评估文档原命令行及顺序不变，两项无板预览通过。51samples/0violations/51skips/0exemptions；未执行量化或板测。见[文档记录](../../releases/unified-migration/2026-09-28-b11-minicpm-readme-review.md)。核心与最终API/清单/独立整体验收仍待完成，不以章节门禁宣称整体完成。

2026-09-28：MiniCPM原生核心深度重构作者完成：R1（legacy单次生命周期，init-after-predict拒绝）、R2（S600指标有限/非负校验并保留零decode_tps）、R3（临时runtime配置RAII化，成功/错误/异常路径均清理）按reviewer基线关闭；runtime/cpp拆出runtime_config.cc（配置/文件IO），公开pre_process/infer/post_process阶段由Generate串联；runtime/legacy拆出chat_template.cc，prepare_request前处理+RequestOutcome状态记录，main.cc输出RESULT行。贪心采样、request_id/会话字段、模板界限、RESULT格式与退出码全部保留；S100/S100P PPL+27.83%与2/6文本匹配、S600+1.60%历史记录原样单列。新增tests/native五个C++驱动（xlm/oellm替身均注明非厂商SDK）与3项真实legacy CLI测试，16项单测通过；51samples/0violations/51skips/0exemptions；模型清单URL/摘要逐字节未动，根/索引51计数与MiniCPM双语链接补齐，Gemma改为"准备完成、Text核心待做"。未运行SDK/模型/板测；作者完成，独立评审待Codex。见[核心重构记录](../../releases/unified-migration/2026-09-28-b11-minicpm-core-refactor.md)。Gemma Text、H8、H9与全分支独立评审继续。

2026-09-28：MiniCPM核心包独立评审changes-required整改完成：CORE-R2先行修复——models.yaml note中未引用的"path: "冒号致整个S清单ScannerError并阻塞并行PointNet测试，改为合法plain scalar后yaml.safe_load与MiniCPM资产resolve通过，三档案URL/SHA-256逐字节未动；CORE-R1将PreparedRequest改为自重绑定carrier（copy/move构造与赋值均bind()，moved-from同样重绑），prepare_request签名不变，新增-fno-elide-constructors所有权驱动覆盖非NRVO返回/短长字符串/四种特殊成员，reviewer prepared_ownership.cpp原样重跑owned=1/1/1、rc=0；CORE-R3把legacy核心控制台IO移除，MiniCPM5Config::text_sink注入流式（默认空=无console可用），main注入stdout sink，END/ERROR文本抑制、状态映射与single-use保持，消费方异常在C回调内兜底记入RequestOutcome::stream_error并停止流式；公开pre_process自身强制空prompt与1–4096边界；六份README banner改为"启动编排+原生核心重构（SDK替身测试）"，双语runtime指南补完整native库示例与carrier/sink生命周期。18项单测（8编排+7驱动+3 CLI）rc=0；51samples/0violations/51skips/0exemptions rc=0；git diff --check rc=0。reviewer报告与证据未改，历史R1–R5证据保留；未commit。见[整改记录](../../releases/unified-migration/2026-09-28-b11-minicpm-core-refactor.md)。独立评审保持changes-required/pending，不自关H7/B11；Gemma Text、H8、H9继续。

2026-09-28：MiniCPM核心包CORE-R4测试隔离整改完成：reviewer安全复现证实s600_metrics/s600_stages在全局temp_directory_path()/model写fixture并remove_all（哨兵被实际metrics二进制删除、rc=0），s600_config_cleanup用固定场景名。新增tests/native/scratch_dir.hpp：mkdtemp原子唯一目录、RAII仅清理自持目录、TMPDIR恢复含原本unset状态（unsetenv）；三个S600驱动重写到该助手，legacy驱动维持自有mkstemp文件；s600_config_cleanup新增进程内TMPDIR状态前后一致断言。test_native_core.py新增两项回归：敌意TMPDIR下既有model/sentinel.txt在全部驱动直连运行后保留、五个S600二进制实例共享同一TMPDIR并发rc全0；RED敏感性用一次性程序在全新隔离目录复现旧模式确认哨兵检查可捕获，reviewer原始复现保留未改，未触碰任何用户既有临时路径。20项单测（18+2）rc=0；51samples/0violations/51skips/0exemptions rc=0；git diff --check rc=0；yaml与reviewer driver复跑rc=0。CORE-R1–R3与API/doc修复保留；reviewer文件未改。见[整改记录](../../releases/unified-migration/2026-09-28-b11-minicpm-core-refactor.md)。独立评审保持changes-required/pending，不自关H7/B11；本包停止交Codex验收，Gemma Text、H8、H9继续。

2026-09-28 Codex独立验收更新：MiniCPM核心包CORE-R1–R4与CORE-N1已按实际生产源/SDK替身复查关闭；20项主机测试、原非NRVO所有权反例、隔离哨兵/并发/unset TMPDIR及四份README实际代码块语法编译通过。仅最后四份README发生后续改动，核心快照未变。见[独立复审](../../releases/unified-migration/2026-09-28-minicpm-core-independent-review.md)。此为host范围接受，不覆盖真实SDK/板端，不关闭Gemma/B11/H7整体。

同日B8范围验收已形成独立报告：[UNet/PP-LiteSeg](../../releases/unified-migration/2026-09-28-b8-segmentation-independent-review.md)、[Depth Anything](../../releases/unified-migration/2026-09-28-depth-anything-independent-review.md)、[LaneNet](../../releases/unified-migration/2026-09-28-lanenet-independent-review.md)、[UNetMobileNet/DiffusionDrive](../../releases/unified-migration/2026-09-28-b8-mobile-planning-independent-review.md)、[YOLO26 Depth](../../releases/unified-migration/2026-09-28-yolo26-depth-independent-review.md)。PointNet R1已关闭但完整审阅新发现R2 int32比较精度问题，已交Claude整改；B8/H4尚未整批关闭。KWS运行时主机范围已审阅，根README算法说明深度待补齐。README源深度恢复由Claude分三包实施：九个B1/B2、十二个后续分类、Ultralytics/FCOS/ByteTrack及KWS；尚不能凭章节检查关闭H1。所有H0–H9原范围保持，板端和真实量化流程按用户决定不执行。

2026-09-28独立评审继续：HIMLoco离线运行时23项通过，Paraformer运行时组合46专项+6原生通过，KWS运行时13项通过但根算法说明需补；ASR21项+5原生通过仍被独立ASR-R1反例阻断（int32经float32形成blank假平局）。以上台账已回填范围化状态；PointNet同类R2正在Claude整改。对应报告为2026-09-28-{himloco,paraformer,kws,asr}-independent-review.md，不能用已有测试通过替代反例修复。B8其余七样例主机评审范围已回填，整批仍等PointNet和共同文档/集成收尾。

2026-09-28：PointNet R2经Codex独立复验关闭：26项通过，原int32精度反例正确输出class 1，checker零违规，26个样例文件哈希在验收期间未变。R1/R2历史失败保留，PointNet主机运行时与关联文档范围接受；B8/H4仍待共同文档和集成收尾，不据此关闭整批。见[独立评审](../../releases/unified-migration/2026-09-28-pointnet-independent-review.md)。

2026-09-28：YOLOE运行时组合和入口经Codex独立审阅，34项Python与11项ASan/UBSan原生主机测试通过，checker零违规；保留S公开量化制品不兼容浮点路径的事实，不宣称板端通过。见[独立评审](../../releases/unified-migration/2026-09-28-yoloe-independent-review.md)。H5/B9转换/评估文档与总体集成仍待最终审核；不执行真实量化验证。

2026-09-28：首批9个B1/B2 sample根README深度恢复经独立验收（18份双语、50处固定源图引用、5份恢复的ResNet图；命令未变且本地链接无缺失）。MobileNetV3图注和EfficientFormerV2泛化说明已由Claude修正并复核。见[评审](../../releases/unified-migration/2026-09-28-readme-depth-b1b2-independent-review.md)。仅关闭DOC-DEPTH-R1的这9个根文档范围，H1整体及其余样例/子目录仍继续。

2026-09-28：YOLOE evaluator的合成计分/数据契约10项经现有pycocotools缓存环境复验通过（默认环境缺依赖的首次失败保留），14份README本地文件链接无缺失。子目录内容已对照参数/状态和源记录；发现conversion/evaluator的C++进度旧声明，连同UNetMobileNet旧评审状态交Claude做文档同步，未扩大量化验证范围。

2026-09-28：H0独立关闭：9条既有B3/B6/B7修复分支均已通过祖先关系或git cherry等价补丁纳入，B3隔离与native audit/两侧日志归档整改已读码核对，290项相关主机回归通过。见[独立集成评审](../../releases/unified-migration/2026-09-28-integration-independent-review.md)。仅H0当前定义范围完成；H3文档收尾、H8目录/来源全量检查、H9整体交付和其余H项继续，历史板测缺口不变。

2026-09-28：ASR-R1经Codex独立复验关闭：27项Python通过，原int32分数对在ctc/legacy均正确返回token1，checker零违规，62份样例文件哈希稳定。原生文件与上次独立评审哈希一致，保留原5项CTest范围而不重复无关板测/工具链验证。见[ASR评审](../../releases/unified-migration/2026-09-28-asr-independent-review.md)。H6/B10总体仍待文档/集成收尾。

2026-09-28：当前597份根/平台入口/sample/dataset README导航快照完成：3595处行内链接/图片引用中，本地目标与显式/标题锚点检查无未解析项。初版把C++ lambda误认作链接的扫描器误报已修正，不改客户代码。见[导航评审](../../releases/unified-migration/2026-09-28-navigation-independent-review.md)；仅导航证据，含并行待审文档哈希，不能替代H1内容深度或H9最终快照。

### 2026-09-28 — Twelve classifier README depth accepted

Codex independently closed DOC-CLASS-R1/R2/R3 after Claude Code + GLM
remediation. The twelve root README pairs and ConvNeXt evaluator pair are
accepted within this documentation scope: 26 unchanged command-block sets,
48 fixed-source image references, and three fresh affected-sample checkers
with zero violations. See
[the independent review](../../releases/unified-migration/2026-09-28-readme-depth-classifiers-independent-review.md).
H1 remains open for the remaining sample/subdirectory documentation audit.

### 2026-09-28 — Reviewed host status synchronized in customer guides

Accepted eight YOLOE/UNetMobileNet subdirectory README status corrections;
YOLOE-DOC-R1 and the UNetMobileNet pending-review sentence are closed.
[Independent review](../../releases/unified-migration/2026-09-28-review-status-docs-independent-review.md)
records the bounded claims and fresh checks. Batch rollups remain open.

### 2026-09-28 — VLA parent integration independently accepted

ACT/Pi0 fixed gitlinks and clean initialized sources, six bilingual parent
guides, CLI/source comparisons and two parent integration tests are accepted.
See [VLA independent review](../../releases/unified-migration/2026-09-28-vla-independent-review.md).
H7 remains open for Gemma and the remaining LLM scope; no board/robot or
quantization workflow was executed. H8 dataset documentation gaps were assigned
to Claude Code + GLM as a separate bounded package, not marked complete.

### 2026-09-28 — H4 B8 非板端整批关闭

八个 sample 全部已有独立内容/代码验收，304 个文件记录重新核对，唯一差异为
已独立通过的 UNetMobileNet evaluator 双语状态修正。当前共享实现下 181 项测试
全部通过，90 份 README 本地文件链接无缺失，8 个规范检查零违规/零豁免。
[整批独立报告](../../releases/unified-migration/2026-09-28-b8-batch-independent-review.md)
逐项列出能力、证据与边界。H4 按当前用户非板端范围关闭；Board=not-run，
整仓 H1/H8/H9 与其他批次继续，不新增实际量化验证要求。

### 2026-09-28 — Source-image depth finding and KWS overview closed

Accepted Ultralytics/FCOS/ByteTrack/KWS README package after two ByteTrack
findings and final caption/conditional refinements: ten changed documents,
twenty source-identical image references, unchanged command blocks and four
passing checkers. [Independent report](../../releases/unified-migration/2026-09-28-readme-depth-special-independent-review.md).
DOC-DEPTH-R1 now has independent disposition for all 24 affected destinations;
KWS-N1 also closed. H1/H2/H3/H6/H9 broader work is not implicitly closed.

### 2026-09-28 B10 integrated host recheck (Codex)

Fresh 109 runtime/host tests pass (ASR 27, KWS 13, Paraformer 46, HIMLoco 23); 54 README inventory/local-link checks and four zero-violation sample checkers recorded in `2026-09-28-b10-batch-independent-review.md`. H6 remains open for B10-DOC-R1 (Paraformer test-data native-consumer status) and independent reread of the Claude Code + GLM correction. No real recipe, quantization, board or robot run was performed.

2026-09-28：Gemma Text 张量包经 Codex 独立复核接受；R1 零维除零与 R2 接管失败泄漏原反例均通过，29 项主机测试及 15 项 ASan/UBSan CTest 通过，代码哈希复核无漂移，双语 README 15/3 测试计数已校正。见 `2026-09-28-gemma-text-tensor-independent-review.md`。后续由 Claude Code + GLM 完成 Text 会话/前处理/原始推理/后处理职责拆分，H7 未关闭；不做真实量化或板测。

2026-09-28：共享 datasets 双语索引/五类资源指南与 YOLO/YOLOE evaluator 导航共 16 份 README，经 Codex 独立审阅和 R1/R2/R3 整改后接受；11 份资源与 X5 固定源逐字节一致，evaluator 命令不变，链接/锚点/中英命令和检查器通过。见 `2026-09-28-datasets-independent-review.md`。H8 仍待目录历史 pin 等其余项；无下载、量化或板测。

2026-09-28：B7 整批复核：117 个既有独立评审代码/测试哈希无漂移，184 项当前主机测试通过。发现 YOLOv5/ByteTrack 多层 README 仍宣称迁移中从未板测，遗漏已归档固定提交的历史 Python 对照及 native smoke；B7-DOC-R1 已交 Claude Code + GLM 修正文档区分历史范围与当前未重测，保留 C++ 源数值对照、S100P 与 MODNet 缺口。见 `2026-09-28-b7-batch-independent-review.md`；H3 保持开放，无新板测或真实量化验证。
