# B4 分类批次：源清点与实现映射

状态：八个 sample 源清点完成；八个 sample 主机实现完成；独立主机评审完成（见独立报告），板测待用户恢复环境。Board=not-run，主机 Review=passed，Closed=no。此文件不是全批完成报告。

固定 X5 源：`ac115717197920355fc390bb04299b20e6436864`。清点针对 Git 源对象逐文件计算 SHA-256 并比较现有 platforms/x5 副本，完整文件清单和原 runtime 源见 [清点证据](evidence/2026-09-22-b4-source-inventory.json)。S 侧支持在实现绑定前核查清单，不由模型名称推断。

| sample | 源任务类 | 源文件数 | legacy 副本逐字节一致 |
| --- | --- | --- | --- |
| repghost | `RepGhost` | 22 | yes |
| repvgg | `RepVGG` | 23 | yes |
| repvit | `RepViT` | 21 | yes |
| mobileone | `MobileOne` | 22 | yes |
| resnext | `ResNeXt` | 18 | yes |
| vargconvnet | `VargConvNet` | 16 | yes |
| googlenet | `GoogLeNet` | 17 | yes |
| hgnetv2 | `HGNetV2` | 28 | yes |

## 旧→新职责映射（实现前约束）

八个源任务类均提供以下方法；最终是否共享需逐个对照预处理/score 语义。

| 源符号 | 目标职责 | 保留要求 |
| --- | --- | --- |
| Config | 复用共享不可变 ModelSelection/ClassificationContract，加 CLI/任务参数 | 沿用已批准分类架构；不新增空 config.py。源默认模型、resize、Top-K、输入路径逐项核对 |
| __init__ | model_binding + model_runner 构造；任务接收 callable | 不在任务加载文件或 SDK；标签读取归 CLI |
| set_scheduling_params | model_runner | 保留 priority、core 参数语义 |
| pre_process | 分类任务 + tensor_io | 数值/shape/dtype/插值逐源核对后共享 |
| forward | 仅调用 runner | 不混入 softmax、排序、标签、绘图 |
| post_process | 分类任务 | score 策略和 Top-K 边界按源保存 |
| predict / __call__ | 统一任务组合与兼容 API | 返回值及标签责任如实说明 |
| main / run.sh | CLI 与薄启动脚本 | 无 SDK 的 help/list/dry-run；默认和显式变体均测试 |

conversion、evaluator 和 test_data 逐文件保留。HGNetV2 有实际 eval.py 与五份 ONNX 导出脚本，不得用通用 README 替代。其余 sample 缺少导出/校准脚本时如实写前提缺口，不新增虚构流程。

下一步先实现 RepGhost 并完整核对文档，再依次 RepVGG、RepViT、MobileOne、ResNeXt、VargConvNet、GoogLeNet、HGNetV2。不得以批量替换模型名代替逐 sample 审核。

## RepGhost 主机实现（2026-09-22）

- 统一任务复用 `_shared.classification`，推理调用与模型绑定/SDK 生命周期分离；sample 本地只声明五个变体及 224/softmax/letterbox 契约。主入口和模型下载默认均为 100。
- 五个发布文件、五份 PTQ YAML、三份 test_data 均保留；X5 原 runtime 未覆盖，S 不虚构制品，原来没有 C++ 的事实不变。全套十份双语 README 按真实参数与原数据重写，清单已提前具有正确 sample_path/download_scripts，无需改发布 URL。分类索引补齐现有统一入口。
- 七项新增主机测试：SDK-free 外部 cwd 入口、默认和所有变体、目标/身份拒绝、CLI 下载实际委托、两种 resize×三个图像形状对源预处理逐字节一致、F32 forward 原样透传与源 softmax 后处理对照、五份转换配方字节保留。初始 red 为入口/模块/配方缺失（1 failure、6 import/file errors）；实现后七项通过。不是板卡验证。
- 转换边界：同一输出目录/前缀不含变体，ONNX 导出、校准集准备、版本固定和 OE 复建均缺失。README 给出有前提的命令和隔离产物要求，不把改名视为复现。
- 自审发现源 main.py 调用未导入 save_image；不改历史源，evaluator 的板端基线改用原任务 API。双语示例包含定义完整的输入、运行器、完整 raw 输出保存及比较命令，板测未执行。
- 新实现为 Codex 作者工作；独立评审尚未执行。B3 prose 漏检与补正另见 B3 host-recheck 报告。

后续顺序：RepVGG、RepViT、MobileOne、ResNeXt、VargConvNet、GoogLeNet、HGNetV2；每个都须核对源处理链、真实变体/默认、转换/评测能力与许可证，不能只换名称。

本次完整主机回归：16 个视觉 sample 套件加 shared/checker，共 559 项通过；migration checker 16 samples / 0 violations / 19 policy skips / 84 原有精确豁免。完整命令、输出、returncode 与 RepGhost 所有文件哈希见 [主机证据](evidence/2026-09-22-b4-repghost-host.json)。该结果不替代 YOLO 热修复的语义评审或板测。

## RepVGG / RepViT / MobileOne 主机落地

| sample | CLI 变体 | 默认 | 源后处理差异 | 转换专属事实 |
| --- | --- | --- | --- | --- |
| RepVGG | a0/a1/a2/b0/b1g2/b1g4 | a0 | 显式 F32 cast 后 scipy softmax | 官方重参数化；六份配置输出含连字符，发布文件含下划线；删除五类节点；历史 A0 quant 51.75 原样披露 |
| RepViT | m0_9/m1_0/m1_1 | m0_9 | 源直接 softmax，无显式 F32 cast；统一限定 F32 metadata 后对照 | timm 导出引用；三份配置共享输出路径，删除五类节点；缺可执行导出与校准准备 |
| MobileOne | s0/s1/s2/s3/s4 | s0 | 显式 F32 cast 后 scipy softmax | 官方 unfused 权重须重参数化；五份配置共用目录，jobs=32；缺固定版本/导出脚本/校准准备 |

源 pre_process/forward 的去 docstring AST 与 RepGhost 相同；post_process 逐实现复核，RepViT 的 dtype 处理差异如上。共享仅限已证明等价的 F32 分类协议，不假装源支持任意输出 dtype。三套主机测试各 9 项，包含实际保留源的前后处理比较、metadata 错误拒绝、完整发布集合/默认、下载 CLI 委托和 README 原生命令解析。先执行未实现时的 red，再实现并通过；源码与 metadata 合成测试不替代板端模型证据。

三 sample 共 14 个发布制品与 14 份 YAML；各自源 test_data 全量保留，历史 benchmark 全八列从各自 evaluator 核对，中英文一致。下载引用仍来自原 manifest，URL/发布 hash 未改。旧入口均有 save_image 定义，未错误继承 RepGhost 的源 CLI 缺陷说明。板测双语 API 示例保存原始向量并区分 ties，当前均 not-run。

作者为 Codex，独立评审未运行。下一步：ResNeXt、VargConvNet、GoogLeNet、HGNetV2；HGNetV2 的真实 eval.py/导出脚本需另行整合，不能套用“无评测程序”的文案。

本组最终主机回归：586 项全部通过，覆盖当前 19 个视觉 sample 加 shared/checker；CI 同命令 19 samples / 0 violations / 22 policy skips / 84 原有豁免。12 个双语 README API 代码块语法检查通过；24 个转换/测试资源文件与各自源逐字节一致，三个样例的双语 benchmark 完整八列表逐行与各自固定源一致。命令、输出、测试 red 记录与新文件摘要见 [本组主机证据](evidence/2026-09-22-b4-repvgg-repvit-mobileone-host.json)。这些检查都不是实际板端执行，也不是独立 reviewer 签署。

## B4 最后一组实现

- ResNeXt：保留 50_32x4d 单资产、默认与唯一 YAML，输出前缀与发布文件匹配；源无可运行导出/校准准备，按真实材料披露。
- VargConvNet：单资产、原 test_data 保留；源无 PTQ 配置/导出/发布基准表，转换和评测 README 明示缺口，不添加虚构配置或性能数值。
- GoogLeNet：单资产、历史完整八列 benchmark 保留，源无 PTQ/导出脚本，转换说明采用发布资产路线。
- HGNetV2：b0–b4 五资产，原五 YAML + 五 ONNX 导出脚本逐字节保留；实际 eval.py 改为统一 ClassificationTask/Runner。递归图片+CSV 相对路径、limit 在 GT 匹配前切片、resize=0 的评测默认（runtime=1）、成功推理分母、调度、可选标签、JSON 产出均保留。新增 topk_acc，仅 K=5 保留 top5_acc；非法 CSV 拒绝，失败/未匹配单独计数，零结果不伪装 0% 成功，输出完整覆盖率。外部 model-path 要求精确 asset-id 是统一身份约束的显式变化。
- HGNetV2 校准前提：源 README 建议 20–50 JPEG，YAML 是 float32 + preprocess_on=true；缺实际 loader/准备证据，保留配置并标为 OE 复验任务，不宣称改名/复制图片即正确。源工具链版本信息和导出模型 ID/opset/cwd 均已保留。

本组 9+9+9+15=42 项主机测试通过：其中 HGNetV2 六项 evaluator 测试涵盖 SDK-free help、CSV/路径/标签错误、分母/部分失败、空结果、默认 resize、真实任务配主机运行器的 JSON 写入。数据集/BPU/OE 均未执行。旧源文件未改写，兼容与数值基线保留。

B4 共八个 sample、27 个发布变体的主机开发已具备，仍不能标 Board passed 或 Closed=yes。独立 reviewer 正在复核前四个 sample，后四个尚待独立复核。


## 本地继续与 README 整改（2026-09-22）

用户已明确不再使用远程电脑；开发、评审和测试全部在本地推进，暂跳过板端环境验证。没有发起远程授权、板卡连接、推送或发布。

前四个 sample 的独立复核发现支持矩阵未采用三态、概述缺算法来源、目录说明残留 ibex 名称，以及前提版本边界不清。已在八个 sample 的 16 份根 README 与 16 份 runtime README 同步整改：逐目标/变体列 Python 与 C++ 状态，恢复各自固定源算法说明和参考来源，标注真实输入名、实测主机依赖版本，明确板端镜像/SDK/最低资源待验证，并补齐默认入口。VargConvNet 源无独立论文/上游仓库链接，如实保留缺口。

独立 reviewer 已修正最初测试缺口等级：共享 ClassificationTask 的 predict 等价和交错 context 测试已存在且通过；逐 sample 真实 binding 场景的增强测试仍为 P2 非阻断建议，未冒称新增。当前 README 修改为作者整改，待独立确认；后四个 sample 尚无独立签署。

本次 628 项主机测试全部通过（23 个视觉 sample、shared、checker）；规范检查 23 samples / 0 violations / 26 policy skips / 84 exemptions。16 份根 README 本地链接与主机版本声明核对通过。完整命令、stdout/stderr、返回码和八个 sample 文件摘要见[本地主机证据](evidence/2026-09-22-b4-local-regression.json)。此前证据保留为历史快照，不作为整改后 README 的哈希绑定。

Board=not-run、Closed=no；不能把本地主机通过解释为 B4 客户交付验收完成。


后四样例独立评审完成，主机可推进；详细通过面与 P2 增强建议见 [独立评审](2026-09-22-b4-independent-host-review.md)。本报告较早的“待评审”描述为执行历史，以此状态为准。八份绑定注释固定源 SHA 已由作者补全。

## 逐样例 predict/context 增强收尾（2026-09-23）

八个 sample 各新增一个遍历全部发布变体的测试，校验显式三阶段与 predict 结果相同，A(17×31)→B(29×11)→A 的原图尺寸、transform/context、输出和先前输入张量不被后续调用覆盖。实际使用各 sample 的真实 binding，并非仅依赖共享层 fixture。没有改动生产代码。

`review_b4_first_four` 编写测试，主任务另行检查并完整运行八套测试：RepGhost 8、RepVGG/RepViT/MobileOne/ResNeXt/VargConvNet/GoogLeNet 各 10、HGNetV2 16（含原有 6 项 evaluator），合计 84 全部通过。该补充测试的作者不是其独立签署人；复核由主任务完成。原先“尚未新增”的 P2 描述保留为历史，在此关闭。命令、完整输出及测试文件摘要见 [增强证据](evidence/2026-09-23-b4-context-coverage.json)。

本次只增加主机覆盖，Board=not-run、Closed=no 不变。
