# B5 源清点与实现映射

状态：固定源清点与五个 sample 主机实现已完成；五个 sample 的代码及文档独立主机复核全部通过。板端验证按用户要求跳过；Board=not-run、Closed=no，Review 逐 sample 单列。没有使用远程电脑。

源逐文件 SHA-256 与 legacy 副本一致性见 [清点证据](evidence/2026-09-22-b5-source-inventory.json)。

| Sample | 固定源 | 文件数 | 输入/输出与不可合并差异 |
| --- | --- | --- | --- |
| clip | ac115717197920355fc390bb04299b20e6436864 | 18 | X5 BPU 图像编码器 + CPU ONNX 文本编码器，保留 BPE 词表、tokenize、余弦相似度及标注输出 |
| siglip | 380e1a2bf42041af54be6f34935e50197cfadff9 | 15 | S100/S100P 共用八个发布 HBM，各含 pooler_output/last_hidden_state 子模型；RGB [-1,1] + AREA letterbox，特征张量非分类 |
| dinov2 | 380e1a2bf42041af54be6f34935e50197cfadff9 | 18 | S100/S100P/S600 三种 march，RGB bicubic短边256中心裁224，F32或显式反量化特征，不套分类 softmax |
| vit | 380e1a2bf42041af54be6f34935e50197cfadff9 | 52 | S100 int8/int16，两平面 NV12 224，CIFAR-10十分类；源默认直接nearest resize，非ImageNet1000 |
| 3dresnet | 380e1a2bf42041af54be6f34935e50197cfadff9 | 24 | S100 r3d_18，已归一化 (1,3,16,112,112) 视频张量，Kinetics-400；保留.npy输入与JSON标签 |

## 实现约束

- CLIP 与 SigLIP 保持独立，不把文本能力从 CLIP 丢弃，也不虚构 SigLIP 的文本编码器。
- ViT 可复用现有分类共享模块，但必须以实际十分类、S NV12分平面、nearest/default=0 契约绑定；用源 hb_compile.log 的输入输出记录支撑基础协议，不能把历史日志当作两个当前发布制品的板测回执。
- DINOv2 的定量输出变换放在后处理，runner 只传 raw + metadata；3DResNet 不转图片。
- 模型准备和标签/词表读取由独立模块承担；任务仅前处理、forward、后处理和 predict，适当的纯辅助模块允许。
- 每个转换目录必须保留原材料并披露缺失前提；ViT 配置与历史编译日志存在路径差异，不能把唯一配方宣称为 int8/int16 两制品的完整可复现来源。
- 后续先复用协议已明确的 ViT，再处理多编码器/特征/视频样例；不改变 B5 五样例完整范围。B4 后四独立评审完成前仅进行源清点和测试准备，不宣告进入交付下一批。


## ViT 主机实现与职责映射

源 ViTConfig.model_path → 显式 ModelSelection；ViT 的三阶段 → 共用 ClassificationTask，十分类/224/分平面/nearest 默认0 写在样例 binding；SDK 加载与调度 → RuntimeModelRunner；标签读取/输出 → CLI。保持 int8 默认、int16 别名与原生 run.sh 位置参数，增加 host list/dry-run 与外部模型精确身份要求。没有新增 C++ 或非 S100 支持。

保留一个 YAML、完整编译日志和全部 test_data（含 CIFAR 字典标签和原插图）。原日志记录 U8 Y/UV 与 F32 [1,10]；这是历史协议证据，不是当前两个发布模型的板测回执。转换说明如实披露校准路径/编码和两量化变体的配方对应缺口。

测试先 red：10 项中 1 failure/9 errors（入口/模块/资源未实现）；修正了源辅助包导入 fixture 才取得该 red。实现后自查纠正两个测试假设：标签实际为字典字面量而非行文本；显式 target 是准备选择，执行身份拒绝在 require_execution_target，新增真实 CLI gate 测试。最终 13 项通过，含两变体×两模式×三图形的 NV12 字节对照、原始输出/softmax 对照、十类 metadata 拒绝、精确下载、predict/显式三阶段/交错 context、双语 README API 实际 fixture 执行。规范检查 0 violations。

B4 本地主机独立评审已完成，故按用户本地推进规则进入 B5。当前只有 ViT 完成实现；CLIP/SigLIP/DINOv2/3DResNet 仍待实现，不能宣称 B5 完成。


## ViT 独立复核与回归收尾

独立 reviewer `review_b4_first_four` 已复核全部 ViT 源码、五级双语 README、转换材料、真实源能力与统一 manifest。独立执行 ViT 13 tests、manifest coverage 4 tests、规范检查 0 violations；原 YAML/log/test_data 与固定源逐字节一致，README API fixture、命令和本地链接通过。没有新的主机推进阻断。

回归中发现统一 manifest 仍登记不存在的 download_model.sh：作者最初误改了平台快照一行，已精确撤销，统一 docs/release/s/models.yaml 改为真实 download.sh；README authority link 同步指向统一清单。shared 71、ViT 13 和全 scope checker 已复跑通过；独立 reviewer 再次确认这两项已关闭。失败与修复后输出都保留，不用重写初始记录掩盖问题。

本轮 24 个视觉样例 + shared/checker 共 641 项测试通过（分套件验证；清单修正后重跑受影响的 shared/ViT）。CI 同命令 24 samples / 0 violations / 27 policy skips / 84 原有精确豁免。完整初次回归、修复复验和当前 README 哈希见 [ViT 主机证据](evidence/2026-09-22-b5-vit-host.json)。

ViT review_decision=pass（本地主机范围），delivery_readiness=not-ready。Board=not-run、Closed=no；两个量化制品的真实运行、精度和 OE 转换仍待板端/工具链交接验证。B5 其余四 sample 待实现。


## SigLIP 实现进展与源事实更正

最初清点仅按制品存储前缀写“S100”，复核完整源根/model/evaluator 后更正：固定源明确同八个 HBM 支持 Nash-E/S100 和 Nash-M/S100P，并有两板历史性能表。统一绑定据此显式列出两个目标共用同一 manifest asset；不是借用 S100 资产静默回退。统一 manifest 的 notes 记录该依据，下载脚本路径同步新入口。

- 8 变体逐项使用源 evaluator 的 image size / embedding D / patch N；384 patch14 保留 729 tokens。
- 每个 HBM 的两个 submodel 都在 lazy runner 加载时核对 metadata，只运行显式所选子模型；priority/core 同时下发两子模型，沿用源行为。
- SigLIPTask 保留三阶段 + predict。前处理 RGB/AREA/127 padding/F32[-1,1]；context 每次返回。forward 只调用runner；post检查 shape/dtype/finite 后返回 ownedcopy，保持源 native shape/dtype，不加 softmax/dequant/squeeze。CLI 独立负责统计和可选完整 NumPy 文件保存。
- 原 docstring 的 pooler (1,D) 与源历史表 (1,1,D) 都接受，但实际模型绑定后执行时必须保持准确shape；hidden要求源表(1,N,D)。
- 独立 reviewer 已只读核对四个 core 和源测试，无阻断，独立13 tests通过；CLI/README及完整交付不在该初次复核范围。
- 新增真实CLI/输出保存/身份失败/runner返回协议及源post对照，作者17 tests通过。README 正在编写，整样例未完成，Board=not-run、Closed=no。

## SigLIP 主机实现与文档齐备（2026-09-23）

五级十份双语 README 已完成，覆盖源能力/算法、两个目标八变体两子模型的支持矩阵、实际 CLI 默认、完整 API、精确 manifest 下载条目和转换材料缺口。四张历史 benchmark 表全部保留，和本轮主机验证分开；没有虚构文本编码器或数据集 evaluator。模型入口支持完整原生 ndarray 保存，统计/写文件均位于 CLI。

作者独立复跑 19 项主机测试全部通过；双语运行 API 在真实统一 binding/runner + 注入 runtime fixture 中执行，原生命令参数与本地链接均检查通过。单 sample checker 0 violations。源码固定源对照、代码/文档 SHA、全部命令输出和初始 red 见 [SigLIP 主机证据](evidence/2026-09-23-b5-siglip-host.json)。

板端对照步骤使用实际保留 legacy 目录，完整保存 raw 张量到唯一 UTC 运行目录；先检查 shape/dtype，整数精确比较、浮点 rtol=0/atol=1e-5 并通过断言决定退出码。该步骤只是待执行方案，不属于已测结果。

SigLIP 完整独立评审正在进行；Board=not-run、Closed=no。DINOv2 正在独立实现、文档并行，尚未宣称通过；CLIP/3DResNet 尚待实现，B5 未完成。

## 剩余 CLIP / 3DResNet 契约预审（2026-09-23）

CLIP 不能套用单图分类任务：X5 image `.bin` + CPU text `.onnx` 必须成对选取。源明确 image F32[1,3,224,224]→[1,512]，text I32[N,77]→F32[N,512]；输入短边224 bicubic、round后中心crop、/255，没有 ImageNet/CLIP mean/std。文本 BPE 词表和清洗/tokenize 必须保留，SOT/EOT 与超长文本拒绝保持源默认；文本编码放入 runner，数值 cast/flatten/cosine/rank 放后处理，forward 仅运行两个编码器。源 cosine 是 float32 norm+1e-12，不加 softmax/温度。保留默认两段文本、dog图、标注图输出和 priority/core。源不存在 conversion YAML/导出/基准表，不虚构。源动态读取 tensor 名称，README 协议表中的概念名不能擅自当发布模型 metadata 的精确名称。

3DResNet 是视频分类：S100 的 r3d_18.hbm 接受预处理好的 (1,3,16,112,112) clip；仓库真实 video0.npy 已读取核实 shape/dtype=float32，Kinetics JSON 为 name→整数 id，共400类，不能按普通行标签读取。保留 npy输入和 Top-K、完整截图与四行线程性能表；源没有视频解码/抽帧脚本和转换 YAML/导出脚本，OE3.5.0 pooling替换说明为历史材料，不补虚构一键转换。共享 Top-K数值逻辑可复用，但不让五维视频误走二维图像预处理。

以上是实现约束与源码/资源核查，两个 sample 仍 pending，尚无主机通过或独立签署。

## SigLIP 独立复核完成（2026-09-23）

独立 reviewer `review_b4_first_four` 审查全部 runtime/CLI/model/download、manifest、十份 README、conversion/evaluator 和固定源能力，独立执行 19 tests 与 import/static checker。八变体×两目标×两子模型、729 tokens、native shape/dtype、完整四张历史表均确认一致，没有主机阻断。唯一 P2 为双语 runtime 的用户自定义相对图片路径说明；已改为“默认由 sample 位置生成绝对路径，用户相对路径按 cwd 解释”，reviewer 再次核对两行与代码，明确关闭。

SigLIP review_decision=pass（本地主机范围）、delivery_readiness=not-ready；Board=not-run、Closed=no。初期 core-only 评审已由本次全范围结论补齐。

## DINOv2 主机实现（2026-09-23）

保留 S100/Nash-E、S100P/Nash-M、S600/Nash-P 三个独立资产，不沿用源 downloader 的未知目标回退。统一任务前处理为 RGB bicubic 短边256/中心224/ImageNet normalize；runner 只适配 raw 双输出，后处理按实际每输出 quant descriptor 复用 shared dequant，返回所选 owned F32 特征，无 activation。CLI 保留两图、统计、cosine（F64计算）并增加完整 NumPy 保存；两图 cosine 不冒充浮点模型精度评估。源 mapper/export/测试图片逐字节保留。

作者验证最初误用系统 Python，其缺依赖失败不构成有效 TDD red，未伪装成先失败证据。改用仓库 `.venv` 后修正注入 runtime 仍提前加载 SDK 的问题及测试 fixture 递归；14 项实际测试通过。独立 reviewer 完整代码复核和14 tests通过，无实质缺陷，文档范围另行复核。主任务再加 3 项 README 验证：实际执行两语言API、原生命令/本地链接、双conversion脚本字节与README checkpoint摘要一致，17 项通过。

README 编写曾继承主任务转述错误的 checkpoint SHA；已按固定源两个脚本的真实64位 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` 更正说明及命令参数，未改源转换代码。新增测试从脚本AST读取常量核对两语言，防止拷贝转述漂移。完整文档独立复核进行中；Board=not-run、Closed=no。

CLIP 核心与CLI已实现，12项主机测试通过，文档编写中；3DResNet正在实现。B5尚未完成，不进入B6交付。

## DINOv2 / CLIP 文档整改与独立收尾

DINOv2 reviewer 的两项P2已关闭：转换文档补齐空环境到固定源码/权重的完整命令、保留Torch并装固定ONNX依赖，恢复源五行量化配置失败/成功矩阵与reg4未发布边界；evaluator新增同板legacy/unified自包含程序，保存完整input、双raw、双结果及JSON/digest。历史ONNX精度仍manual/not-implemented，未伪装新增数据集evaluator。主任务实际以两个host runtime fixture执行双语对拍程序：正常对照通过，故意偏移的整数raw触发失败并仍保存9个数组及JSON，整套18tests通过。这不等于实板执行。独立reviewer确认代码与文档无主机阻断。

CLIP 以两个真实发布资产构造独立runner，BPU图像和CPU文本按顺序执行；真实BPE词表/清洗/77tokens保留，后处理保持源F32 cosine+argsort，无温度或softmax。文件读写/绘图和词表初始化独立于推理task，原两文本/dog图/inference.png默认保留（默认会覆盖该样例结果图，文档明确）。stage错误保留image/text归属并阻断后续调用。7项有效初始red后实现；增加CLI实际pipeline、下载mock、错误归属、异常raw及双语API执行后15项通过。主机仅在仓库虚拟环境安装ftfy6.3.1、regex2026.9.10及其wcwidth依赖以运行真实BPE；没有安装板端SDK/下载模型。

CLIP完整代码与十份README独立复核通过。三项P2均整改并由reviewer确认：绝对默认/相对cwd路径语义、短边224/长边round几何、evaluator必须先身份gate再legacy/SDK。额外移除raw隐式reshape，比较前先校验精确shape/dtype及input/token，并撤回跨版本tie排序稳定承诺。两sample均review_decision=pass（host）、delivery_readiness=not-ready。

## 3DResNet 主机实现与集成修正

S100预处理视频输入与400类JSON标签保留，分类任务不解码视频、不做二维图像变换；已验证源float32 cast、真实source softmax/Top-K和CLI标签输出。源全部测试资源与八张截图保留，OE3.5.0的pooling改写仅作为历史材料；源无导出/校准/YAML不虚构。

主任务集成审查发现初稿用通用裸模块名且任务文件误叫embedding，作者已改为classification.py、库内相对导入、入口完整importlib模块名，避免3dresnet数字包名语法问题及跨sample冲突；新增干净repo子进程导入验证。绑定重新校验完整manifest事实，复用shared score_vector_shape/Top-K，严格JSON整数标签。作者有效RED后11tests、集成修正12tests；主任务补两语言README API实际执行及原生命令/链接，14tests通过。独立review正在进行；不能把作者完成声明作为独立签署。

## B5 主机收尾状态（2026-09-23，以本节为准）

五个sample的主机开发、五级双语README和独立评审完成：ViT 13、SigLIP 19、DINOv2 18、CLIP 15、3DResNet 16，共81项样例测试。独立评审过程、关闭项和未验证边界见 [独立主机评审](2026-09-23-b5-independent-host-review.md)。

全量28个视觉样例+shared/checker初次715项通过；其后只改3DResNet文档/测试，增加shell参数和tie负例两项，受影响完整16项复跑通过，当前覆盖717项。规范检查28 samples / 0 violations / 31 policy skips / 84原有B9精确豁免；未扩大豁免或规则降级。完整命令/输出见 [全量回归证据](evidence/2026-09-23-b5-local-regression.json)，当前五sample文件哈希、源资源字节核对和red记录见 [工作树证据](evidence/2026-09-23-b5-working-tree-snapshot.json)。

3DResNet最终独立评审发现一项P1（README误用shell下载脚本的命名参数）和一项P2（测试数字漂移）。已改成位置参数s100、删去易漂移的数量承诺，并加fake python3验证shell转发参数，全程不联网；reviewer复核关闭。evaluator初稿自动放行近似平局不符合既有B2原则，主任务撤回：任何ID不一致都写出证据并失败，新增全零精确平局负例实际执行README程序验证不自动豁免。

**评审过程偏差已披露：** reviewer误执行了一次真实3DResNet模型下载到本机临时目录，未执行HBM或板端测试。主任务核实38,795,408字节及observed SHA后已删除该文件和空目录，未进入仓库。不能将整轮描述成“零模型下载”；具体命令/URL/输出/清理证据见 [事件记录](evidence/2026-09-23-b5-review-download-incident.json)。自动测试中的下载均是mock；该误操作不构成支持验收。

最终：review_decision=pass（本地主机范围）、delivery_readiness=not-ready；B5各行Board=not-run、Closed=no。旧platforms基线未修改；没有使用远程电脑、连接板卡、提交、push或发布。B6只读源清点已完成，尚未实现；B6–B11与全仓基础合并收尾仍待继续，整体目标未完成。
