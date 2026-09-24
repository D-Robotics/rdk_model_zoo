# B7 native SDK 独立核对（2026-09-24）

基点：HP 作者整改提交 `5fc14a4`；本地 GLM 开发正在继续，本报告不代表其后续修改状态。

## 阻断：S adapter 使用不存在的 X5 API

通过已有 SSH 配置在真实 S100 读取 `/usr/include/hobot/dnn/hb_dnn.h`：`hbDNNTensorProperties` 没有 `alignedShape`；`hbDNNQuantiType` 只有 NONE/SCALE，没有 SHIFT；`dnn/hb_dnn_ext.h` 不存在。当前 `samples/vision/yolov5/runtime/cpp/src/s_adapter.cpp` 却包含该头、引用 SHIFT，并访问 properties.alignedShape。必须按 S UCP 实际 SDK 单独适配，不能用 host portable tests 证明 native 可编译。S 头位于 `/usr/include/hobot/dnn`，CMake 的 include root 也须配合 `#include <dnn/hb_dnn.h>` 核查。

X5 的对应字段/枚举确实存在，不能将 S 修复机械应用到 X5。完整命令、原始输出和时间见 [SDK evidence](evidence/2026-09-24-b7-native-sdk-preflight.json)。

## 环境与边界

X5 8GB/4GB、S100、S100P、S600 64GB 五个既有 SSH 目标均可达。真实板端构建正在准备 GitHub 上的上述提交，推理尚未执行。不得将连接成功或头文件核对记录为板端推理 passed。开发仍交给本地 Claude Code + GLM，由 Codex 统筹 GitHub 和独立评审。

## 实板执行追加

同一 `5fc14a4995110990f2abaa6ec8189d052a0195e6` 在 X5 8GB 编译 rc=0，S100 编译 rc=2，首个编译错误为 `dnn/hb_dnn.h` 找不到。原始构建输出保存在 evidence/2026-09-24-b7-board-initial/。

X5 C++ 默认 s-v2.0 已实际推理 rc=0，bus.jpg 上输出五个 detection，模型 observed SHA-256 为 `1b627740f4a9d322efda8ece8ae01a088900ab28bd62c11a47b31981651e2aee`。发布者未提供 hash；运行库报告 model/runtime HBRT 小版本不一致，原始警告完整保留。此结果仅为 smoke，不是源数值一致性验收。

进一步阻断：

- C++ dump 实际只保存三个 output bin。manifest 的 inputs 只有描述、dtype 为 unknown，没有 native NV12 输入 bytes，也没有部署代码/二进制 digest。仍不满足完整 source/unified 比较证据，不能将 dump 的存在当作该 finding 已关闭。
- X5 Python evaluator 在真实 SDK 抛 `TypeError: cannot pickle hbm_runtime.HB_HBMRuntime.QuantParams object`：`compare.py` 对 `RuntimeMetadata` 调用 dataclasses.asdict，触发对 SDK 对象的 deepcopy。失败发生在 legacy 模型初始化的 metadata capture 中，尚未比较输出。需要显式序列化 SDK metadata/quant 字段，并用禁止 deepcopy 的测试替身覆盖；排查其他 evaluator 的同类模式。失败 comparison.json 已原样保留，禁止修改为通过。

证据目录：[首次板端构建/推理/失败记录](evidence/2026-09-24-b7-board-initial/)。C++ raw 输出仍在板端 `/tmp/rdk-b7-x5-native-dump`，尚未作为仓库可携带原始数组归档；故当前不支持完整对照验收。

## 后续独立复现：dump 文件覆盖与共享序列化模式

使用实际 `yolov5_dump.cpp` 编译 SDK-free probe，输入同名 output0 的 int32 原始 bytes `{1,0,0,0}` 和 float32 转换 bytes `{0,0,128,63}`：write_dump 返回成功，但两者都写 `0-output0.bin`，原始数组被覆盖。manifest 原始 SHA 与磁盘实测不一致。见 [可复现 probe 源码和结果](evidence/2026-09-24-b7-board-initial/b7-dump-collision-probe.json)。修复需按类别隔离文件名，并测试从磁盘重新读取和核对两侧 digest；同一 dump 目录的覆盖策略也应明确。

代码核对发现六个 B7 evaluator 都存在 dataclasses.asdict(RuntimeMetadata) 或等效写法：YOLOv5、ByteTrack、FCOS、LPRNet、MODNet、YOLOWorld。真实 X5 SDK 已证实无法 deepcopy QuantParams；其他五项尚未逐一实板执行，当前结论为共享风险，不虚称已逐项板端复现。建议统一可 JSON 序列化的 metadata 投影，保留 scale/zero_point/axis/type/stride，不改推理语义，并覆盖真实对象不可 deepcopy 的边界。

## S100 发布资产的实际 padded 输出

已下载 manifest 指定 `s100/yolov5x_672x672_nv12.hbm`，observed SHA-256 `3bc8ffc82a842a5d1fcb493eeedc8f6e3dd0a55379338d27e5cdca4e34121402`（publisher hash 未提供）。真实 HB_HBMRuntime 输出为 S32/SCALE，逻辑 channels=255，但每像素 stride=1024 bytes，即256个S32存储槽：84头 stride `[7225344,86016,1024,4]`，42头 `[1806336,43008,1024,4]`，21头 `[451584,21504,1024,4]`。因此当前 `check_s32_dequant` 要求 `stride[2] == 255*4` 将拒绝真正发布制品；必须按 stride 读取并生成紧凑逻辑输出，不能将所有 padded 布局拒绝作为完成条件。输入SDK报告的前两个 strides为-1，需要依据原有 prepare_input_tensor 初始化后再校验，不能把初始化前描述直接当可寻址存储。该步骤仅为模型加载和 metadata 核对，尚未进行 S 推理。完整输出见 evidence/2026-09-24-b7-board-initial/b7-s100-native-metadata.json。

## S100 Python 正常入口：signed dtype alias 阻断

`python3 samples/vision/yolov5/runtime/python/main.py --target s100` 在相同发布制品/基点实板返回2：`Unsupported native output dtype 's32'`。SDK enum `hbDNNDataType.S32` 经共享 `canonicalise_dtype` 变成未识别 s32，而模型绑定要求int32；应把有证据的 S8/S16/S32 与相应 int dtype 对齐，不能放宽未知类型 gate。原始命令/输出见 evidence/2026-09-24-b7-board-initial/b7-s100-python-smoke.json。

本地开发已拆为三个隔离 GLM 工作单元：native 主任务（GLM-5.3），B7 metadata serializer（GLM-5.3-flash），signed dtype alias（GLM-5.3-flash）。Codex 负责逐项复审和 GitHub 同步，作者不自行提交/合并；新增修复尚未声称通过板测。

## signed dtype 修复独立复验通过（限定范围）

作者专项 `7f27c8c0e752ef6e7849eb509138bb44d0984ed1` 已推送 GitHub 分支 `codex/b7-glm-dtypes-20260924`。Codex 审查修改仅对 S8/S16/S32 做 signed alias 规范化，独立运行 shared111 / YOLOv5 32 测试全部通过。在 S100 经 GitHub fetch/checkout 该提交后，默认 Python 入口由 rc=2 变为 rc=0，输出14个检测。逐项解析完整 stdout JSON，全部 boxes/scores/class_ids 与同板同模型固定源 Python 结果完全相同。修复后的 runtime_meta.py 板端SHA-256 `563331b8faea635f38a0a26c07d7eded11f063e8872493b2cace84b9a8abebb6`，详见 dtype-recheck 和 result-comparison evidence。

固定源 S C++ 也已用真实 SDK 编译并执行 rc=0。该记录证明环境/制品可用，不替代统一 C++ 尚未完成的整改验证。dtype 单项可关闭；B7 整体 remains changes-required，完整 native input/raw 比较仍待 serializer 修复后补齐。专项分支尚未整批合入 develop。

## 目录构建补验

在 coordinator develop 使用项目要求的 Node22 补跑 HP 未具备环境的 catalog build/typecheck，发现 X5 summary 仍是补回八个 YOLOv5 制品前的计数。仅将 asset_count 177→185、downloadable_asset_count 176→184、sha256_unrecorded_count 148→156 对齐清单实际条目，未改变任何制品 URL/hash。修复后 build/typecheck rc=0；完整 Ultralytics Python suite 78 tests OK（含需要生成catalog的检查）。这是当前 develop 的独立验证，不把尚未合入的其他 GLM 工作树宣称为全量通过。完整日志见 evidence/2026-09-24-b7-catalog-recheck.json。

## Native 第一阶段 dc3aac5 再审

主机35项 YOLOv5 / 16项 3DResNet 通过后，将 GLM 第一阶段快照 `dc3aac50230e6d6a17324592be1bbc71f04abd53` 经 GitHub 同步 S100。真实构建仍rc=2：S packed handle类型实际为 `hbDNNPackedHandle_t`，代码与其主机stub却用了X5的 `hbPackedDNNHandle_t`。因此stub编译通过不足以关闭真实SDK兼容性，需校正stub而非添加虚构别名。

同阶段独立portable probe还证实 gate误放行重叠像素存储（channels255/stride3=4但stride2=400）和单个scale被交给逐channel helper。新gate错误地以width代替channels判定stride2下界。完整probe源码、gate SHA和输出已归档；这是该阶段快照结果，后续修复需另留复验。

S100额外SDK核对确认S8/U8/S16枚举均存在，可正确标注NV12输入U8；UCP backend是位掩码，core0/1/2/3分别为1/2/4/8，ANY为128，不能直接传用户core索引。上述问题、dump覆盖/缺输入bytes及部署身份缺口已作为native第二轮有界任务交同一本地GLM会话；不再让它重复处理已经独立完成的dtype和catalog。代码快照仍只在专项分支，未合入develop。

## dtype 单项整合 develop

在板端复验通过后，仅将signed dtype独立修复cherry-pick到develop，提交 `4dce619`。在develop实际树重新运行shared111 / YOLOv5 24项通过（此树尚未合入HP/native新增测试，故不混称35项）。原B7整体仍changes-required；native/evaluator整批修改继续隔离。整合日志见 evidence/2026-09-24-b7-dtype-develop-integration.json。

## Python 首轮完整对照通过：ae0f185

将metadata核心检查点 `a141c48` 与已实板通过的dtype修复组合为板测分支提交 `ae0f18522f5512b4505f3bc2457ff2afd39af3c9`，独立shared118 / YOLOv5 32项主机测试通过。两块板均从GitHub获取此提交：X5 8GB使用s-v2.0/bus，S100使用x-672/kite；固定源与统一入口同板、同制品执行，两个 evaluator 均rc=0/所有checks=true。全部输入、raw输出、boxes/scores/class_ids逐项数值最大差均为0（class ID dtype按既有明确契约归一化，不声称全文件逐字节相同）。

原始14份X5数组、16份S100数组及完整metadata/code/model/input SHA、argv/cwd/时间已保存在各自tar.gz，协调者从归档重新读取每份数组并验证comparison.json列出的SHA全部一致。见 evidence/2026-09-24-b7-python-comparison/ 的执行记录、comparison、archive-verification及归档。模型publisher hash仍未知；X5模型/运行库小版本警告原样保留，未用成功对照掩盖。

这是两个具体模型/图片case，不外推到其它变体、板卡、样例或C++。六个evaluator的专项接线回归测试仍由GLM补齐，B7整体保持changes-required；核心代码在专项/板测分支，尚未作为完整B7合入develop。

## X5 扩展三个样例（ae0f185）

同一X5 8GB和板测分支，显式使用manifest下载LPRNet/FCOS默认efficientnetb0/YOLOWorld模型后，执行各自完整source/unified evaluator：

- YOLOWorld默认dog提示、固定dog.jpeg：rc=0/ passed=true，14份原始数组及metadata/参数/代码hash归档，并从tar.gz逐数组复验SHA。仅覆盖这一提示/图片，不声称全开放词汇精度。
- LPRNet：rc=2，实际发布模型native output是 `(1,68,18,1)`，统一绑定错误地只允许 `(1,68,18)`；保留3份失败前数组和完整失败记录。
- FCOS efficientnetb0：rc=2，实际输出dict的15个name集合与metadata相同，但插入顺序不同；tuple(outputs)比较造成误拒绝。保留18份失败前数组、traceback和完整metadata记录。

三份tar.gz与comparison/执行/归档校验记录均在 evidence/2026-09-24-b7-python-comparison/，失败归档不冒充完整成功对照。LPRNet/FCOS native输出契约已交独立GLM任务修复，保持名字、dtype、实际shape与有限值验证，不允许以任意reshape或删gate规避。
