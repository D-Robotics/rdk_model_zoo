# RDK Model Zoo — 暂停与 Claude Code 交接

用户在 2026-09-23 明确要求暂停，并交给之前的 Claude Code 接手。Codex goal 已设为 **paused**，三个并行任务全部中断。本文件是暂停点交接，不代表 B7 或整体迁移完成。暂停后只整理文档和工作区快照，未继续实现、未执行新一轮测试。

## 还剩多少工作

按本地交付包计算，剩 **6 个大步骤**，不是六个小修复：

| 顺序 | 剩余范围 | 当前状态 |
| --- | --- | --- |
| 1 | B7 六个 sample 的整改、README 复核、批次回归及独立评审收尾 | 已有全部初版；Python 核心部分已验证，仍有明确阻断和中途修改 |
| 2 | B8：unet、unetmobilenet、pp_liteseg、yolo26_depth、depth_anything_v2、lanenet、pointnet、diffusiondrive | 8 个 sample 待实现；刚完成只读源清点，250 文件全部与固定源相同 |
| 3 | B9：ultralytics_yolo 收编与 yoloe 家族 | 2 组家族整合，涉及多个源 sample；必须清掉 84 条延期 README 豁免 |
| 4 | B10：himloco、asr、kws、paraformer | 4 个 sample 待实现 |
| 5 | B11：gemma4-e2b、minicpm5-2b、VLA act/pi0 | 2 个 LLM + 2 个 gitlink 集成单元待处理 |
| 6 | 全仓整合收尾和最终本地主机评审 | 根文档/入口/manifest/registry/skills/兼容路径/增量核对/完整交接 |

真实板端复验、OE 转换/校准和最终客户验收另列后续任务。用户允许先完成本地基础开发，未板测不阻止进入下一批，但不能据此宣布客户版本 ready。旧 platforms 目录的删除仍受能力、引用和证据条件约束；当前大量 evaluator 还依赖其中的固定源，不能提前删除。

## 接手位置与必须保留的状态

- 实际 Git 根：`/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk_model_zoo`。父目录不是 Git 根。
- 分支：`develop`；HEAD：`16c5d04d0c71ebd160400c14fb2be1c2eb2513f0`。
- 固定源：X5 `ac115717197920355fc390bb04299b20e6436864`；S `380e1a2bf42041af54be6f34935e50197cfadff9`。
- 本轮 B3 文档修正、B4–B7 大量代码/测试/文档/证据仍在**未提交工作区**；不少整个 sample 目录是 untracked。仅拉远端拿不到这些成果。不要 `git clean`、reset、checkout 覆盖或按旧 HEAD 重建目录。
- 暂停快照：[文件 SHA-256 和 Git 状态](evidence/2026-09-23-handoff-working-tree.json)。快照覆盖修改与未跟踪文件（不包含该快照自身和被 Git 忽略的环境文件）；先核对后继续。
- 原主计划：`/Users/Max/.claude/plans/golden-scribbling-micali.md`；暂停说明已追加并保留备份。仓库内还需读 [AGENTS](../../../AGENTS.md)、[Spec](../../superpowers/specs/2026-09-16-rdk-model-zoo-x5-s-agent-people-spec.md)、[台账](x5-s-migration-map.md)、[主机推进/板端交接规则](2026-09-22-host-development-and-board-handoff.md)。
- 不使用远程电脑，不重启 GitHub 设备码授权，不探测局域网板卡。板端环境验证先跳过；不把 `not-run` 写成 passed。B1/B2 既有证据保留，S600 MobileNetV2 C++ 维持用户决定的 not-run。
- 本轮没有 commit/push/发布，也没有切换默认分支。不要把有文件等同于已入库或已评审通过。

## 已完成到哪里

| 批次 | 本地主机进度 | 证据入口 |
| --- | --- | --- |
| B1/B2 | 原有验收历史保留 | 既有批次报告和台账，不重写历史结论 |
| B3 | 原 R1–R4 复核；三份中文 evaluator 无关精度文案修正 | [B3 recheck](2026-09-22-b3-host-recheck.md) |
| B4 | 八个分类 sample 迁移、双语文档和独立主机评审完成 | [B4 独立评审](2026-09-22-b4-independent-host-review.md) |
| B5 | ViT、CLIP、SigLIP、DINOv2、3DResNet 主机实现和独立评审完成 | [B5 独立评审](2026-09-23-b5-independent-host-review.md) |
| B6 | EfficientSAM/MobileSAM 共享 binding/runner/stages/evaluator、转换资源和二十份 README 完成，独立主机复核完成 | [B6 独立评审](2026-09-23-b6-independent-host-review.md)、[B6 回归](evidence/2026-09-23-b6-local-regression.json) |
| B7 | 实现及整改中，**尚未批次通过** | [B7 批次记录](2026-09-23-b7-detection-tracking-review.md)、[独立初审](2026-09-23-b7-independent-host-review.md) |

B6 最后覆盖口径是初轮全量 781 tests，通过后新增两项边界并复跑受影响 shared101 / EfficientSAM19 / MobileSAM17，覆盖共 783；当时 checker 为 30 samples / 0 violations / 84 原有 B9 豁免。它不是当前 B7 工作树全量通过证据。B3–B6 的 Board/Closed 仍保持 not-run/no。

## B7 暂停点：逐项接手

### YOLOv5 Python 和 ByteTrack

root 编写的 Python 实现已通过 **15 + 11 = 26 项**定向测试，并由另一 reviewer 独立运行和源码复核；报告全文尚未来得及单独落盘，结论在本交接中如实保留。[完整自检日志/版本/代码哈希](evidence/2026-09-23-b7-yolov5-bytetrack-host.json) 已入工作树。没有板端运行。

- YOLOv5 X5 九个已发布源模型 URL 已从固定源 README 补回活动 `docs/release/x5/models.yaml`；S100/S600 两资产。逐 URL 源证据在上述 JSON。YOLOv5 不支持 S100P；ByteTrack 通过它自己的三个 S manifest 资产支持 S100/S100P/S600，不可混淆。
- 保留 X5 packed NV12/640、S split NV12/672、X5 默认 n-v7.0/stretch、S x-672/letterbox。S 先按 metadata 反量化，**YOLOv5 两侧均继续 sigmoid logits**；旧计划的“S 已激活”概括不能机械套在这里。
- X5 Python 原实现将 XYXY 传入 OpenCV NMSBoxes 且不分 class 的历史行为刻意保留以便源对照；S 使用按类 XYXY NMS。不能把这种保留描述成数学修正。
- 有意修复：S 显式 score/NMS=0 不再被 `or default` 吞掉；geometry 用冻结 per-call context；ByteTrack 三阶段只更新一次；frame_rate 真正参与 buffer 缩放；零面积 person 不再使 Kalman XYAH 除零。旧源 NaN 的 fixture 捕获会如实失败留证。
- ByteTrack 真实 CPU tracker 五源文件已迁移，只有 matching 的相对 import 改动，见 `runtime/python/TRACKER_SOURCE_MAP.json`。S YOLOv5 与源 ByteTrack 内置 detector 只差文件末尾换行，已核实共享合法。
- ByteTrack evaluator 用两个新进程，严格比较 track IDs，不能用 ID 偏移豁免板测；只有同进程 CPU 单测为消除源全局计数器历史而规范化初始 ID 偏移。
- 主机 `.venv` 已安装 `lap==0.5.12`、`cython-bbox==0.1.5`（构建依赖 Cython）。这是包源访问，未下载模型；不要把本轮说成完全未联网。版本/安装日志已存 evidence。
- **暂停前刚生成 YOLOv5/ByteTrack 共二十份 README，尚未 root 复核或最终 checker**。先检查真实参数/API、完整历史表和命令。已看到 ByteTrack runtime README shell 续行有字面双反斜杠，需要校正并验证；YOLOv5 evaluator 中“在控制板卡的主机运行”应明确实际在目标板上执行。

### YOLOv5 C++：仍有阻断，不能以 9 个 host tests 当作 native 通过

初版拆了 X5 HB-DNN / S UCP adapter、纯 decoder、visualizer、launcher、CMake。[作者证据](evidence/2026-09-23-b7-yolov5-cpp-host.json) 记载 9 项 host tests 和 portable decoder 编译，**真实 SDK 编译/运行均 not-run**。

已修首轮问题：S guard 改为拥有 vectors；X5 模型数量/F32/维度检查和 allocation/flush 返回值；每 anchor 只选 argmax；CMake 加 SOC_S600、nn_math.cpp、OpenCV include；label-file 真正用于渲染；launcher 参数和模型文件校验。

以下第二轮意见发出后被用户暂停，**不要假定已修**，逐文件核对：

1. X5 alignedByteSize 仅检查 >0，仍可能不足以读取 `count*sizeof(float)`；alignedShape/padded layout 未明确拒绝或处理。紧凑 NV12 memcpy 也要证明实际存储布局。不能靠 shape 看起来正确就强转指针。
2. S `dequantizeTensorS32` 按 `scale[c]`/int32 或 float 读内存，尚缺 native dtype、scale 长度、stride/容量 gate；不要支持未知布局后强读。guard 只释放成功分配的资源，避免对未分配 sysMem 盲 free。
3. 固定 X5 C++ 源默认 **s-v2.0**，当前 launcher 仍沿用 Python **n-v7.0**；应在 variant/asset-id 都未给出时保留 C++ 源默认并同步文档/测试。X5 源按 class NMS 还有候选 top_k=300 和 OpenCV 严格 score 边界；S 行为不同，当前 decoder 未完整保留。
4. X5 CLI 接收 priority/bpu-core，但 infer ctrl 尚未应用；应按有证据的 SDK 能力实现或明确拒绝不支持值，不能静默忽略。S100/S600 build 宏影响 alignment，native binary 要拒绝与构建目标不一致的 target。
5. 当前只有渲染图，尚缺机器可比的 C++ native inputs/raw/results dump。补显式 `--dump-dir` 或等效路径，部署/模型/metadata/参数/返回码能绑定；dump 和可视化应独立模块。SDK/ABI/缓存/真数值仍需板端复验。
6. 初版 tests 有较多代码 substring 断言；补与上述行为相关的 portable 数值/边界或模拟资源测试，不能用“文件中出现 RAII/校验名称”充当行为证明。

### FCOS

初版独立审查见 [专项报告](2026-09-23-b7-fcos-independent-review.md)：manifest 下载 alias 缺失、依赖本机 .venv、letterbox 逆几何不完整、evaluator 证据不足等。作者已整改，暂停时 [最新作者 evidence](evidence/2026-09-23-b7-fcos-host.json) 记录 **23 tests passed**。root 尚未完成独立复审。

接手须核查：三个 variant、15 输出 shape/quant 绑定；letterbox A/B/A；下载 alias 和纯 python3 shell；完整 source/unified evaluator。source helper 已改为固定 X5 路径，数组相对路径也改过；检查 source baseline 是否实际调用原函数、文件 hash/非空 gate、实际 metadata/全部输入 raw 及失败日志是否完整。F32 携带 SCALE 不能凭经验去掉反量化，须按源语义确认。原独立报告结论保留，关闭时追加整改确认，不覆盖历史。

### LPRNet / MODNet：中途整改状态，旧测试结果不能代表暂停后的树

原初版 14 项主机测试和二十份 README 已落地，但独立复现出了 [初审反例](evidence/2026-09-23-b7-independent-initial-probes.json)：直接 runner API 缺 target/file gate、伪造 publication row 被接受、直接脚本入口失败、MODNet dry-run 接受不支持 ref-size、evaluator 只有手工文件比较。

暂停时已经看到部分修改：runner 加 gate/verify_asset_file，binding 开始重校验 publication，main 加 root bootstrap，MODNet dry-run 拒绝非512，LPR 新增 tensor_io，MODNet 新增 visualization。**这些是中途改动，未复跑或独立确认**。原 [14-test evidence](evidence/2026-09-23-b7-lprnet-modnet-host.json) 应保留为旧快照，不能改写为最新通过。

未完成重点：统一参数边界（finite、空 core list、非法 priority/ref-size）、实际 auto 身份处理、exact 输出名/shape/dtype、输入/输出有限值、公开 API/README/tests 与拆分同步；两 evaluator 仍是手工 raw/matte 比较，需要自包含同板 source→unified 捕获、全部数组、代码/模型/输入 digest、metadata、UTC/argv/cwd/rc 和失败记录。

源数值审查已确认 LPR 68 字符 CTC 与 MODNet normalize/padding/geometry 正常路径保持一致。LPR 输入是源预打包 float32 `.dat`，不要发明图像 resize；MODNet manual 模型 URL/hash 缺失应保留事实。

### YOLOWorld

初版十份 README、代码与 10 项作者测试已落地，但整改尚未闭环。初版 evidence 位于根目录 [evidence/2026-09-23-b7-yoloworld-host.json](../../../evidence/2026-09-23-b7-yoloworld-host.json)，与其他批次路径不一致；后续若迁移需同步所有引用。

待修：runner verify_asset_file；direct binding publication/metadata 校验和输出形状锁定；finite 边界；vocabulary 从调用方 ndarray 做只读快照。evaluator 目前缺 native inputs、完整部署 hash/metadata/UTC/argv/cwd/失败 JSON，源 helper 导入依赖环境；改为可执行完整对照并加真实源函数+fake SDK 正反 fixture。保持 32-slot prompt、最后一个 prompt 填槽、空/溢出 prompt 拒绝与 per-call scale/ID context。

### B7 整合任务

所有整改经独立复核后，再运行全量相关测试和 migration checker，更新双语 sample 索引、台账、计划、批次报告、代码快照和 board handoff 精确命令。现在不存在 B7 全批 green 或完整工作树回归记录。B7 源清点 136 文件应复核仍未改动；[B8 的250文件只读清点](evidence/2026-09-23-b8-source-inventory.json) 也不算 B8 开发开始或验收。

## 接手后的验证方式（仅供下一位执行，本次暂停后未运行）

在实际 Git 根使用 `.venv/bin/python`。先读暂停快照及修改中的文件，再跑定向测试，避免看到旧 evidence 就跳到 B8。

```bash
.venv/bin/python -m unittest discover -s samples/vision/yolov5/tests -v
.venv/bin/python -m unittest discover -s samples/vision/bytetrack/tests -v
.venv/bin/python -m unittest discover -s samples/vision/fcos/tests -v
.venv/bin/python -m unittest discover -s samples/vision/lprnet/tests -v
.venv/bin/python -m unittest discover -s samples/vision/modnet/tests -v
.venv/bin/python -m unittest discover -s samples/vision/yoloworld/tests -v
.venv/bin/python -m unittest discover -s samples/_shared/tests -v
.venv/bin/python -m unittest discover -s tools/sample_contract/tests -v
.venv/bin/python tools/sample_contract/check.py --scope migration --parser-mode import \
  --exemptions tools/sample_contract/baselines/ultralytics-readme-debt.json \
  --report /tmp/b7-resume-contract.json
git diff --check
```

之后按实际影响补全既有 sample 回归。checker 只验证可机器判定规则，不能替代 README 命令/API 数值与职责审查；相关 NMS/CTC/几何 helper 允许保留，规则不是“整个文件只能四函数”。

## 后续批次特别容易漏的事项

- B8 的 unet 与 unetmobilenet 保持两个 sample；PP-LiteSeg 原生已是 int32 类别图，不能套 logits argmax；DiffusionDrive 是四输入轨迹/BEV任务并有 quant 输入，不能硬套单图 NV12。yolo26_depth 标准/lite 及 X5/S 输出含义不同，29 个 S 非 README conversion 文件应逐项保留；本次只有只读清点。
- B9 不覆盖现有 ultralytics_yolo，按输出协议收编。先裁定 S cls 文件名真伪；清掉 `tools/sample_contract/baselines/ultralytics-readme-debt.json` 和 CI 的 `--exemptions`。yoloe26_seg 取固定 S tip。
- B10 为首次音频/观测序列任务，CLI/API、输入准备和状态语义按任务写，不能复制图片 sample 模板内容。
- B11 minicpm5 取 tip legacy evaluator/results；VLA 只改 gitlink/.gitmodules 路径并保持 pinned SHA `326ea043`/`a32de276`，不 vendor 子模块内容。
- 最终整合须处理根双语 README、Guidelines、CLAUDE.md、registry、VERSION/CHANGELOG/ADR、skills 引用/行为校验与源增量核对。保留 board not-run 的精确能力矩阵；在实际证据满足前不要删除 evaluator 所依赖的源目录或宣布客户可发布。

本交接写完后 Codex 停止工作，等待用户在另一个 Claude Code 会话安排后续。
