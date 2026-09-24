# B7 整改进度记录（作者侧，执行中）

本文件是 [GitHub 协作记录](2026-09-23-github-coordination.md) 第 15 节「B7 整改」的作者侧落盘进度。只记录本轮的实现、验证和未完成项；**不覆盖**独立评审历史结论（[B7 独立评审](2026-09-23-b7-independent-host-review.md)、[FCOS 专项](2026-09-23-b7-fcos-independent-review.md) 的 changes-required 与 Closed=no 保持原样）。板端一律 not-run，未做任何 commit/push/merge。

## 环境与基点

- 工作区：`/home/xgs/work/rdk-b7-remediation-20260923`（独立 worktree）
- 分支：`codex/b7-remediation-20260923`，基点 `d28d8c9b042e2b92823c8b57f77b80c2a4ac0fb2`（= 协调记录所述检查点）
- 固定源不变：X5 `ac115717197920355fc390bb04299b20e6436864`；S `380e1a2bf42041af54be6f34935e50197cfadff9`
- 主机解释器：本轮新建隔离 `.venv`（Python 3.12.14，numpy 2.5.3 / opencv 5.0.0 / scipy 1.18.1 / pyyaml 6.0.3 / lap 0.5.12 / cython-bbox 0.1.5）。仅访问包源，未下载模型、未加载 SDK、未接板卡。

## 阶段进度

| 阶段 | 范围 | 状态 |
| --- | --- | --- |
| R0 | 基线与 checker 复现 | 已完成 |
| R1 | YOLOv5 C++ 六项阻断 | 已完成 |
| R2 | YOLOv5 C++ 双语 README（14 项 checker 违规） | 已完成 |
| R3 | FCOS 四项独立发现 | 已完成 |
| R4 | LPRNet / MODNet（B7-LM1..3） | 已完成 |
| R5 | YOLOWorld（B7-YW1..2） | 已完成 |
| R6 | B7 README 全量复核 | 已完成 |
| R7 | 回归 + CI 同命令 checker + 证据 | 已完成 |
| R8 | 作者整改报告/台账 | 已完成 |

## R0 基线复现（2026-09-23）

- checker（CI 同命令）：`.venv/bin/python tools/sample_contract/check.py --scope migration --parser-mode import --exemptions tools/sample_contract/baselines/ultralytics-readme-debt.json --report /tmp/b7-recon-contract.json` → **rc=1，36 samples / 14 violations / 84 豁免**。14 项违规全部是 `samples/vision/yolov5/runtime/cpp/README.md` 与 `README_cn.md` 各缺 7 个必需 section anchor（supported-boards、dependencies、build、run、parameters、interface-lifecycle、results-interpretation）。未新增豁免、未降级规则。
  - 注：缺 numpy/cv2 时 checker 会 skip `R-CLI-DEFAULTS`；装上依赖后 skip 由 49 降到 39，违规数仍为 14，说明被 skip 掩盖的只是跳过项，不是违规。
- 主机测试基线（`unittest discover`，全绿）：

| 套件 | 结果 |
| --- | --- |
| samples/vision/yolov5 | 24 OK |
| samples/vision/bytetrack | 11 OK |
| samples/vision/fcos | 23 OK |
| samples/vision/lprnet | 7 OK |
| samples/vision/modnet | 7 OK |
| samples/vision/yoloworld | 10 OK |
| samples/_shared | 101 OK |
| tools/sample_contract | 27 OK |

- 板端：全部 **not-run**（本机无 X5/S 板卡、无 SDK、无模型资产）。

完整环境/包版本/命令返回码见 [evidence/2026-09-23-b7-remediation-recon.json](evidence/2026-09-23-b7-remediation-recon.json)。

## R1 YOLOv5 C++ 六项阻断（2026-09-23）

新增与 SDK 无关、主机可编译可跑的三个模块，阻断项逐一对应：

- `runtime/cpp/include|src/yolov5_gate.{hpp,cpp}` — 张量元数据 gate：X5 紧凑 NV12 输入（拒绝带 padding 的 aligned 布局/不足分配）、X5 F32 NONE 头（容量 `>= h*w*c*4`、stride 唯一 8/16/32、拒绝 aligned padding）、S split NV12 平面（行 stride 与分配覆盖）、S 反量化前的 native dtype/scale 长度/连续 NHWC stride/容量、以及「编译目标身份 == `--target`」。对应阻断 1、2、4。
- `runtime/cpp/include|src/yolov5_dump.{hpp,cpp}` — 机器可比 dump（含自实现 SHA-256），与可视化模块独立；`--dump-dir` 写出 `manifest.json`（deployment/model/image SHA-256、输入/输出 metadata、参数、UTC/argv/cwd/rc）与逐张量原始文件；失败也写 rc/error。对应阻断 5。
- `yolov5_decode.{hpp,cpp}` — 新增 `DecodePolicy`：X5 保留源 `cv::dnn::NMSBoxes` 语义（严格 `>` 边界、每类 top_k=300），S 保留源 `nms_bboxes` 语义（`>=`、无上限）。对应阻断 3。
- 两个 adapter：X5 非默认 `--priority/--bpu-core` 明确拒绝（无经验证 HB-DNN 映射），S 应用调用方调度；S guard 只释放真正分配成功的张量。对应阻断 4。
- `launcher.py`：X5 在未给 variant/asset-id 时采用固定 C++ 源默认 `s-v2.0`（不再沿用 Python `n-v7.0`）。
- 测试 `tests/cpp/portable_checks.cpp` + 重写 `test_cpp_contract.py`：以真实元数据驱动 gate/解码边界/top_k/dump 行为，替换原先的代码 substring 断言。对应阻断 6。

未决/限制：本机无 OpenCV 开发头文件、无目标 SDK，两个 adapter TU **未编译**（板端 SDK/ABI 编译仍 not-run）；`alignedShape`、`scale.scaleLen` 等字段名取自 SDK 公开结构，需板端编译确认；带 padding 的 aligned 布局当前是「明确拒绝」，若板端实测确有 padding，正确做法是按 aligned stride 寻址（列为后续项）。

## R2 YOLOv5 C++ 双语 README 与 14 项 checker 违规（2026-09-23）

`runtime/cpp/README.md` 与 `README_cn.md` 按 `docs/sample-standards/templates/runtime-cpp.{en,zh}.md` 重写，补齐 supported-boards / dependencies / build / run / parameters / interface-lifecycle / results-interpretation 七个 anchor，并把 R1 的默认变体、NMS 边界、letterbox、调度与 dump 差异逐条写明。未加任何豁免、未降级 checker 规则。

验证：`check.py --scope migration` 由 **rc=1 / 14 violations** 变为 **rc=0 / 0 violations**，84 条 B9 豁免保持不变（见 R7 证据）。

## R3 FCOS 四项独立发现（2026-09-23）

逐项核对当前树，四项中三项此前已整改，一项补齐：

| 独立发现 | 当前状态 |
| --- | --- |
| P1 manifest 指向缺失 `download_model.sh` | 已存在；`download.sh`/`fulldownload.sh` 均为 `download_model.sh` 同源 wrapper，`test_download_alias_and_wrappers_are_portable` 覆盖 |
| P1 board 路径硬编码 `.venv/bin/python` | 三个脚本均改用 `python3`，测试断言不含 `.venv/bin/python` |
| P1 `--resize-type 1` letterbox 逆几何错误 | 已实现：`fcos.py` 按冻结 `ImageContext` 的 `resized_shape`/`pad` 反算并 clip；`_validate_context` 拒绝几何不一致 context；有非方图 A/B 测试 |
| P1 evaluator 只是手工配方 | 已实现：`evaluator/compare.py` 自行运行 source+unified，gates x5，捕获全部 15 路 raw/metadata/argv/cwd/rc/失败记录，单一 `EVIDENCE` 变量 |
| P2 F32 输出仍走反量化 | 保留源语义（源 `dequantize_tensor` 对 SCALE 描述符不区分 dtype），并**补齐另一半分支测试**：`test_non_scale_outputs_pass_through_in_both_helpers` 验证非 SCALE 描述符在统一与源 helper 中均为恒等 |

## R4 LPRNet / MODNet（B7-LM1..3）（2026-09-23）

- **B7-LM1**：两个 runner 的 `load()` 在构造真实 SDK factory 之前完成 `require_execution_target` + `verify_asset_file`（拒绝时不会调用 factory）；binding `bind_model` 重新解析 manifest 并比对完整发布事实（伪造同形 `evil.bin` publication row 会被拒）；输入/输出 exact shape/dtype/finite 锁定。（本轮前的实现里该 gate 对注入 factory 也生效，导致约定的 host injection seam 不可用，已与 FCOS 参考实现对齐为「仅真实路径 gate，显式注入 factory 才跳过」。）
- **B7-LM2**：两个 `main.py` 支持从任意 cwd 直接 `python /abs/main.py --help`；MODNet `--dry-run --ref-size 256` 返回 rc=2。另发现并修复 dry-run 不校验调度参数（`--priority 300/-1`、`--bpu-cores 0 -1` 之前 rc=0），现在 dry-run 与真实路径同样拒绝非法值。
- **B7-LM3**：两个 evaluator 从「手工对比文件」重写为自包含 `compare.py`：gates x5、同一模型/输入/参数下自行运行固定源与统一实现、经 Recorder 捕获输入与 raw 输出、记录 metadata/argv/cwd/UTC/return_code/失败 JSON、绑定模型/输入/代码 SHA-256、逐数组落 `.npy` 并各自记录摘要。新增 `evaluator/source_reference.py` 明确固定源加载 seam。双语 evaluator README 同步重写（保留 7 个模板 anchor）。

主机验证：LPRNet 12 tests OK、MODNet 12 tests OK（原各 7）；sample checker 0 violations。

## R5 YOLOWorld（B7-YW1..2）（2026-09-23）

- **YW2（源码层）**：`yoloworld.py` 词表改为只读自有快照（`np.array(copy=True)` + `writeable=False`），调用方事后改写自己的数组不再影响后续调用；`model_binding.bind_model` 重新解析 manifest 并比对发布事实（伪造同形 publication row 被拒）；`model_runner.load` 在真实路径补 `require_execution_target` + `verify_asset_file`（注入 factory 的 host seam 才跳过），并对 raw 输出补有限值校验。新增 4 项测试覆盖以上行为。
- **YW1（evaluator）**：`evaluator/compare.py` 重写为自包含对照：自行运行固定源与统一两侧，捕获预处理输入（image/text）、raw score/box 与最终检测；记录模型/图像/词表/代码 SHA-256、两边 runtime metadata、prompts、阈值、argv、cwd、UTC、return_code；失败也写 `error`+`return_code: 2`。新增 `evaluator/source_reference.py`：显式绑定固定平台 postprocess 模块，并安装含 `QuantParams` 的注入 stub，去掉对调用方环境 `utils` 导入的依赖。双语 evaluator README 同步重写。
- 宿主验证：yoloworld 18 tests OK（原 10）。

## R6 B7 双语 README 复核（2026-09-23）

- 修正 4 个 B7 README 中 19 处**字面双反斜杠 shell 续行**（`\\` → `\`），修后 `bash -n` 通过。
- 修正 YOLOv5 evaluator 中「在控制板卡的主机运行」的错误表述（中英同步）为「直接在目标板上运行，不是外部主机驱动板卡」。
- 机械交叉核对：6 个 B7 sample 的 24 份 README 中出现的每个 `` `--flag` `` 都存在于对应 `main.py`（0 处未知 flag）；文档 shape 与 binding 常量一致；所有 B7 目录 en/cn anchor 集合完全一致。
- C++ README 的 7 个 anchor 与内容在 R2 重写时一并完成。

## R7 回归与 CI 同命令 checker（2026-09-23）

- CI 同命令（`.github/workflows/sample-contract.yml`）：checker 单测 27 OK rc=0；migration checker **rc=0，36 samples / 0 violations / 39 skips / 84 豁免**；`git diff --check` rc=0。
- 全量主机回归：**906 tests / 38 个有效套件**。B7 与相关：yolov5 32、bytetrack 11、fcos 24、lprnet 12、modnet 12、yoloworld 18、_shared 101、contract 27 全 OK；B4/B5/B6 既有套件全 OK。
- 已知非本轮问题（原样报告，未修）：3dresnet 1 项失败（文件未改动，基点行为，B5 范围）；ultralytics_yolo 2 项需 Node 生成的 catalog，本机无 Node → not-run；catalog-publisher 定义 0 用例。
- 证据：[regression.json](evidence/2026-09-23-b7-remediation-regression.json)、[regression.log](evidence/2026-09-23-b7-remediation-regression.log)。
- **陷阱与回退**：首次更新台账时把 B7 的 Refactor 列写成 `作者整改完成（待独立复审）`，而 checker 的 scope 解析要求该列**精确**为 `in-progress`/`done`，会导致 B7 被悄悄移出检查范围（样本数 36→30）。已改回 `in-progress` 并复跑确认仍为 36 samples。

## R8 报告与台账（2026-09-23）

- 新增[作者整改报告](2026-09-23-b7-remediation-author-report.md)（逐项整改、验证、未完成项）。
- 台账 `x5-s-migration-map.md`：6 条 B7 行更新作者侧状态与证据链接（Refactor 保持 `in-progress`，Board 保持 `not-run`，评审列写明 **changes-required 保留、待 Codex 复审**，Closed 保持 `no`）；末尾新增本轮小节，历史暂停节原样保留。
- 根目录历史 evidence `evidence/2026-09-23-b7-yoloworld-host.json` **未移动**（移动会破坏 handoff 冻结快照中的路径记录），列为后续统一处理项。

## 总结：完成与未完成

**完成**：YOLOv5 C++ 六项阻断（含新增 gate/dump 模块与行为测试）、14 项 checker 违规清零、FCOS 四项、LPRNet/MODNet LM1–3、YOLOWorld YW1–2、B7 双语 README 复核、CI 同命令验证与全量回归、作者报告与台账。

**未完成 / not-run**：真实板端 source↔unified 对照；两个 C++ adapter TU 的本机编译（缺 OpenCV 开发头与目标 SDK）；`alignedShape`/`scale.*Len` 字段的板端确认；padded aligned 布局的处理策略；X5 调度参数若 SDK 支持则应改为应用；根目录 evidence 路径一致性；B8 未开始。独立评审仍 changes-required，Closed=no，delivery=not-ready。
