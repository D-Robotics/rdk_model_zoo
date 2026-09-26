# X5/S Sample Integration Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development; execute and independently review each bounded task. Do not confuse host tests with board acceptance.

**Goal:** Implement the supplied X5/S architecture in verified batches, preserving source capabilities and historical evidence.

**Architecture:** One task pipeline receives a callable model runner. Explicit bindings and local tensor adapters carry actual artifact differences. Shared code requires two real consumers.

**Tech Stack:** Existing Python/NumPy/OpenCV, board hbm_runtime, YAML manifests, existing C++ and TypeScript publisher.

**Spec:** [Approved implementation input](../specs/2026-09-16-rdk-model-zoo-x5-s-agent-people-spec.md).

## Current delivery scope

The user narrowed delivery to YOLO detection, one PaddleOCR sample and ResNet18
classification. Runtime checks for these representatives are recorded in the [acceptance report](../../releases/unified-migration/2026-09-16-p2-validation.md).
Unchecked broader phases below are deferred, not required for this delivery.
The 2026-09-17 user review rejected completion of sample source/docs/conversion
integration; [the correction plan](2026-09-17-representative-integration.md)
now governs that work.

## Global Constraints

- Targets: X5, S100, S100P, S600; `s` is a data group, not an execution target. No new X3 work.
- Preserve existing IDs, URLs, hashes, historical evidence, gitlinks, conversion, evaluation and C++ capabilities.
- No per-Sample workflow configuration, global runtime base class, global execution CLI, or required Agent dependency.
- No SDK import, network, model loading, output creation in help/list/dry-run or library import.
- Runtime changes require affected-target board evidence; absent boards are `not-run`.
- No push, tag, release, online deployment or Hub source switch in this implementation session.

## Decisions and baseline

The supplied Spec supersedes the earlier X3-inclusive plan. The integration workspace is the existing clean linked worktree `_worktrees/catalog-sample-audit`, branch `develop`, initial commit `cd74a2b241075bb21036d8d0855d0403f8e8c963`. Other repositories have unrelated dirty state and are read-only sources. Existing unified YOLO host baseline: 31 tests passed with Python 3.13.5 on Windows. The user subsequently supplied five boards (X5 8GB/4GB, S100, S100P, S600). Their environments, exact published model metadata and original-source inference baselines are recorded under `docs/releases/unified-migration/evidence/`; new-source comparisons are a separate acceptance step.

## Task 0 — Source and capability inventory

- [x] Record complete source SHAs, source directories, gitlinks, notebooks, manifests and existing migration work in `docs/releases/unified-migration/2026-09-16-baseline.md` and `x5-s-migration-map.md`.
- [x] Distinguish source presence, artifact availability, host checks and board validation. Record unavailable external dependencies.

## Task 1 — Shared target identity with two pilot consumers

**Files:** `samples/_shared/platforms.py`, `samples/_shared/tests/test_platforms.py`; YOLO platform/entry glue; new ResNet entry.

**Interface:** `resolve_target(requested='auto', *, soc_name=None, board_type=None) -> str`, `detect_target() -> Optional[str]`, `require_execution_target(requested: str) -> str`.

- [x] Write tests for exact known identity, unknown strings, ambiguous group `s`, explicit preparation on host and execution mismatch/missing identity. First run must fail without the module.
- [x] Implement dependency-free identity reading with exact known aliases; use the existing board-info locations and recorded aliases. X5U was observed on both available X5 boards; X5H/X5M are user-supplied aliases and remain unobserved here.
- [x] Both pilots consume it; retain YOLO public platform profiles as compatibility values, not another detector.
- [x] Run `python -m unittest discover -s samples/_shared/tests` and existing YOLO tests.

## Task 2 — Real ResNet classification pilot

**Files:** `samples/vision/resnet/runtime/python/`, bilingual README and tests. Sources are `platforms/x5/samples/vision/resnet` and `platforms/s/samples/vision/resnet18`.

- [x] Read both implementations, conversion metadata and manifests; identify source postprocessing before sharing it. Retain the explicit legacy-softmax score policy; graph-level X5 output semantics remain unverified and recorded separately.
- [x] Write a fixed-logit top-k fixture and negative metadata/input tests, run red, then implement typed binding, callable runner and one classification task.
- [x] Add a no-SDK entry for help/list/dry-run; use existing manifest facts, preserve concrete target boundaries and no automatic download in inference.
- [x] Document old/new symbols, dependencies, inputs/outputs and untested targets. Keep existing conversion/evaluation/C++ and old paths intact until compatibility integration is verified.
- [x] Run the new host suite and entry commands from root and an unrelated working directory.

## Task 3 — YOLO detection callable boundary

**Files:** Existing `yolo_detect.py`, new local binding/runner/tensor or geometry modules where needed, `tests/test_detection_binding.py`, `DETECTION_CONTRACT.md`.

- [x] Write deterministic DFL and negative role/shape tests and a non-square/padding geometry fixture; run red.
- [x] Separate model loading/validated I/O from the single task flow. Preserve current constructor compatibility and tuple-compatible result where possible.
- [x] Record actual resized dimensions and padding; use independent X/Y grid geometry. Bind explicit source-proven output roles; reject unknown quantization/protocol facts.
- [x] Verify old tests and new fixtures; document affected consumers and numerical changes requiring board comparison.

## Task 4 — Independent acceptance of pilot changes

- [x] Review code and spec compliance separately from implementer reports; fix concrete defects.
- [x] Record exact host commands and outcomes under `docs/releases/unified-migration/`; record board executions as `not-run` until authorized environments are available.
- [x] Use available board configurations only after the user supplies them; retain exact input/artifact identities and old/new output comparison.

## Subsequent batches and gates

P2/P3 batches depend on the actual Task 0 inventory and P1 contracts. Their per-function plans must be written after reading their code; do not prefill fictitious mappings. P4 preserves publisher as a whole project and updates path consumers atomically. P5 requires locating and reviewing the actual seven-skill source package, not recreating a claimed import. P6 prepares but does not publish. All unmet D01–D14 items remain open in the evidence report; this plan does not shrink the supplied Spec to the pilots.

## P1 checkpoint

Completed scoped validation: [P1 report](../../releases/unified-migration/2026-09-16-pilot-validation.md)
and [independent/standards review](../../releases/unified-migration/2026-09-16-p1-review.md).
87 host tests passed; YOLOv8n compared on all five supplied boards, ResNet18 on
X5 8GB/4GB, S100 and S600. S100P classification not run (no published asset).
Continue with [P2 protocol and composition work](2026-09-16-p2-protocols.md).

## 2026-09-24 恢复执行：本地 GLM 开发、GitHub 同步、真实板端复验

本节为当前安排，取代历史暂停及“不连接板卡/不提交”的执行限制，保留历史结论。用户已授权本地 Claude Code + GLM 开发，Codex 独立评审并负责每轮 commit/push，通过 GitHub 固定提交在板上取回；不再把开发派给 HP。五个既有板卡地址现可达，恢复真实 SDK/制品对照。未执行的板测仍为 not-run，不能由主机结果代替。

B7 仍 changes-required / Closed=no。dtype 单项已经独立实板确认并合入 develop；metadata 核心与 native 整改在隔离分支集成复验。已实际通过的限定 case：YOLOv5 Python X5 8GB s-v2、S100 x-672，YOLOWorld X5 dog，以及 ByteTrack S100 四帧合成运动源对照。LPRNet/FCOS 首次板测失败已留证并交 GLM 修复；native 两类 SDK 构建/完整数值对照继续进行；MODNet manual 制品缺口尚未解决。权威动态记录为仓库 `docs/releases/unified-migration/2026-09-24-b7-native-sdk-review.md` 与迁移台账；不把 feature 分支成果说成已全部进入 develop。

下一步仍包括：B7 修复与独立闭环；B3–B6 已主机通过项的真实板端验证；B8、B9、B10、B11；最终源增量核对、README/Agent 规范与路径整合及全仓验收。不能因当前几个 case 通过而缩小原迁移目标。B8 继续保持 pending，先解决 B7 阻断；跨样例共享缺陷同步排查已迁移消费者。

## 2026-09-24 后续板测与整改检查点

B7：X5 YOLOv5九变体×两种内存、FCOS三变体×两板、LPRNet/YOLOWorld两板、ByteTrack S100/S600真实视频前30帧均有完整源对照证据。ByteTrack720数组引用按SHA去重保存270份，已入develop；不代表整段视频或数据集指标。四个样例README复审通过提交8474641（集成b04b6fc），仍在整改分支。C++观察/比较工具仍有独立反例未闭环，B7不关闭；MODNet缺manual模型，ByteTrack S100P源URL404，不跨target替代制品。

B6：真实板测发现Python3.10摘要API与CLI异常退出码缺陷（B6-B1）；修复1bfd8fa经独立主机复核后，通过GitHub f888c8f送板。S100两SAM源对照通过；X5统一runner调度scalar与真实SDK Mapping协议冲突（B6-B2），已交本地GLM。源X5 helper吞掉同类TypeError也须在evaluator控制记录中透明处理，不能伪称source调度成功。修复后继续X5及S100受影响范围；其余目标待测。

B3：本地GLM正在整理convnext/edgenext/fasternet/fastvit共13变体的可复用板端完整证据捕获工具，先独立审核再通过GitHub送两块X5。B4/B5剩余板测、B6剩余目标、B8–B11与最终整体验收均保留。目录inventory旧595断言与当前603的差异另行核对精确新增集合，不降级断言。开发继续使用本地Claude Code+GLM；Codex负责独立复审、板测与GitHub同步。


### 2026-09-24 18:17 复验进展

- B6-B2：GitHub检查点a72f92b完成X5 8GB默认/priority7及S100默认两sample共6次源对照，90数组与每例14代码摘要独立核验通过；B6其他目标仍待测，Closed=no。
- 目录清单测试修正b2e24c0已合入develop，完整Node22检查120项通过；不代表B7关闭。
- B7原生C++对照工具第三轮身份/审计整改由本地Claude Code+GLM继续，B3取证工具仍在开发。B8及后续范围未缩减。


### 2026-09-24 网络中断后恢复顺序

新增仓库恢复入口 `docs/releases/unified-migration/2026-09-24-board-resume.md`。X5 4GB SAM日志rc0但数组待回收；S100两套原生C++编译成功但数值未跑；S600/S100P准备未完成。先检查远端存活任务再恢复，不重复执行。B3固定源依赖与原生audit持久化继续本地GLM整改。B8–B11及完整验收范围不变。
