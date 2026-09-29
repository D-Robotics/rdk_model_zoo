# B3 板测对照工具本地准备作者报告（2026-09-24）

本报告记录 B3 历史板测待办的**本地工具准备**交付。作者只做工具与主机验证；板卡已恢复，实际板测由 Codex 执行。本任务**未执行任何板端推理**、未下载模型、未 SSH、未 commit/push/merge，未进入 B8，未修改任何 sample 算法、runtime 代码、模型 manifest、客户 README 或台账。

**状态口径（不得被本报告改写）：**

- B3 **Board = not-run；Closed = no**；本工具的存在不构成板测证据，主机测试不替代板测。
- 四个 sample evaluator README 的 board comparison 行仍为 not-run；等待 Codex 板端执行后按证据更新，本报告不代填。
- 完成后交 Codex 独立审查；本报告为作者自述，不冒充独立评审。

基点：独立 worktree `codex/b3-glm-board-capture-20260924` @ `6adb0b4`，起点工作树 clean。变更仅为新增 `tools/board_validation/`（未跟踪目录，`git status` 无其他改动）。

## 1. 任务与审阅依据

[B3 板端待办](2026-09-22-host-development-and-board-handoff.md)要求：X5 8GB/4GB，ConvNeXt atto；EdgeNeXt base/small/x_small/xx_small；FasterNet s/t0/t1/t2；FastViT s12/sa12/t12/t8（13 变体/板），同板同模型同输入记录固定源与统一入口，完整 Top-K/分数与部署哈希；默认入口与明确变体均验证。四个 sample evaluator README 已写有"源任务 API 对照"步骤，但没有持久化完整身份/数据的可执行器。

实现前先审阅了实际两侧 API 与既有先例，避免重复造轮子：

- **源侧**（`platforms/x5/samples/vision/<sample>/runtime/python/<sample>.py`）：四个 sample 同构，`{Prefix}Config(model_path, label_file, resize_type=1, topk=5)` + `{Prefix}` 类，scipy softmax + `np.argsort` 后处理；`main.py` 默认 priority=0、bpu-cores=[0]，`run.sh` 从 sample 目录内运行（`utils.py_utils` 由此解析）。
- **统一侧**：`resolve_selection` → `RuntimeModelRunner`（惰性 SDK、绑定表校验、raw_f32/形状/dtype gate）→ `ClassificationTask`（stable softmax、float64 计算后 float32 输出）。
- **先例**：`samples/_shared/sam_evaluator.py`（SDK run 层 `_RecordingRuntime` 记录 native 输入/输出、`metadata_evidence` 投影 SDK 描述符、全新证据目录、rc 0/1/2 语义、`_load_legacy` 桥接 `hbm_runtime`）；`lprnet/evaluator/compare.py` + `source_reference.py`（单模型源/统一对照、publisher/observed SHA 分列、`verify_asset_file`）；`modnet/evaluator/source_reference.py`（legacy `utils.py_utils` 依赖以观察到的解析路径记录）。B2 板测先例（ids 全等 + 分数 |差|<1e-5，top-8 per-ID 平局裁定）与 B3 更严指令（平局不得自动放行）共同构成本工具判据；B2 评审明确"gap=0 的裁定不外推为未来近似平局自动放行"，故本工具对平局只记录、不裁定。

## 2. 交付物

`tools/board_validation/`（新增目录，遵循 `tools/sample_contract/` 的 script+tests+README 布局）：

| 文件 | 内容 |
| --- | --- |
| `b3_classification_compare.py` | 一次一个 sample/target/variant 的同板源/统一对照 CLI（927 行，Python 3.10 兼容） |
| `tests/test_b3_classification_compare.py` | 主机 fake-runtime 行为测试（22 项，不引入推理 SDK） |
| `README.md` | 最小依赖、逐 sample 命令、输出与限制说明 |

工具行为要点（全部有对应测试或先例出处）：

- **固定源**：X5 pin `ac115717197920355fc390bb04299b20e6436864` 写入每份证据；执行 `platforms/x5/.../<sample>.py` 原模块（不改源码），记录其 SHA-256 与 **实际解析到** 的 `utils.py_utils.file_io/preprocess` 文件路径+哈希（modnet 先例：仓库根解析、按观察值留证）。
- **两侧各自完整执行**：源侧 `Config → 模型 → set_scheduling_params → pre_process → forward → post_process`，统一侧 `resolve_selection → RuntimeModelRunner.load → set_scheduling_params → ClassificationTask.pre/forward/post`；同一图像字节、同 resize/topk/调度；SDK `run()` 层记录 native 输入/raw 输出，无一侧结果代另一侧（测试断言两个独立 runtime 实例各 run 一次）。
- **目标身份精确 gate**：入口即 `require_execution_target`；非 X5 目标按 sample 语义返回 "No published sample asset"（s100 负例测试）；单一资产绝不隐式默认未知 target；`--variant` 走 sample 已发布变体精确选择，省略时按 sample 默认语义（atto/base/s/s12）；`--model-path` 必须配精确 `--asset-id`。
- **判据**：前处理输入 **exact bytes**（源侧 `(1,3H/2,W,1)` 视图与统一侧 flat 缓冲逐字节比对，形状分记）；raw 输出同 shape/dtype/finite 且**报告完整差值**（`raw_abs_diff_*.npy` + max/mean/nonzero/argmax；raw 相等按 B3 evaluator 口径报告不判定）；Top-K **IDs 必须全等**、逐 ID 分数 **abs≤1e-5**；任一侧 top-(k+1) 窗口内出现**精确平局**即 `passed=false`，双侧 top-8 per-ID 分数完整保留供独立裁定，绝不自动放宽或把 tie 记 pass。
- **证据**：全新目录（已存在即拒绝）；UTC 起止、argv/cwd、git head/branch/dirty、代码 SHA-256（工具+源模块+观察依赖+shared+sample runtime）、board 身份四文件 + `/etc/os-release` + MemTotal（区分 8GB/4GB）、SDK 模块文件/版本、模型 publisher/observed SHA（manifest 为 null 则保持 null）、图与 labels observedSHA、两侧 SDK metadata（复用 `metadata_evidence`，禁止 asdict——测试含拒绝拷贝的 QuantParams 替身）、真实异常与 rc、所有输入/raw/结果/证据数组 `.npy` + shape/dtype/SHA-256。哈希复用 `samples/_shared/assets.py::sha256_file`（3.10 分块实现），并加 `ast feature_version=(3,10)` 语法门测试。
- **rc 语义**：0 全部通过；1 比较完成但判据失败（数组保留）；2 执行异常（错误证据保留）。stdout 输出单条机器可读 JSON（checks/tie/双侧 Top-K/证据路径）。
- **范围公开**：证据内 `measurement` 字段与 README 均声明"固定图迁移一致性，非数据集精度/时延"。

## 3. 验证（CI 同款命令，仓库 `.venv`：Python 3.14.7 / NumPy 2.5.3 / OpenCV 4.14.0 / scipy 1.18.1）

| 步骤 | 命令 | 结果 |
| --- | --- | --- |
| 新工具行为测试 | `python -m unittest discover -s tools/board_validation/tests -v` | **22 tests OK** |
| 四 sample 套件 | `python -m unittest discover -s samples/vision/{convnext,edgenext,fasternet,fastvit}/tests` | 28/26/28/28 = **110 OK**（与 B3 主机复核同数） |
| 共享套件 | `python -m unittest discover -s samples/_shared/tests` | **124 OK** |
| migration checker（CI 同参数） | `python tools/sample_contract/check.py --scope migration --parser-mode import --exemptions tools/sample_contract/baselines/ultralytics-readme-debt.json --report /tmp/…` | **36 samples, 0 violations, 39 skips, 84 exemptions, rc=0**（与基线一致） |

22 项测试覆盖：四 sample 全协议通过（含双侧独立执行、调度参数一致、输入逐字节一致、legacy 视图/统一 flat 形状分记）；变体/资产精确选择（small/sa12/t0）与默认语义；未知变体、`--model-path` 无 `--asset-id`、错误目标（s100）的精确拒绝；模型缺失时错误证据保留；两侧 raw 形状/源侧 dtype 不一致判 rc=1；统一侧非 F32 被 binding 拒绝判 rc=2；分数超差、ID 失配（含 per-ID absent 标注）判 rc=1；**精确平局保留 top-8 证据且不记 pass**；统一侧执行异常保留部分证据；证据目录复用拒绝；拒绝拷贝的 QuantParams 经 `metadata_evidence` 投影；纯函数规则（容差含 1e-5 边界、NaN 失败、平局窗口边界）；parser 默认与 3.10 语法门。

## 4. 未验证 / 未做（明确边界）

1. **没有任何板端运行**：工具在主机仅以注入的 fake SDK 验证协议；真实 `hbm_runtime`、真实制品、8GB/4GB 实板行为全部待 Codex 执行。host fixture 的 max_abs_diff=0 只证明对照逻辑，不是板端数值结论。
2. **没有下载任何模型**；`download.sh` 前置准备由 README 说明，工具自身不触网。
3. **未修改** sample 算法、`samples/_shared`、manifests、四个 sample 的任何文件；未更新客户 README 的板测状态与台账（等真实证据）。
4. **转换（OE export/校准/编译）** 与数据集精度/时延不在本工具范围，保持 not-run。
5. 平局裁定权在独立审查：工具只留证不裁决，也不沿用 B2 的 gap<1e-6 近似平局口径。
6. 本报告为作者自审；独立复核待 Codex。

## 5. Codex 板端执行交接（建议）

每矩阵单元（2 板 × 13 变体）一次调用、唯一证据目录、保留 rc/stdout/stderr，cwd=板端仓库根：

```bash
# 前置：逐 sample 显式下载（见 tools/board_validation/README.md），部署本 worktree 对应提交
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/tmp/b3-<sample>-<variant>-<board>-$STAMP   # board ∈ {x5-8g, x5-4g}
python3 tools/board_validation/b3_classification_compare.py \
  --sample <convnext|edgenext|fasternet|fastvit> --target x5 --variant <variant> \
  --output-dir "$OUT" > "$OUT.stdout" 2> "$OUT.stderr"; echo $? > "$OUT.rc"
```

判定：`rc=0` 且 `comparison.json` 全部 checks=true 记为通过；`rc=1` 时按 tie/per-ID 证据独立裁定；`rc=2` 为执行失败按错误记录处理，不重试伪装。默认入口（省略 `--variant`）建议每 sample 至少一次，与显式变体并行留证。执行后按证据更新各 sample evaluator README 的 board comparison 行与批次台账；B3 关闭仍以独立审查结论为准。
