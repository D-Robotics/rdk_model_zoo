# B6 SAM evaluator Python 3.10 兼容修复（作者记录）

本记录覆盖 2026-09-24 的 B6 专项：修复共享 SAM evaluator 在真实板 Python 3.10 环境下的
`hashlib.file_digest` 兼容缺陷及其暴露的 CLI 退出码违约。任务由 GLM（Claude Code 会话）在
`rdk-b6-glm-python310` worktree（基线 `07f9a09`）执行，等独立复核；不复述、不改写其他报告的结论。

**状态口径：**

- 全程**未 commit / 未 push / 未 merge**；仅改动本任务专属的三个文件（见 §3）。
- **Board = not-run**：本任务不连板卡。板端复跑（X5 8GB / S100 的 `evaluator/compare.py`）留给协调方统一安排。
- 主机测试通过是作者自检，不等于独立验收；本记录**不宣称实板通过**，不关闭 B6/B7。

## 1. 缺陷与证据

板端初测（固定源 `73a6de1`，X5 8GB 与 S100，efficient_sam / mobile_sam 四组对拍；完整记录见
[原始两板记录](evidence/2026-09-24-b6-initial-board/)，只读引用）：

- 四组 `model/download.py --target ...` 全部 rc=0，真实模型对已按清单落地并算出 observed SHA-256；
  随后四组 `evaluator/compare.py` 全部以
  `AttributeError: module 'hashlib' has no attribute 'file_digest'` 失败，出错点
  `samples/_shared/sam_evaluator.py:32`（`_digest` → `hashlib.file_digest`），触发于
  `run_comparison` 首个资产摘要（line 202 `entry['observed_sha256'] = _digest(...)`）。

**缺陷一（Python 3.10 兼容）**：`hashlib.file_digest` 自 Python 3.11 才存在；两块板均为
Python 3.10。两个 evaluator README（en/zh）的 Environment 一节均承诺 *Python 3.10+*，
代码使用了 3.11+ API，与文档契约直接冲突。全 samples 检索确认 `file_digest` 仅此一处。

**缺陷二（CLI 退出码契约，同一异常传播问题）**：README 承诺 *exit 2 = target gating,
argument, model, image or execution failed*，且 `run_comparison` 落盘的
`comparison.json` 内 `return_code: 2`；但 `main()` 只捕获
`(OSError, ValueError, RuntimeError)`，重抛的 `AttributeError` 未被捕获，进程以
未处理 traceback + **外层 rc=1** 退出。板端四组记录均为 rc=1 + traceback，违反文档承诺。
（`comparison.json` 本身在 `finally` 中已完整写出并保留，证据未丢，丢的是退出码契约。）

## 2. 修复方案

按根因修复，不提高最低 Python 版本、不改变摘要算法/数值、比较标准、输入输出契约：

- **复用既有兼容实现**：`samples/_shared/assets.py` 的 `verify_asset_file` 原本就有
  1 MiB 分块 `hashlib.sha256()` 循环（板端 `download.py` 的 observed SHA-256 正是它算出的、
  在 Python 3.10 上已验证可用）。将其提取为模块级 `sha256_file(path)`（含
  `_SHA256_BLOCK_BYTES` 常量），`verify_asset_file` 改为调用它；行为逐字节不变。
- `samples/_shared/sam_evaluator.py` 的 `_digest` 改为委托 `sha256_file`，删除
  `hashlib.file_digest` 用法。摘要值与原实现完全一致（同为 SHA-256，FIPS 常量与
  全尺寸 oracle 对拍见 §4），`comparison.json` 各 sha256 字段语义与格式不变。
- **退出码契约**：`main()` 改为捕获 `Exception`（`SystemExit`/`KeyboardInterrupt` 仍自然穿透），
  stderr 打印 `error: <类型名>: <消息>`（原 `error: <消息>` 不含类型，板端排障困难），
  返回 2。`run_comparison` 的 evidence 落盘路径（失败写 error + return_code 后重抛）不变，
  任何执行期异常都保留完整 `comparison.json` 并以文档约定的 rc 退出。

## 3. 变更文件（仅此三个）

| 文件 | 变更 |
| --- | --- |
| `samples/_shared/assets.py` | 提取 `sha256_file`（含块大小常量）供复用；`verify_asset_file` 改用它（+17/−7 行，纯提取） |
| `samples/_shared/sam_evaluator.py` | `_digest` 委托 `sha256_file`；`main` 捕获 `Exception` → rc 2 + 带类型的 stderr（+4/−4 行） |
| `samples/_shared/tests/test_sam_evaluator.py` | 新增 `Python310FileDigestCompatTests` 5 个用例（见 §4） |

未改任何 SAM 样例文件、manifest、evaluator README（文档本就正确，是代码违约）、其他共享模块。

## 4. 测试与结果

新增回归先在未修复代码上运行并观察失败，再修复复跑。Python 3.10 环境以隐藏
`hashlib.file_digest` 属性的方式模拟（`_NoFileDigest` 上下文管理器；主机解释器为 3.14，
原生没有 3.10，此为可移植的等价模拟；摘要正确性另以 FIPS 180 常量与内存 oracle 锚定）。

新用例（均为行为断言）：

1. `test_digest_matches_known_sha256_without_file_digest` — 缺失 `file_digest` 时
   `_digest` 的真实摘要回归：空文件、`abc`（FIPS 180-2 已知常量）、以及
   0/1/1023/1024/1025/65535/65536/65537/1MiB−1/1MiB/1MiB+1/2MiB+17 字节确定性数据
   （覆盖 64 KiB 与 1 MiB 分块边界、跨块与多块+尾块；Path 与 str 入参交替），
   期望值 = 内存 `hashlib.sha256(同字节).hexdigest()`。
2. `test_board_cli_passes_without_file_digest` — 板端失败形态的端到端复现：
   在缺失 `file_digest` 环境下以 fixture 模型 + fake runtime 走真实 `main()` CLI
   （efficient_sam/s100、mobile_sam/x5），断言 rc=0、`comparison.json` passed、
   encoder/decoder observed_sha256 与已知摘要一致、code_sha256≥8。
3. `test_board_cli_reports_comparison_failure_as_one` — 运行完成但检查失败 → rc=1、
   JSON `return_code:1`、14 份数组保留、mask_equal False。
4. `test_cli_maps_execution_exception_to_error_code_two` — README 契约：执行期异常
   （以板端同款 `AttributeError` 模拟）→ rc=2 且 stderr 有明确 error 行，而非裸 traceback。
5. `test_execution_exception_after_capture_keeps_evidence` — 捕获阶段之后发生意外异常
   （legacy decoder run 抛 `AttributeError`，不经 runner 包装的原始类型）→ rc=2、
   JSON error.type/return_code 如实记录、已捕获数组保留。

**修复前（未修复代码 + 新测试，先证伪）**：`Ran 11 tests — FAILED (errors=18)`；
18 个 error 即用例 1 的 14 个 subTest（每个尺寸/常量逐一以板端同款
`AttributeError: module 'hashlib' has no attribute 'file_digest'` 失败，日志中共 17 处该错误行）
加用例 2 的两个 subTest、用例 4、用例 5（后两者的 traceback 均显示异常从 `main` 逃逸，
与板端 rc=1 形态一致）；用例 3 修复前即通过（该路径原本正确，属契约钉住）。

**修复后**（解释器 `/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk_model_zoo/.venv/bin/python`，
Python 3.14.7，numpy 2.5.3 / opencv 4.14.0 / pyyaml 6.0.3；cwd 为本 worktree 根）：

| 套件 / 命令 | 结果 |
| --- | --- |
| `unittest discover -s samples/_shared/tests -p test_sam_evaluator.py` | **Ran 11 tests — OK** |
| `unittest discover -s samples/_shared/tests`（全量 shared，含 assets 重构后的下载/校验用例） | **Ran 124 tests — OK** |
| `unittest discover -s samples/vision/efficient_sam/tests` | **Ran 19 tests — OK** |
| `unittest discover -s samples/vision/mobile_sam/tests` | **Ran 17 tests — OK** |
| `unittest discover -s tools/sample_contract/tests`（checker 自身 fixture 套件） | **Ran 27 tests — OK** |
| `tools/sample_contract/check.py --scope migration --exemptions tools/sample_contract/baselines/ultralytics-readme-debt.json` | **rc=0；36 samples / 0 violations / 39 policy skips / 84 exemptions applied** |
| 主机真 CLI 抽查：两个 `evaluator/compare.py --target s100`（无板，gate 应拒） | **rc=2，stderr 单行 `error: ValueError: Local execution requires recognized board identity...`，无 traceback** |

checker 不带 `--exemptions` 时为 84 violations（即被 CI 基线豁免的 B1-R6 ultralytics README
既有债务），与本次改动无关；带 CI 同款豁免文件后归零。

## 5. 未完成项 / 移交

- **板端复验 not-run**：需在 X5 8GB 与 S100（Python 3.10）以包含本修复的提交重跑两组
  `evaluator/compare.py`，确认不再触发摘要兼容异常且真实进程退出码与持久化记录一致；
  完整数值对照必须 rc=0 且全部检查通过。原始失败记录保留，不覆盖。
- 板端 Python 3.10 为协调方日志所述；本任务未连板，未直接观测板端解释器版本。
- B6/B7 状态不由本记录关闭；等待独立复核（Codex）后再由协调方决定提交与同步。

## 6. B6-B2 追加：共享 SAM 调度根因（2026-09-24 第二轮）

B6-B1 已由 Codex 独立复审并以提交 `1bfd8fa` 推送（非本会话操作）；板端复测用相同补丁的
GitHub 分支 `f888c8f`。上一节（§1–§5）为第一轮历史记录，原样保留。本轮在
`rdk-b6-glm-python310` worktree（基线 `1bfd8fa`）继续，**未 commit / 未 push / 未 SSH**；
板端 **not-run**，不宣称实板通过，不关闭 B6/B7。

### 6.1 缺陷与证据

`../.coordination/b6-x5-python310-recheck.json`（只读引用）：

- **S100**：efficient_sam / mobile_sam 全部检查通过（第一轮修复在 S 侧板端完全生效）。
- **X5**：两样本越过 hashlib 后，统一 runner 在 encoder 调度处失败，真实 rc=2、证据保存正确：
  `error: RuntimeError: encoder scheduling failed: set_scheduling_params(): incompatible
  function arguments. ... priority: collections.abc.Mapping[str, typing.SupportsInt] | None ...
  Invoked with: ... kwargs: priority=0`。

根因：`samples/_shared/sam_runner.py` 的 X5 分支传**标量** `priority`，而真实原生 API 在所有
target 上都要求按模型名的 Mapping（共享 classification `model_runner.py:132` 即始终
`{binding.model_name: priority}` 的既有正确路径；S 固定源同样按模型名构造 Mapping）。
主机侧 FakeRuntime 的 `set_scheduling_params(**kwargs)` 来者不拒，让该错误在主机测试中
一直通过——本轮一并修正为真实协议风格。

### 6.2 固定源行为核实（未修改固定源）

逐一读取四个固定源（只读核实，未改动）：

- **X5 两样本**（`platforms/x5/.../{efficient_sam,mobile_sam}.py`）：helper 对 encoder/decoder
  各尝试 `model.set_scheduling_params(priority=<scalar>)` 并 `except TypeError: pass`——
  在真实 X5 SDK 上**实际未应用任何调度**。此前板端 legacy 侧能跑完，只是源 helper 的调度
  配置从未生效；本轮证据与 README 均按此事实表述，不声称 source 已成功应用 priority。
- **S 两样本**（`platforms/s/...`）：helper 按模型名构造
  `{'priority': {name: p}, 'bpu_cores': {name: cores}}` 后逐模型调用，与原生 Mapping API
  一致（S100 板端成功互证）。

### 6.3 修复方案

1. **根因**（`sam_runner.py`）：`priority` 始终以 `{binding.model_name: priority}` 下发
   （对齐 classification runner 与板端真实签名），`bpu_cores` 原本就是 Mapping 保持不变；
   X5 显式 `--bpu-cores` 的拒绝语义原样保留。
2. **evaluator 调度对照**（`sam_evaluator.py`）：
   - `_RecordingRuntime` 拦截每次原生调度调用，按 side 记录 `{stage, args, applied|error}`
     时间序列——失败 call 与成功设置都保留（X5 源 helper 的被拒标量调用也如实入账）。
   - 先**原样**调用固定源 helper（与源 CLI 行为一致，结果不加工）；随后由 evaluator 显式为
     legacy 侧各原生 runtime 设置同一份**已验证的按模型名 Mapping 控制参数**，使两侧在完全
     一致且确实生效的调度下对照；统一侧经修复后的 runner 应用同样参数。
   - `comparison.json` 新增 `scheduling` 节：`requested` / `source_helper_call` /
     `source_helper_behavior`（分 x5/s 的核实结论）/ `explicit_control`（每 stage 模型名 +
     原生实参）/ `native_calls`（在 `finally` 快照，早失败也保留部分调用）/ `note`
     （明示 explicit_control 是 evaluator 对拍控制项、不是固定源 CLI 自身调度行为、
     不改变两侧 pre/forward/post）。数值阈值与两侧算法零改动，保持 Python 3.10 兼容。

### 6.4 测试与结果（先证伪，再修复）

- **FakeRuntime 协议化**（`test_sam_binding.py`）：签名改为
  `set_scheduling_params(priority=None, bpu_cores=None)`，仅接受非空 `Mapping[str,int]` /
  `Mapping[str,序列]`，其余抛板端同款 `incompatible function arguments` TypeError；
  scheduler 断言更新为按模型名 Mapping，并新增
  `test_scalar_priority_never_reaches_the_native_protocol`（标量永不触达原生层）。
- **新增 evaluator 调度用例**（`test_sam_evaluator.py::SchedulingControlTests`）：
  X5 两样本 × priority 0/7：legacy 每 stage 两条记录（helper 标量被拒 + explicit Mapping
  生效）、unified 一条 Mapping 生效、explicit_control 模型名逐 stage 正确；
  S100 两样本：helper Mapping 与 explicit 控制均生效并各自入账；
  X5 显式 bpu_cores 在任何原生调用与证据目录创建之前被拒。
- **修复前证伪**：binding 套件 `Ran 12 — FAILED (errors=2)`（两个调度用例被协议 fake 以
  `TypeError: ... priority must be a nonempty Mapping` 拒绝）；evaluator 套件
  `Ran 14 — FAILED (failures=1, errors=8)`，其中 X5 用例复现板端同形
  `RuntimeError: encoder scheduling failed: set_scheduling_params(): incompatible function
  arguments...`，**既有** X5 全流程用例也转为失败——证明旧 fake 的宽容确实掩盖了板端缺陷；
  S100 用例因调度证据节缺失而失败（功能当时不存在）。
- **修复后**（解释器同 §4）：

| 套件 / 命令 | 结果 |
| --- | --- |
| `unittest discover -s samples/_shared/tests` | **Ran 128 tests — OK** |
| `unittest discover -s samples/vision/efficient_sam/tests` | **Ran 19 tests — OK** |
| `unittest discover -s samples/vision/mobile_sam/tests` | **Ran 17 tests — OK** |
| `unittest discover -s tools/sample_contract/tests` | **Ran 27 tests — OK** |
| checker `--scope migration`（含 CI 84 条豁免） | **rc=0；36 samples / 0 violations / 39 skips / 84 exemptions** |

- 数值对照阈值未动：既有全部对拍用例（含 128 shared）继续通过。
- S100 受影响范围：统一 S 路径数值行为不变（原本即 Mapping），新增的调度证据节与
  helper/explicit 双记录已由 S100 用例覆盖；板端 S100 复跑由协调方决定。

### 6.5 本轮变更文件

| 文件 | 变更 |
| --- | --- |
| `samples/_shared/sam_runner.py` | priority 改为始终按模型名 Mapping（+3/−1 行，含板端证据注释） |
| `samples/_shared/sam_evaluator.py` | 调度调用拦截记录、helper 原样调用、explicit 同调度控制、`scheduling` 证据节 |
| `samples/_shared/tests/test_sam_binding.py` | FakeRuntime 原生协议化 + 调度断言更新 + 标量回归 |
| `samples/_shared/tests/test_sam_evaluator.py` | 新增 `SchedulingControlTests` 3 用例 |
| 两个 evaluator `README.md` + `README_cn.md` | Command 节追加“Scheduling control/调度控制”说明（中英同步），无板测通过表述 |
| 本报告 | 追加本节（§6），§1–§5 历史保留 |

### 6.6 未完成项 / 移交

- **X5 板端复验 not-run**：待 Codex 审查后经 GitHub 送板，重跑两组 X5
  `evaluator/compare.py`，期望越过调度、以 rc=0/1 结束，且 `comparison.json` 的
  `scheduling.native_calls` 显示 X5 源 helper 标量被拒 + 两侧 explicit/runner Mapping 生效。
- S100 板端复验（新证据节范围）由协调方决定是否安排。
- B6/B7 状态不由本记录关闭；等待独立复核。
