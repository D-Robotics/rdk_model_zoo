# PointNet R2 整改（作者记录）

本记录覆盖 2026-09-28 独立评审
[2026-09-28-pointnet-independent-review.md](2026-09-28-pointnet-independent-review.md)
的 **POINTNET-R2**（P2）：`samples/vision/pointnet` 绑定宣称接受 int32 SCALE 输出，
但 `post_process` 经 `apply_output_transform` 的默认反量化路径把 raw 值先转成
float32 再 argmax。评审方反例（标量 scale=1、zero_point=0，raw
`[16777216, 16777217, 0, 0]`）中 float32 把 16777217 舍入成 16777216，制造人为
平局，使实际输出 class 0、精确值应为 class 1。本会话在整改前的产线
`post_process` 路径上先复现了该反例（actual=[0]，exit 1，见
[evidence](evidence/2026-09-28-pointnet-int32-remediation/reproducer-before-fix.json)），
再做红-绿整改。任务由 GLM（Claude Code 会话）实施，等 Codex 独立复审。

**状态口径：**

- 全程**未 commit / 未 push / 未 merge / 未 stash**；作者改动仅限
  `samples/vision/pointnet` 内 4 个文件（runtime `pointnet.py`、
  `tests/test_pointnet.py`、`runtime/python/README.md`、`runtime/python/README_cn.md`），
  外加本记录与
  [evidence/2026-09-28-pointnet-int32-remediation/](evidence/2026-09-28-pointnet-int32-remediation/verification.json)
  新目录。评审报告与证据、计划台账、仓库根文件、其他 sample 零接触。
- **Board = not-run**：本任务不连板卡、不下载模型、不跑导出/量化流程、不装依赖。
  这是纯主机张量级修复，不构成板端或量化验收证据。
- 主机测试通过是作者自检，不等于独立验收；POINTNET-R1 保持 closed，
  PointNet 样例与 B8 整体验收仍以评审方结论为准。

## 1. 修复方式：按评审指定走共享 `dequantize_tensor(dtype=float64)`

`runtime/python/pointnet.py` 的 `post_process` 弃用 `apply_output_transform`
的默认 float32 解码，改为按 dtype 分派（与已接受的 UNetMobileNet 先例一致）：

- **F32 路径逐字节等价**：`decoded = raw`，vestigial 量化描述符照旧忽略；
  `forward` 返回 raw、raw 张量在 `post_process` 后不变（有测试断言）。
- **整数路径（int8/uint8/int16/int32，即绑定宣称的完整集合）**：
  `dequantize_tensor(raw, meta.output_quants[name], dtype="float64")`。
  float64 尾数 53 位可精确表示全部 int8..int32 值；对有限正 scale，不同整数
  raw 的解码差 ≥ scale，远大于该量级的 float64 ULP，故严格排序在 argmax 前保留，
  不再产生舍入平局。描述符缺失时仍抛 `ValueError`（与原
  `apply_output_transform` 的 `OutputTransformError` 语义一致）。
- **真实平局语义不变**：只有解码后完全相等的分数才平局，argmax 取最小 part 序号
  （新测试钉死）。

**边界保持**：`samples/_shared/quantization.py` 零改动（其他 sample 的
`apply_output_transform` 默认 float32 行为不受影响，共享套件 158 项回归通过）；
binding 的 int32 宣称契约不删（`model_binding.py` 仍接受并校验 int8..int32 SCALE
描述符）；目标选择（`resolve_selection`/board identity gate）、R1 已修复的
`visualization.py` 与 `main.py` 均 diff 为零。

## 2. 新增边界测试（tests/test_pointnet.py，+3 项）

前两项在整改前实现上为 **RED**（临时原位复活缺陷后用最终测试文件重跑取证，
`[tests-red-before-fix.log](evidence/2026-09-28-pointnet-int32-remediation/tests-red-before-fix.log)`），
第三项在两种实现上都通过，防止"修复"时顺手改变平局规则：

- `test_int32_fine_differences_survive_decode`：评审方原反例
  （`2**24` 与 `2**24+1`）加一个负值对称点（`-(2**24+1)` vs `-(2**24)`，尾随
  `-2**25` 保证负类不致胜）：float32 在 `|n| > 2**24` 处间距为 2，正负两点都把
  精细差分舍入成人为平局，整改前两点全错（RED 实测 ACTUAL `[0,0]`）。
- `test_int32_per_channel_scale_and_offset_ranking`：逐通道 scale（`[1,2,1,1]`）
  与大数值逐通道 offset（`2**30`），`2**29+1` 乘 2 应胜过 `2**30`——float32 解码
  在乘 scale 之前就丢掉低位并得出错误赢家。
- `test_int32_true_ties_keep_lowest_index`：`3*2 == 6*1` 的精确解码平局仍取
  index 0，区分"精度造成的假平局"与"真实平局"。

既有 23 项测试（含 `test_integer_logits_dequantized_only_in_post` 的 int16
逐通道用例与 F32 vestigial 用例）零改动、全部通过。

## 3. Runtime 两语 README 同步

`runtime/python/README.md` / `README_cn.md` Stage IO 表的 post_process 行与表后
说明段同步为：整数输出以 float64 仿射解码，int8..int32 不同 raw 值在 argmax 前
保持大小关系，float32 解码会把大整数舍入成人为平局（标注 POINTNET-R2）；只有
完全相等的解码分数才平局取最小 ID。两文档锚点 ID、参数表、代码示例均未动；
`test_readme_api_example_executes_with_real_runner_and_sdk_fixture` 继续执行两语
示例代码块并对比结果。

## 4. 验证结果（解释器、命令、退出码与日志见 [verification.json](evidence/2026-09-28-pointnet-int32-remediation/verification.json)）

| 检查 | 结果 |
| --- | --- |
| 评审方反例复现（整改前，exact fixture） | actual=[0] / expected=[1]，exit 1 |
| RED：新 3 项测试 × 复活缺陷（3.14.7） | Ran 26，FAILED (failures=2)，平局守卫双向通过 |
| GREEN：PointNet suite（3.14.7，live） | **Ran 26 tests — OK**（23 既有 + 3 新增） |
| 反例复现（整改后） | actual=[1]，exit 0，raw 张量在 post_process 后不变 |
| `tools/sample_contract/check.py --sample samples/vision/pointnet` | 0 violations，1 个记录在案的 CLI policy skip，0 exemptions |
| `samples/_shared/tests`（回归保险，共享代码零改动） | Ran 158 tests — OK |

环境：仓库根 + `rdk_model_zoo/.venv`（CPython 3.14.7，numpy 2.5.3）。与 R1 相同的
口径：3.10–3.12 解释器本机不可用，相关套件 not-run；新增测试不依赖解释器版本行为
（纯 numpy 数值），但此为作者判断，待评审复核。

## 5. 边界与移交

- Board 推理、模型/HBM 产物、导出与 OE/Mapper 量化、依赖安装：全部 not-run，
  也未声称。本修复不改变量化配方或转换文档（`conversion/`、`model/` 零改动）。
- float64 解码只影响整数输出的 argmax 精度边界；F32 制品路径与整改前逐字节一致，
  无已知的其他行为差异，但未做板端精度对比。
- 未触碰并行任务的 MiniCPM/gemma 改动、评审报告与历史证据、计划台账、
  `docs/release/` manifest。
- 本包到此为止，等 Codex 独立复审；PointNet 样例整体与 B8 验收另行推进。
