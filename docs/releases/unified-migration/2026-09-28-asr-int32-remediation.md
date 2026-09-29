# ASR-R1 int32 整改（作者记录）

Author: Claude Code + GLM。Reviewer: Codex（独立复审 pending，**ASR-R1 不由本包关闭**）。
依据 [2026-09-28-asr-independent-review.md](2026-09-28-asr-independent-review.md)
的 **ASR-R1**（P2）：Python 绑定宣称接受 int32 SCALE 输出，但 `transcribe` 走默认
float32 仿射解码并在 `decode_logits` 前再次强制 float32，raw `16777216`（blank ID 0）
与 `16777217`（token ID 1）被舍入成人为平局，产线返回空串而非 `token1`。
分支 `codex/b7-board-integration-20260924`；整改前实现字节与评审候选基线逐字节一致
（postprocess `9c6ab03c…`、decoding `4dffc32d…`，同评审
[evidence](evidence/2026-09-28-asr-independent-review/verification.json) 记录）。
任务由 GLM（Claude Code 会话）实施，等 Codex 独立复审。

**状态口径：**

- 全程**未 commit / 未 push / 未 merge / 未 stash**。作者改动仅限
  `samples/speech/asr` 内 6 个文件（runtime `postprocess.py`、`decoding.py`、
  `tests/test_contract.py`、`tests/test_binding.py`、双语 runtime `README.md`/`README_cn.md`），
  外加本记录与
  [evidence/2026-09-28-asr-int32-remediation/](evidence/2026-09-28-asr-int32-remediation/verification.json)
  新目录。评审报告与其证据、计划台账、manifest、`samples/_shared`、其他 sample 零接触；
  并行会话的 PointNet 同类修复（已提交 `e5ff50f2`）未触碰，本包亦不覆盖。
- **Board = not-run**：不连板卡/SSH/远程主机，不下载模型或安装依赖，不跑导出/校准/
  量化。纯主机张量级修复，不构成板端或量化验收证据。
- 会话进行中 HEAD 因并行提交前移；本包编辑与验证均锚定上述评审基线字节（哈希复核），
  红绿链条不受并行提交影响。

## 1. 修复方式：dtype 分派 + float64 比较精度贯穿 argmax

`runtime/python/postprocess.py` 的 `transcribe` 按 dtype 分派（与已接受的
UNetMobileNet/PointNet 先例一致）：

- **F32 路径逐字保留**：`raw_f32` 变换 + `decode_logits`（float32），vestigial 量化
  描述符照旧接受并忽略；该分支语句与整改前等价。
- **整数路径（int8/uint8/int16/int32，即绑定宣称的完整集合）**：显式取
  `output_quants` 描述符（缺失仍抛 `ValueError`，与原 `apply_output_transform` 的
  `OutputTransformError` 语义一致），以
  `dequantize_tensor(raw, quant, dtype="float64")` 反量化，argmax 在 float64 上
  完成后再进 ID 解码。float64 尾数 53 位可精确表示全部 int8..int32 值，故不同
  整数 raw 的大小关系在 argmax 前不再被舍入抹平。
- **解码器协调**：评审明确"只改共享 helper 参数不够，下一层强制 float32 必须协调"。
  `decoding.py` 新增 `decode_exact_logits`（校验有限 float64 `[1,T,V]` → argmax →
  既有 `decode_ids`），`decode_logits` 与 `decode_ids` 一字未动。

**边界保持**：`samples/_shared/quantization.py` 零改动（float64 是显式实参，
`apply_output_transform` 及其他 sample 依赖的默认 float32 行为不变，共享套件
158 项回归通过）；binding 的 int32 宣称契约不删（未用"移除整数支持"规避）；
`model_binding.py`、`asr.py`、`frontend.py`、`main.py`、`model_runner.py`、
`vocabulary.py`、`audio_io.py`、`run.sh`、`runtime/cpp/`、`model/`、`conversion/`、
`evaluator/`、sample 级双语 README、`test_data/` 均 diff 为零。CTC 先折叠后去
blank、legacy 模式、每 chunk 无状态、全部现行 API 签名保持不变。

## 2. 新增测试（+6；既有 21 项零改动）

`tests/test_contract.py` 新增 `IntegerTranscribeTests`：

- `test_reviewed_int32_counterexample_decodes_token1`：评审原反例
  （`2**24`/`2**24+1`、`[1,1,3503]`、scale `[1]`、offset `[0]`）在 ctc 与 legacy
  下都必须返回 `token1`，并先断言 float32 假平局前提本身为真（钉住缺陷机理）。
- `test_int32_per_channel_scale_and_offset_ranking`：逐通道 scale（`[1,3,1]`，
  raw 偏好 `b` 而 dequant 胜出 `a`）；逐通道 offset（zero_point `[9,0,0]`、
  scale `[2,1,1]`，raw 偏好 blank 而去偏后 `b` 胜出，覆盖负中间值与非标量
  zero_point 广播）。
- `test_int32_true_ties_keep_lowest_id`：dequant 后精确相等的真平局仍取最小 ID
  （`[2,2,0]` → blank、`[0,2,2]` → 最小非 blank），两种实现下都通过——区分
  "精度造成的假平局"与"真实平局"，防止修复顺手改平局规则。
- `test_integer_logits_follow_selected_decode_mode`：整数输出下 ctc/legacy 双模式
  （`[a,a,blank,a]` → `aa`/`aaa`）。
- `test_float32_transcribe_keeps_f32_decoding_and_ignores_vestigial_quant`：
  F32 输出仍按 float32 解码（含极负值与精确平局）、vestigial 描述符仍被忽略。
- `test_exact_decoder_rejects_non_exact_inputs`：`decode_exact_logits` 拒绝
  float32/int32/NaN/错误 batch 与词表宽度。

`tests/test_binding.py` 的 `test_stages_and_integer_logits` 原地扩展：真实
`bind_model` 门禁接受 int32 SCALE 元数据后，`ASR.predict` 全组合
（frontend→forward→post_process）在评审分数对上返回
`vocabulary[1] + vocabulary[5]`，legacy 返回 `vocabulary[1] + vocabulary[5]*3`。

## 3. Runtime 双语 README 同步

`runtime/python/README.md` / `README_cn.md` 三阶段 I/O 的 `post_process` 行与解码
说明段同步为：F32 输出照旧按 float32 解码；整数 SCALE 输出以 float64 比较精度经
共享量化模块反量化，不同整数在 argmax 前不失序——float32 会把 `2**24` 与
`2**24 + 1` 这类相邻整数舍入成假平局；只有完全相等的分数才平局并取最小 ID。
两文档锚点 ID、参数表、集成示例均未动。

## 4. 验证结果（解释器、命令、退出码与工件见
[verification.json](evidence/2026-09-28-asr-int32-remediation/verification.json)）

| 检查 | 结果 |
| --- | --- |
| 评审反例复现（整改前，原 fixture，真实 `bind_model`+`ASR.post_process` 路径） | actual `""` / expected `token1`，exit 1；float32 假平局前提实测为真 |
| 红：终版 27 项测试 × 仅复活整数路径 float32 强转（postprocess 单文件临时回退） | Ran 27，FAILED (failures=2)：恰为反例的 transcribe 层与绑定/任务层两用例；真平局守卫、双模式、F32 用例双向通过 |
| 评审反例（整改后，最终字节） | ctc 与 legacy 均 `token1`，exit 0 |
| 有界 ASR suite（`unittest discover -s samples/speech/asr/tests -v`） | **Ran 27 tests — OK**（21 既有 + 6 新增） |
| `tools/sample_contract/check.py --sample samples/speech/asr` | 0 violations，1 个记录在案的 R-STAGE-PURITY CLI policy skip，0 exemptions |
| `samples/_shared/tests`（共享零改动回归保险） | Ran 158 tests — OK |

环境：仓库根 + `rdk_model_zoo/.venv`（CPython 3.14.7，numpy 2.5.3）。

## 5. 边界与移交

- ASR-R1 保持 open，待 Codex 独立复审本 diff；本包不自行关闭，也不声明 ASR/B10
  整体验收。
- 这是合成主机张量级反例的修复：不证明已发布 HBM 实际发出 `2**24`/`2**24+1`
  这一对分数（评审原文同此口径），只恢复被反证的一般整数契约。
- native CTest 未重跑：`runtime/cpp/` 零改动，评审记录的 5 项 native CTest 对未改
  源仍然有效；C++ float 解码器不在 ASR-R1 范围（原生明确拒绝整数输入，
  runtime/cpp README 已声明，无契约宣称需要协调）。
- float64 只影响整数输出的 argmax 精度边界；无其他已知行为差异，未做板端精度对比。
- 未触碰 `~/.claude/` memory、并行会话的新增文件（如并行 readme-depth-classifiers
  记录）、`docs/release/` manifest、计划台账、任何 reviewer 文件。
- 本包到此为止，等 Codex 独立复审。
