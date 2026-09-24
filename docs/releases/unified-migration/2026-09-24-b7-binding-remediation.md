# B7 绑定整改记录 — LPRNet / FCOS native output binding（2026-09-24）

> 作者：本地 GLM 专项会话（Claude Code on GLM）。本文记录有界整改阶段的
> 位置、内容与回归证据，待 Codex 独立复验（GitHub / 板测 / 独立评审）。
> 基点 `ae0f185` 已包含 SDK 序列化（`metadata_evidence` 不拷贝扩展对象）
> 与 S32 dtype 别名修复，本阶段未重做。

## 范围与证据输入

仅修复两个刚被实板证伪的 native output binding，不做其它 B7/B8 工作：

- **LPRNet**（`.coordination/b7-x5-lprnet-ae0f185-compare.json`，rc=2）：
  发布版 `lpr.bin` 的 runtime metadata 输出为 `(1, 68, 18, 1)`，绑定硬编码
  `(1, 68, 18)`，`bind_model` 直接报
  `Expected output shape (1, 68, 18), got (1, 68, 18, 1)`。源码
  `platforms/x5/.../lprnet.py` 的 `forward` 用 `.squeeze()` 把
  `(68, 18)` CTC 载荷交给解码。制品正常加载（HorizonRT 1.24.3 构建日志
  完整），排除资产/SDK 不可用。
- **FCOS**（`.coordination/b7-x5-fcos-ae0f185-compare.json`，rc=2）：
  实板 `runtime.run` 返回 dict 的键顺序与 `metadata.output_names` 不同，
  但 15 个名字集合完全相同；`validate_outputs` 的
  `tuple(outputs) != self.output_names` 按插入序误拒。

两个负例都是**绑定层协议错误**，不是模型或 SDK 问题。

## 修复原则

- 绑定记录 runtime metadata 报告的**完整 native shape**；`forward` 保持
  真实 raw shape；**只有 `post_process`** 消除协议允许的单元素轴。
- 不做任意 rank/轴顺序的 reshape/重排；不支持的布局在绑定期与
  逐次调用期都显式拒绝，不静默接受。
- 3D `(1, 68, 18)` 是**旧统一契约/host fixture 兼容布局**，仅为既有
  host 测试与注入 runner 保留；未观察到报告该布局的已发布 SDK 制品，
  不得描述为某个"旧版本 SDK"。唯一实测发布协议是 4D `(1, 68, 18, 1)`。
  两种布局都必须绑定期接受且逐次输出严格匹配，不接受任意其它 shape。
- FCOS 输出校验按**名字集合**精确匹配（缺失/多余都拒绝），逐绑定名
  校验 shape/dtype/finite；保留 raw ndarray 与原 dict 身份，角色只从
  binding 的 cls/box/center 名字元组解析，不按输出 dict 遍历猜角色。

## LPRNet 整改明细

- `samples/vision/lprnet/runtime/python/model_binding.py`
  - `OUTPUT_SHAPE = (1, 68, 18)` → `OUTPUT_SHAPES = ((1, 68, 18), (1, 68, 18, 1))`
    （注释写明 4D 为唯一实测发布协议、3D 为旧统一契约/host fixture 兼容
    布局且未观察到对应已发布 SDK 制品，以及"只在 post_process 消单元素
    轴"），新增 `CTC_LOGITS_SHAPE = (68, 18)`。
  - `ModelBinding` 新增字段 `output_shape`，记录绑定时的完整 native
    shape；`bind_model` 只接受 `OUTPUT_SHAPES` 内的 metadata shape，
    其它 rank/轴序（如 `(1, 18, 68, 1)`、`(1, 68, 18, 2)`、`(1, 1, 68, 18)`）
    报 `MetadataMismatchError`。
- `samples/vision/lprnet/runtime/python/model_runner.py`：`__call__` 的
  输出检查由硬编码 `(1, 68, 18)` 改为 `raw.shape != binding.output_shape`
  （绑定后形状漂移在解码前拒绝），返回值保持真实 raw shape，不再暗示 squeeze。
- `samples/vision/lprnet/runtime/python/lprnet.py`：新增模块级
  `ctc_logits()`——只 drop size-1 轴（`np.squeeze`），结果不是 `(68, 18)`
  即报错，绝不 reshape/重排；`post_process` 改为校验
  `value.shape == self.binding.output_shape` 且 dtype float32，再经
  `ctc_logits` 进入源 CTC 解码。`decode_plate` 的 `(68, 18)` 契约不变。
  `forward` 不做任何形状变换。
- `samples/vision/lprnet/runtime/python/main.py`：`--dry-run` JSON 的
  `output_shape` 改为发布版真实 native `[1, 68, 18, 1]`，新增
  `output_layout` 字段写明 4D 实测板端协议、3D 旧 host/API 兼容契约
  （未观察到对应已发布 SDK 制品）与消单元素轴规则。**注意**：这是
  dry-run JSON 的可见字段变化，板测复验时以新字段为准。

## FCOS 整改明细

- `samples/vision/fcos/runtime/python/model_binding.py`
  `validate_outputs`：`tuple(outputs) != self.output_names` 改为名字集合
  精确匹配——`missing`/`unexpected` 分别列出不匹配项（缺失、多余、空
  映射都拒绝）；非 Mapping 输入显式拒绝（旧实现遇到 list 会因 tuple
  比较抛出含糊的 ValueError）；随后**仍按 `self.output_names` 顺序**逐名
  校验 ndarray 类型、精确 shape、canonical dtype、finite——原严格语义
  全部保留。返回值仍是调用方原 mapping 对象（`assertIs` 级别身份不变）。
- 后处理（`fcos.py`）本就按 `binding.cls_output_names` /
  `box_output_names` / `center_output_names` 取数，未按输出 dict 遍历猜
  角色，无需改动。

## 文档同步（双语）

- `samples/vision/lprnet/runtime/python/README{,_cn}.md`：环境节输出
  shape 描述改为"按 metadata 原样绑定：`(1,68,18,1)` 为实测板端协议；
  3D `(1,68,18)` 仅是旧 host/API 兼容契约（未观察到对应已发布 SDK
  制品）"，不接受其它秩/轴序；Results 节写明 post_process 先消
  单元素轴到 `(68,18)` 再 CTC；Stage I/O 的 `forward`/`post_process`
  行同步；故障排查新增"输出 metadata shape 越界 / 运行时形状漂移"条目。
- `samples/vision/lprnet/conversion/README{,_cn}.md`：部署契约输出改为
  `output float32 (1,68,18,1)`（发布版 native logits；CTC 消单元素轴后
  消费 `(68,18)`）。
- `samples/vision/lprnet/README{,_cn}.md`：支持矩阵状态由"板测未运行"
  更新为"2026-09-24 修复板端绑定（native 输出 `(1,68,18,1)`），板测复验
  待运行"——如实反映实板已运行且暴露绑定缺口的事实，复验通过后由
  Codex 翻转。
- `samples/vision/fcos/runtime/python/README{,_cn}.md`：`forward` 行写明
  "按精确名字集合匹配，板端 dict 顺序可与 `metadata.output_names` 不同
  （X5 证据 2026-09-24），缺失/多余/shape/dtype/非有限值一律拒绝，
  调用方数组身份不变"；故障排查新增对应错误消息条目。

## 新增测试（不编辑旧测试文件）

新增测试数以 `unittest.TestLoader().loadTestsFromModule(...)` 实际加载
计数为准：LPRNet 新文件 **10** 项、FCOS 新文件 **13** 项
（基线 12/24 → 套件 22/37）。

- `samples/vision/lprnet/tests/test_board_output_contract.py`（10 项）：
  4D native `(1,68,18,1)` 绑定正例并记录 `output_shape`；3D 旧 API 兼容
  契约正例；错误 rank/轴序/非尾单元素轴 4 项负例；双输出负例；`forward`
  保持 4D raw + `post_process`/`predict` 解码 "京A7"；3D 旧 API 契约
  端到端；`ctc_logits` 只消单元素轴；绑定后形状漂移（4D 绑定返 3D、3D
  绑定返 4D）负例；`post_process` 拒绝形状/轴序/dtype 漂移；dry-run
  JSON 契约（`output_shape == [1,68,18,1]` 且 `output_layout` 写明实测
  协议、旧 host/API 契约与 CTC 载荷三组形状）。模块说明与测试命名
  （`LEGACY_API_OUTPUT_SHAPE`、`test_legacy_api_3d_contract_*`）明示 3D
  是旧 host/API 兼容契约，未观察到对应已发布 SDK 制品。
- `samples/vision/fcos/tests/test_board_output_contract.py`（13 项）：
  打乱 dict 顺序下 `validate_outputs` 通过且返回原 mapping/原 ndarray
  （`assertIs`）；乱序 runner 的 task `predict` 与顺序版结果全等；乱序
  runtime dict 的 runner 直通；**实际 evaluator 回归**（`run_comparison`
  注入乱序 `run()` 的 FakeRuntime + legacy seam，return_code 0、passed）；
  负例：缺一输出、多一输出、空映射、非 Mapping、非 ndarray、错误 shape、
  错误 dtype、非有限值、task 级缺输出——原严格语义逐项保留。
- **未编辑任何旧测试文件**：`test_lprnet.py` 的 3D `(1,68,18)` FakeRuntime
  fixture 作为旧统一契约/host fixture 兼容布局继续有效（`bind_model`/
  `runner`/task 对其行为不变）；`test_fcos_contract.py` 的 metadata
  fixture 顺序与 `output_names` 一致，行为不变。

## 回归证据（`rdk_model_zoo/.venv/bin/python`，工作树内）

- LPRNet 全套件：`python -m unittest discover -s samples/vision/lprnet/tests`
  → **Ran 22, OK**（基线 12 + 新 10）。
- FCOS 全套件：`python -m unittest discover -s samples/vision/fcos/tests`
  → **Ran 37, OK**（基线 24 + 新 13）。
- 合同检查器（仅受影响范围）：
  `tools/sample_contract/check.py --sample samples/vision/lprnet` 与
  `--sample samples/vision/fcos` → 各 **0 violations**（各 1 个既有的
  R-STAGE-PURITY policy skip）。
- **修复前基线复跑**（临时 detached worktree @ `ae0f185`，仅拷入两个新
  测试文件）：LPRNet `FAILED (failures=1, errors=6)`——含
  `Expected output shape (1, 68, 18), got (1, 68, 18, 1)` 的绑定错误与
  dry-run 3D 断言；FCOS `FAILED (failures=3, errors=4)`——含乱序 dict
  被旧顺序比较误拒（与实板 rc=2 同因）。该基线复现后即删除临时
  worktree，未留任何工作树外改动。

## 边界与移交（Codex 复验清单）

- 本阶段**未**改：C++ 代码、`samples/_shared/runtime_meta.py`、全局台账
  （migration map Refactor 列）、manifest、发布标签；未 commit/push/
  merge/SSH。
- 板测复验建议：在 x5-8g 上以相同命令重跑
  `python3 -m samples.vision.lprnet.evaluator.compare --target x5 --output-dir <new>`
  与 `python3 -m samples.vision.fcos.evaluator.compare --target x5 --output-dir <new>`；
  LPRNet 期望 4D metadata 绑定成功且 raw logits 两侧同形 `(1,68,18,1)`；
  FCOS 期望乱序 dict 通过校验、rc=0/1（数值差异另有容差语义）。
  LPRNet dry-run JSON 的 `output_shape` 已改为 `[1,68,18,1]` 并新增
  `output_layout`，对比脚本如依赖旧字段需同步。
- 复验通过后：LPRNet 根 README 状态行翻转为板测通过、台账行由执行方
  更新、按需评估是否需要在两个 sample 的 evaluator evidence 中显式记录
  native output shape（本文不预填任何板测数值）。
