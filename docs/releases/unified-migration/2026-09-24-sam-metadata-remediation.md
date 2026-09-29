# 2026-09-24 — B6 SAM metadata JSON 序列化回归修复（作者记录）

**性质**：作者自检记录（host-only），B7 metadata 专项（见
`2026-09-24-b7-metadata-remediation.md`）授权后的**有界同类修复**。**未板测、
未独立审阅通过**；B7 共享 helper 已在 X5 YOLOv5/YOLOWorld 与 S100
YOLOv5/ByteTrack 实板 compare 通过（协调者复跑，commit `ac4046c` 前后），
**该事实不覆盖、不声称 SAM 板测结论**。
工作树：`rdk-b7-glm-metadata`，基点 `ac4046c`（clean）；解释器
`/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk_model_zoo/.venv/bin/python`
（Python 3.14.7，host 专用）。

## 缺陷（B6 证据路径，静态认定 + fixture 同构复现）

`samples/_shared/sam_evaluator.py` 的 `recording_factory.create()` 此前执行

```python
summary['metadata'][side][stage] = asdict(RuntimeMetadata.from_runtime(runtime))
```

与 B7 六样例同构：`dataclasses.asdict` 深拷贝叶子值，板端
`hbm_runtime.HB_HBMRuntime.QuantParams` 拒绝 pickle → `TypeError`。SAM 的
host fixture（`test_sam_binding.metadata`）的 `output_quants` 值是**普通
dict**（可 deepcopy），故既有 host 测试全绿而板端证据路径存在同类崩溃面。
本工作树无 SAM 板端崩溃日志——缺陷经由 B7 板端证据（X5 yolov5 compare，
evidence `2026-09-24-b7-board-initial/b7-x5-python-compare.json`）同类推导并
以 copy-hostile fixture 复现，**不是**已捕获的 SAM 板端失败记录。

## 修复（最小接线，分层不变）

仅改 `samples/_shared/sam_evaluator.py` 一处调用点：`asdict(...)` 换为既有
`metadata_evidence(...)`（`samples/_shared/runtime_meta.py`，B7 已提交并通过
实板验证的共享 helper）。`RuntimeMetadata`/`metadata_evidence` 保持函数内
局部导入（沿用原文件模式，模块顶层零新增依赖）；移除文件顶部不再使用的
`from dataclasses import asdict`。

不改模型、不改推理/录制（`_RecordingRuntime` 原样保留 raw tensor 捕获）、
不改 `compare_records` 容差（raw 1e-5 / IoU 1e-6 / rtol 0 不动）、不改证据
字段与 `_json_default` 兜底。

**共享入口核对**：`samples/vision/efficient_sam/evaluator/compare.py` 与
`mobile_sam/evaluator/compare.py` 均为薄壳，直接委托
`sam_evaluator.main('efficient_sam'|'mobile_sam')`，自身不涉及序列化，无需
改动；两份 evaluator/README.md 对 `comparison.json` 内容的描述（"metadata"）
在修复后仍准确，无需更新。

**绑定层核对**（fixture 可行性）：`sam_binding._freeze` 对 Mapping/list/ndarray
之外的值原样透传（`sam_binding.py:36`），SDK 对象描述符可无损通过绑定快照；
`bind_stage` 不检查 `output_quants` 值；`sam_runner`/`sam_stages` 从不读取
quants（SAM 永不反量化），对象描述符对数值路径零影响。

## 回归测试（`samples/_shared/tests/test_sam_evaluator.py`，+1）

`test_metadata_evidence_survives_copy_hostile_board_quant_params`：

- 注入 `__deepcopy__`/`__copy__` 显式抛
  `TypeError: cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object`
  的 `_BoardQuantParams`（fixture 与 B7 板端失败面同构），随
  `runtime_factory` seam 同时挂到 encoder/decoder 两个 stage 的两个 runtime
  上，**真实走完 `run_comparison` 全流程并写出 `comparison.json`**。
- 断言：`passed`（比较本身不受影响）；`raw_close`/`inputs_equal` 仍真、
  `mask_changed_pixels==0`（raw tensor 捕获与容差判断原样）；legacy/unified ×
  encoder/decoder 四份 metadata 快照每份 `output_quants` 保留
  `quant_type='SCALE'`、`scale=0.25`、`zero_point=7`、`axis=3`（float32 精确
  值，不经文本往返）；落盘 JSON `json.loads` 后字段齐全；raw
  `legacy_encoder_outputs_image_embeddings.npy` 仍在盘。

**阴性对照**：把 `sam_evaluator.py` 临时还原为 `asdict` 接线后，该测试以
板上同文 `TypeError: cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams'
object` 失败；恢复修复后转绿——证明测试对原缺陷敏感，而非恒真。

## 验证汇总（host，作者本人执行）

| 套件 | 结果 |
| --- | --- |
| `python -m unittest discover -s samples/_shared/tests` | 109/109 OK（基线 108 + 新增 1） |
| `python -m unittest discover -s samples/vision/efficient_sam/tests` | 19/19 OK |
| `python -m unittest discover -s samples/vision/mobile_sam/tests` | 17/17 OK |
| `py_compile` 改动文件 | 通过 |

**未运行**：X5/S 板端 EfficientSAM/MobileSAM compare 重跑（无板卡授权）；
独立 review（属协调者流程）。

## 边界与交接

- **未板测**：host-passed 仅证明 deepcopy 失败面消除与投影保真；SAM 实板
  `comparison.json` 的 metadata 快照以协调者板测为准，此前 SAM 状态为
  host-passed / board-not-run。B7 样例的实板通过不构成 SAM 证据。
- **未独立通过**：本记录为作者自检，未经独立审阅。
- 工作树剩余未提交（待协调者）：`samples/_shared/sam_evaluator.py`、
  `samples/_shared/tests/test_sam_evaluator.py` 与本记录。作者未执行任何
  git commit/push/merge/SSH。
