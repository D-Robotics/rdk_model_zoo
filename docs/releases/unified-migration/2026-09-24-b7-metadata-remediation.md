# 2026-09-24 — B7 metadata JSON 序列化专项修复（作者记录）

**性质**：作者自检记录（host-only）。**未板测、未独立审阅通过**；X5/S 板端
compare 重跑与独立 review 由协调者安排，本记录不声称板端结论。
工作树：`rdk-b7-glm-metadata`，基点 `5fc14a4`；解释器
`/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk_model_zoo/.venv/bin/python`
（Python 3.14.7，host 测试专用）。

## 症状（真实板端，只读引用）

X5 板上 `yolov5/evaluator/compare.py` 崩溃：

```text
File "samples/vision/yolov5/evaluator/compare.py", line 67, in create
    summary['metadata'][side]=asdict(RuntimeMetadata.from_runtime(runtime))
  ...
  File "/usr/lib/python3.10/dataclasses.py", line 1279, in _asdict_inner
    return copy.deepcopy(obj)
TypeError: cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object
```

原始记录（仅读路径，位于集成分支检出
`rdk_model_zoo`，本工作树内无此目录）：
`docs/releases/unified-migration/evidence/2026-09-24-b7-board-initial/b7-x5-python-compare.json`
（commit `5fc14a4`，host `x5-8g`，rc=1）。

## 根因与数据流

`RuntimeMetadata.output_quants` 按设计**原样**携带 SDK 量化描述符（F32 输出
上的 vestigial 描述符必须留在证据里，见 Phase 1.5 H3）。`dataclasses.asdict`
对叶子值执行 `copy.deepcopy`；deepcopy 走 pickle，而板端
`hbm_runtime.HB_HBMRuntime.QuantParams` 不可 pickle → `TypeError`。host 测试
的 fake 描述符（SimpleNamespace 等）可 deepcopy，因此既有 host 测试全绿、
板端必崩——host fixture 与板端失败面不同构，这正是漏测点。

同构调用共六处：

| 文件 | 调用点 |
| --- | --- |
| `samples/vision/yolov5/evaluator/compare.py` | `create()`：`summary['metadata'][side]` |
| `samples/vision/bytetrack/evaluator/capture.py` | `factory()`：`summary['metadata']` |
| `samples/vision/fcos/evaluator/compare.py` | `_metadata_json()`（source/unified/normalise 三条路径共用） |
| `samples/vision/lprnet/evaluator/compare.py` | `create()`：`summary['metadata'][side]` |
| `samples/vision/modnet/evaluator/compare.py` | 同上 |
| `samples/vision/yoloworld/evaluator/compare.py` | 同上 |

## 修复

**新增共享投影 helper**（`samples/_shared/runtime_meta.py`，不改推理侧
`RuntimeMetadata`/`from_runtime`/binding 的对象与语义）：

- `metadata_evidence(metadata)`：按 dataclass 字段（或 Mapping，供 host seam）
  显式投影为 JSON 可序列化结构；不 deepcopy 任何 SDK 对象，不改写输入。
- `_evidence_value` 显式处理：`None/bool/int/float/str`、numpy scalar
  （`.item()`）、`ndarray`（`.tolist()`）、`Mapping`、`list/tuple`、嵌套
  dataclass、以及 SDK 量化描述符（`hasattr(value,'quant_type')`）：固定输出
  `quant_type`（取 enum-like `.name`）、`scale`、`zero_point`、`axis` 四键
  （缺省键为 `null`），并把描述符上其余**公开属性一并投影**，SDK 扩展字段
  不会被静默丢弃。未知对象抛 `TypeError`（带类型名），**不用 `str(object)`
  糊弄结构**。
- numpy 延迟导入（沿用 `quantization.py` 惯例），模块本身保持零 SDK/numpy
  顶层依赖。
- 不改任何检查容差、比较逻辑与证据字段；六处 evaluator 仅把
  `asdict(...)` 换为 `metadata_evidence(...)`（各文件内 `asdict` 仅此一处，
  summary 级 `_json` default 保留原样作为兜底）。`runtime_meta.py` 本就在
  六处 evaluator 的 `code_sha256` 清单内，无证据哈希清单变更。
- `samples/_shared/README.md` Runtime metadata 节补充 `metadata_evidence`
  说明（该目录无 README_cn 层级）。

## 回归测试

**共享层**（`samples/_shared/tests/test_runtime_meta.py`，+7 测试）：

- fixture `_BoardQuantParams`：`__deepcopy__`/`__copy__` 显式抛
  `TypeError: cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object`，
  与板端失败面同构；锚定测试证明该 fixture 确使 `asdict` 复现板端崩溃。
- 覆盖：投影后 `json.dumps(..., allow_nan=False)` 通过；quant 四键与逐通道
  scale/zero_point 数值完整（float32 精确展宽，不经过文本往返）；形状/dtype/
  stride/字段全集保留；投影无副作用（metadata 仍持有原描述符对象、数组值与
  dtype 不变、投影可变不回渗）；额外公开属性保留；Mapping 入参；未知对象
  拒绝字符串化。

**六处接线**（每 sample 一个端到端测试，注入 copy-hostile 描述符跑真实
evaluator 路径，并 `json.loads` 落盘证据文件验证 quant 字段齐全）：

- yolov5 `test_evaluator.py`：`test_metadata_evidence_survives_copy_hostile_board_quant_params`（x5 F32 vestigial + s100 int8 per-channel 双 target）
- bytetrack `test_entrypoints_evidence.py`：`test_capture_metadata_survives_copy_hostile_board_quant_params`
- fcos `test_fcos_contract.py`：`test_evaluator_metadata_survives_copy_hostile_board_quant_params`（CopyHostileQuant 继承既有 Quant，数值路径不变，`checks['metadata']` 双侧投影相等）
- lprnet / modnet / yoloworld 各自 `test_*`：同名测试（F32 + vestigial NONE 描述符）

**阴性对照**：把 yolov5 接线临时还原为 `asdict` 后，上述新测试以
`TypeError: cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object`
失败（x5/s100 两 subTest 均复现板端错误），恢复修复后转绿——证明测试对原
缺陷敏感，而非恒真。

## 验证汇总（host，全部作者本人执行）

| 套件 | 结果 |
| --- | --- |
| `python -m unittest discover -s samples/_shared/tests` | 108/108 OK（基线 101 + 新增 7） |
| `samples.vision.yolov5.tests`（discover 全量） | 33/33 OK |
| `samples.vision.bytetrack.tests`（discover） | 12/12 OK |
| `samples.vision.fcos.tests`（discover） | 25/25 OK |
| `samples.vision.lprnet.tests`（discover） | 13/13 OK |
| `samples.vision.modnet.tests`（discover） | 13/13 OK |
| `samples.vision.yoloworld.tests`（discover） | 19/19 OK |
| `samples/vision/resnet/tests`（AGENTS 指定回归，重导出 RuntimeMetadata） | 52/52 OK |
| `samples/vision/paddle_ocr/tests`（重导出 RuntimeMetadata） | 44/44 OK |
| `samples/vision/ultralytics_yolo/tests` | 76/78；2 ERROR 为**改动前已存在**的环境缺件：需要 `npm --prefix tools/catalog-publisher run build` 产物 `dist/catalog.json`（本工作树未构建；两失败测试与本次改动文件无交集） |
| `py_compile` 全部改动文件（`-W error::SyntaxWarning`） | 通过（修复了一处 docstring 反斜杠转义警告） |

**未运行**：X5/S 板端 compare 重跑（无板卡授权）；独立 review（属协调者
流程）。

## 边界与交接

- **未板测**：修复消除的是 host 可证明的 deepcopy 失败面；板端
  `QuantParams` 真实结构仅按 SDK 公开属性（quant_type/scale/zero_point/axis
  + 其余公开字段）读取，最终以 X5 yolov5 compare 重跑为准。协调者板测前，
  本修复状态为 host-passed / board-not-run。
- **未独立通过**：本记录为作者自检，未经 `rdk-model-zoo-review` 独立审阅。
- **同类模式（未动，超出本次六处授权范围）**：
  `samples/_shared/sam_evaluator.py:215` 存在相同的
  `asdict(RuntimeMetadata.from_runtime(runtime))` 模式（B6 SAM 证据路径），
  其 fake fixture 同样 deepcopy-able，板端风险与本缺陷同构。建议协调者
  单独立项接线 `metadata_evidence`。
- 提交状态：源代码侧（helper + 共享测试 + 六处接线）已由协调者在本次会话
  期间以 `a141c48` 收编；六处 sample 测试新增、`samples/_shared/README.md`
  与本记录在工作树中待协调者提交。作者未执行任何 git commit/push/merge。
