# B7 S100 signed dtype 规范化修复（作者记录）

本记录覆盖 2026-09-24 的一个独立小任务：修复共享 `canonicalise_dtype` 不识别真实 SDK signed dtype 别名（`S8`/`S16`/`S32`）导致的 YOLOv5 S100 实板启动失败。任务由 GLM（Claude Code 会话）单独执行，等 Codex 提交/复验；不复述、不改写其他报告的结论。

**状态口径：**

- 全程**未 commit / 未 push / 未 merge**；仅改动本任务专属的三个文件（见 §3）。
- **Board = not-run**：本任务不连板卡；板端复跑 `main --target s100` 留给协调方统一安排。
- 主机测试通过是作者自检，不等于独立验收，也不等于实板修复证明（实板修复以板端复跑 rc=0 为准）。

## 1. 缺陷与证据

S100 实板 smoke（固定源 `5fc14a4` 的部署副本）失败：

- `.coordination/b7-s100-python-smoke.json`：`samples/vision/yolov5/runtime/python/main.py --target s100` 返回 **rc=2**，stderr 为 `Error: Unsupported native output dtype 's32'.`
- 同目录 `b7-s100-native-metadata.json`：真实 `HB_HBMRuntime` 在 S100 上上报 `output_dtypes = {'output': <hbDNNDataType.S32: 8>, '1310': ..., '1312': ...}`（枚举 `.name` 为 `S32`），输入为 `<hbDNNDataType.U8: 3>`；三个检测头形状 `(1,84,84,255)`/`(1,42,42,255)`/`(1,21,21,255)`。

根因：`samples/_shared/runtime_meta.py` 的 `canonicalise_dtype` 只接受 `i32`/`int32`/`hbdnndatatype.int32` 等 int 拼写；真实 SDK 枚举名 `S32` 经 `.name` 小写后为 `s32`，不命中任何分支，按未知令牌原样返回，随后被 `yolov5/runtime/python/model_binding.py:bind_model` 的 S 侧允许集（`float32/int8/uint8/int16/int32`）拒绝。X5 不受影响（raw_f32 路径输出本就是 F32）。

## 2. 修复方案

`canonicalise_dtype`（`samples/_shared/runtime_meta.py`）为每个 signed 宽度补齐真实 SDK 拼写，规范到既有 canonical 形式：

| 原生令牌（经 `.name` 小写） | canonical |
| --- | --- |
| `s8`、`hbdnndatatype.s8`（含 `.s8` 后缀） | `int8` |
| `s16`、`hbdnndatatype.s16`（含 `.s16` 后缀） | `int16` |
| `s32`、`hbdnndatatype.s32`（含 `.s32` 后缀） | `int32` |

- 带 `.name` 属性的枚举对象走既有 `getattr(dtype, "name", dtype)` 路径，无需新代码分支；带 enum 前缀的字符串（如 `"hbDNNDataType.S32"`）由 `.s32` 等 `endswith` 分支覆盖，与既有 `.int32`/`.f32` 写法一致。
- **行为保留**：既有令牌（`f32`/`i8`/`u8`/`i16`/`i32`/`f16`/`nv12` 及其变体）的映射不变；未知令牌（含 `s64`、`hbDNNDataType.WEIRD`、`S128`）继续原样返回，不扩大成对未知类型的猜测。
- 消费侧不改：`bind_model` 的 S 侧允许集本就含 `int32`，源头规范后即可通过。

## 3. 变更文件（仅此三个）

| 文件 | 变更 |
| --- | --- |
| `samples/_shared/runtime_meta.py` | `canonicalise_dtype` 增加 s8/s16/s32 别名（+6/−3 行），含说明注释 |
| `samples/_shared/tests/test_native_signed_dtypes.py` | 新增，10 个用例（见 §4） |
| `docs/releases/unified-migration/2026-09-24-b7-dtype-remediation.md` | 本记录 |

未改任何 YOLOv5 样例文件、manifest、其他 sample 或共享模块；native C++ 与 metadata serializer 属其他 worktree 的并行任务，本任务未等待也未触碰。

## 4. 测试与结果

新增 `samples/_shared/tests/test_native_signed_dtypes.py`（全为行为断言，无源码 substring 检查）：

- **真实 enum-like 对象**：本地 `enum.IntEnum`（成员值取自 S100 证据：`U8=3`、`S32=8`）→ `int32`/`uint8`；仅带 `.name` 的包装对象 → `int8/int16/int32`；signed 与 unsigned 无串扰（`s8 ≠ uint8`、`s32 ≠ uint32`）。
- **字符串**：裸 `s8/s16/s32`（含大写）与 enum 前缀 `"hbDNNDataType.S32"` 等三种宽度全部规范正确。
- **负例**：`weird`/`s64`/`hbDNNDataType.WEIRD`/`S128` 原样返回（`s64` 不得映射为 `int64/uint64`）；`None → None`；既有令牌（`F32`/`U8`/`i32`/`int16`/`f16`/`nv12`）canonical 形式不变。
- **真实形状路径**：以 S100 证据元数据（单模型 `yolov5x_672x672_nv12`、`data_y`/`data_uv` 输入、三个 S32 检测头及证据形状）驱动 `RuntimeMetadata.from_runtime`，断言 `output_dtypes == {"output": "int32", "1310": "int32", "1312": "int32"}`、`input_dtypes` 为 `uint8`、形状逐项保留、且各输出 dtype 落在 `bind_model` 的 S 侧允许集内；另覆盖 `from_mapping` 收 enum 前缀字符串的路径。

执行结果（解释器 `/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk_model_zoo/.venv/bin/python`，cwd 为 worktree 根）：

| 套件 | 结果 |
| --- | --- |
| `samples._shared.tests.test_native_signed_dtypes`（新增） | **Ran 10 tests — OK** |
| `unittest discover -s samples/_shared/tests`（全量 shared，含新增与既有 dtype 令牌用例） | **Ran 111 tests — OK** |
| `unittest discover -s samples/vision/yolov5/tests`（全部 5 个模块） | **Ran 32 tests — OK** |

既有 `test_runtime_meta.py::test_dtype_tokens_are_canonicalised`（含 `canonicalise_dtype("weird") == "weird"` 回归断言）在全量 shared 套件内通过，确认既有行为未变。

## 5. 未完成项 / 移交

- **板端复验 not-run**：需在 S100 实板重跑 `python3 samples/vision/yolov5/runtime/python/main.py --target s100`，以 rc=0 关闭 `b7-s100-python-smoke.json` 的 rc=2。
- 修复只覆盖 Python 侧 dtype 规范化；C++ 侧 signed dtype 处理与 metadata 序列化由并行任务负责，本记录不声明其状态。
- 本任务结束后停止，等 Codex 提交/复验；不做跨任务合并决策。
