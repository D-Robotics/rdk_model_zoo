# Phase 1.5 — 架构硬化 H1–H4（2026-09-21）

批量迁移（B1–B11）前的门槛步骤：在一个 commit 序列内落地 H1 输出变换链、
H2 packed NV12 扁平化、H3 元数据多模型、H4 输出 rank 归一化，全部改动收敛在
`samples/_shared/` 与已迁移 sample 的代码/测试内。H5（Profile 契约）与 H6
（sample↔manifest 覆盖检查）按计划随 B1 首批落地，不在本记录内。

## H1 — 输出变换链（dequant + 激活语义归属）

**新增 `samples/_shared/quantization.py`**：

- `dequantize_tensor`/`dequantize_outputs` 从交付分支
  `utils/py_utils/postprocess.py`（源记录：rdk_s @ 380e1a2）移植：per-tensor 与
  per-channel（axis 广播）SCALE 反量化；非 SCALE 类型原样透传；标量 zero_point
  在 per-channel 分支按源语义置零。
- `OUTPUT_TRANSFORMS = ("raw_f32", "dequant")` + `apply_output_transform`：
  raw_f32 透传并**拒绝**整型 dtype 与已上报的量化描述符（pilot 时代的
  reject-int8 由隐式硬编码变为**声明式选项**）；dequant 要求每个输出都有
  output_quants 描述符并返回 float32。
- **激活语义不进本链**：raw logit 是否需要 sigmoid、dequant 后是否已激活是任务级
  事实，由各 sample 的 post_process 声明并执行（X5 YOLO raw→sigmoid 与 S
  dequant 后已激活的区分在 B7/B9 迁移时落地）。

**落点**：`ClassificationContract` 新增 `output_transform` 字段（已发布 ResNet
制品声明 `raw_f32`）；`ModelBinding` 快照 `output_quants` 与
`output_transform`；`bind_model` 按声明分流校验（raw_f32：F32+无描述符；
dequant：int8/uint8/int16/int32 + 必须有描述符）；`classification.post_process`
执行声明变换后再走 legacy_softmax；runner 仅做容器校验。

**验证**：合成 fixture（int8 输出 + SCALE 描述符，经 `dataclasses.replace`
构造 dequant 契约）走通 bind→post 全链，数值手算核对（per-tensor/per-channel/
标量 zp 置零/非 SCALE 透传）；raw_f32 + 描述符 → 拒绝。S 制品真值元数据需板端，
**未运行**（用户门禁）。

## H2 — packed NV12 扁平 canonical

- **canonical = flat 1-D**：`as_packed(y, uv)` 产出 `H*W*3/2` 字节连续 uint8
  （224×224 → 75,264）；`as_split(y, uv)` 显式返回 split 平面。`prepare_nv12`
  的 packed 分支改喂 flat；`validate_input_tensors` 严格只收 flat 1-D（4D 视图
  与错误字节数均拒绝）。
- **元数据声明 shape 校验不变**：X5 输入仍按 `(1,3,H,W)` NCHW 逻辑形校验
  （bind_model 未动）；仅统一喂入布局。
- `pack_nv12_single` 返回 flat（字节与旧 4D 视图相同）。
- 受影响断言更新：test_classification / test_runner / test_integration（legacy
  X5 适配器现在喂 flat，同字节）。
- **X5 板冒烟确认 flat ≡ 4D 等价：未运行**（用户板测门禁；随下次板端会话或 B1
  冒烟执行，此前不声称板端等价）。

## H3 — 元数据多模型泛化

**新增 `samples/_shared/runtime_meta.py`**（resnet 与 paddle_ocr 各自的
RuntimeMetadata 收敛为共享类再导出——两个真实消费者，满足 §5.1 准入）：

- `from_runtime(runtime, model_name=None)`：**多模型 runtime 必须显式选择**，
  不再静默取 `model_names[0]`；选择名不存在即报错。
- 任意数量输出张量可表示（单输出规则留在各 sample 的 binding）。
- `output_quants` 按输出名**原样**携带描述符（不 float 化）——F32 契约据此拒绝
  而非静默丢弃；替代了原 `output_scales`/`output_zero_points` 推测性字段。
- 保留 paddle 需要的 `input_strides`/`output_strides`；dtype 规范化统一为
  `canonicalise_dtype`（新增 int8/int16/int32/f16 令牌）。
- 错误层：resnet/paddle 的 `MetadataMismatchError` 同时继承本地 BindingError 与
  共享类。

## H4 — 输出 rank 归一化

- `score_vector_shape(shape, class_count)`：所有单例维坍缩后必须恰为
  `(class_count,)`——`(1,1000,1,1)`（X5）、`(1,1000)`（S）、`(1000,)`、
  `(1,1,1000)` 均通过；`(1,999)`、`(2,1000)`、`(1,500,2)`、`()`、`(1,)` 拒绝。
  **从不静默 flatten 歧义布局**。
- `_contract_for` 删除 `(1,1000,1,1) if x5 else (1,1000)` 双硬编码；B1 的
  resnet50/152 S 变体无需再分目标写形状。
- bind_model、runner `__call__`、`classification._extract_output` 三处全部改用
  rank 规则；`ModelBinding.output_shape` 保留为观察值（诊断用）。

## 验证汇总

| 套件 | 结果 |
| --- | --- |
| samples/_shared/tests（+19 新测试） | 37/37 OK |
| samples/vision/resnet/tests（+7 新测试，4 处断言更新） | 46/46 OK |
| samples/vision/paddle_ocr/tests（+1 新测试，2 处更新） | 44/44 OK |
| samples/vision/ultralytics_yolo/tests（未改动，回归） | 59/59 OK |
| tools/sample_contract 自身测试 | 23/23 OK |
| Q3 检查器 resnet / paddle_ocr | 0 violations（skip 均为已登记政策） |
| Q3 检查器 ultralytics_yolo | 84 violations = **既有状态**（本步未触碰该 sample；按计划 B9 收编时按同一门槛验收） |

文档同步：resnet 双语 README 的 Results 章改为 flat canonical + rank 规则表述；
`_shared/README.md` 新增 H1/H3 两节并修正 assets.py 清单路径描述（A4 搬迁后
README 仍写 `platforms/{x5,s}/docs/release`，本次一并纠正为
`docs/release/{x5,s}`）。

## 未运行项（如实记录）

- X5 板冒烟 flat packed ≡ 旧 4D 等价（H2 板端确认；用户门禁，随下次板端会话/B1）。
- S 制品真值 output_quants 元数据核验（H1 的 fixture 是合成的；真值需板端）。
- 板端端到端、S600（SSH 未恢复）。

## 结论

H1–H4 完成，3 个已迁移 sample 全量主机测试回归通过，Q3 门禁无新增违规；B5
dinov2、B7 yolov5/bytetrack/fcos、B8 unetmobilenet/lanenet/diffusiondrive、
B9 yolo 系所需的反量化链路与 rank 归一化就位。**Phase 1.5 门槛（H1–H4）达成**；
H5 Profile 契约与 H6 覆盖检查随 B1 落地。下一步：B1 批次（mobilenetv1–v4 +
S resnet50/152 variant 扩展 + H5/H6）。
