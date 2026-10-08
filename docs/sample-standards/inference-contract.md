# 推理职责与接口契约（inference-contract）

> 状态：Phase 0.5 Q2 基线（2026-09-20）；2026-10-05 修订：可读 Runtime 推广后
> `preprocess`/`infer`/`postprocess` 为首选阶段名，旧名 `pre_process`/`forward`/
> `post_process` 保留为同一实现的兼容委托（见 `docs/migration/
> 2026-10-05-all-sample-readable-runtime.md`）。适用范围：`develop` 统一 samples
> 架构下全部 sample 的 Python 运行时业务接口。C++ 遵循对应 sample README 声明的
> 结构约定，不强制逐方法镜像本契约；原生生成接口（`Generate`/stream/reset）见 §3。

## 1. 单模型公开业务接口

单模型任务的公开业务接口统一为四个方法：首选（canonical）阶段名
`preprocess` → `infer` → `postprocess`，由 `predict` 串联。既有
`pre_process`/`forward`/`post_process` 名称保留为**同一实现的薄兼容委托**
（通常 alias 调 canonical，不维护第二份实现）；共享 SAM stage 与既有共享分类兼容
出口允许复用原实现、本地提供 canonical 视图，具体边界见上述迁移说明。静态
检查器对两套拼写同等做纯度扫描。
允许另加初始化、必要调度/资源生命周期接口（如 `set_scheduling_params`、
`close`）和委托 `predict` 的 `__call__`。**不以“整个文件只能四个函数”作为
规则**——辅助函数与本地模块见 §4。

```python
prepared = model.preprocess(inputs)       # tensors + 本次调用的 context
outputs = model.infer(prepared.tensors)   # RawOutputs（结构/metadata 校验后）
result = model.postprocess(outputs)       # 消费 context 的任务在此取回 context
model.predict(inputs)                     # 必须串联以上三步，返回同一 Result 契约

# 兼容委托（同实现，不另写一份）：
model.pre_process(inputs) == model.preprocess(inputs)
model.forward(tensors)    == model.infer(tensors)
model.post_process(out)   == model.postprocess(out)
```

`postprocess` 是否接收 context 由任务是否消费几何信息决定：检测/分割等需要坐标还原
的任务显式传入（`postprocess(outputs, prepared.context)`）；纯分类等不消费 context 的
任务可在 docstring 说明后省略该参数。两种形态都符合本契约，但 **predict 的串联语义与
三步显式调用的结果一致性必须在 sample tests 中验证**。

单模型任务在其 sample 本地模型文件（如 `classify.py`、`detect.py` 或任务同名模块）
中提供**具名模型类**：初始化、`preprocess`/`infer`/`postprocess`/`predict` 的真实
主线在一个文件内可读，不是空子类或共享类的转出；可复用的张量/数学/绑定/runner
helper 继续留在共享或本地模块。`main.py` 是薄入口：解析参数、model-free 模式委托、
显式构造模型对象、调用 `predict`、展示结果；参数声明、展示、文件 IO 等应用辅助
可放在 sample 本地 `cli.py`（不得把整个旧运行流程搬进 cli 再让入口隐藏模型构造）。

## 2. 阶段数据流的概念契约

每个任务在 docstring/类型定义中**具体化**以下五个概念，不要求全仓新增通用基类：

| 概念 | 含义 | resnet 实例 | paddle_ocr 实例 |
| --- | --- | --- | --- |
| Input | 业务输入（验证后） | BGR `np.ndarray` | BGR 图像 |
| Tensors | 后端物理输入映射 | NV12 packed/split 字节张量 | 检测 NV12 / 识别 RGB F32 NCHW |
| Context | 本次调用的几何/状态信息 | `ImageTransform`（尺寸、scale、padding） | 检测：原图几何；识别：crop 身份 |
| RawOutputs | 结构与 metadata 校验后的原始输出 | F32 分数张量 | 验证后的平坦输出映射 |
| Result | 业务结果（owned、有序） | `ClassificationResult` | `OCRResult` / `DetectionResult` |

**Context 纪律**：context 必须显式保存在 `prepared` 返回值中（frozen 结构或等价物），
不允许放入会被下一次调用覆盖的实例字段。有状态任务（跟踪、语音流、生成式）必须显式
声明 session/state/reset 语义并说明是否线程安全；不虚称线程安全。

## 3. 职责边界

| 模块/阶段 | 职责 | 禁止混入 |
| --- | --- | --- |
| preprocess（旧名 `pre_process`） | 业务输入验证、数值/布局变换、构造本次调用上下文 | CLI、下载、数据集遍历 |
| infer（旧名 `forward`）/ runner | 匹配后端加载/调用、结构与 metadata 校验、明确的输出容器适配 | NMS、任务解码、反量化/激活业务变换、绘图、保存结果、评估 |
| postprocess（旧名 `post_process`） | 按制品契约反量化/激活、解码、坐标还原、业务结果 | 模型下载、再次推理、报告/文件输出 |
| predict | 串联 canonical 阶段、返回业务结果 | 第二套前后处理实现 |
| main | CLI、输入读取、显式构造模型、调用 predict、结果展示/保存 | 另写模型算法、把构造藏进 `_run` |
| 本地 cli（`cli.py`） | 参数声明、model-free 模式、展示/文件 IO 等应用辅助 | 整个旧运行流程的搬运（入口仍须显式构造模型并 predict） |
| binding | target/stage/asset 的张量与语义契约 | CLI、文件下载、任务编排 |

说明与豁免边界：

- NMS、CTC、几何操作等相关辅助函数允许存在；复杂或复用逻辑拆成职责明确的本地模块，
  禁止把杂项塞进泛化 utils，也不为满足函数数目限制把大段代码塞进一个方法。
- 可视化由 main 调用专门模块完成，不进三阶段。
- runner 可以暴露量化 metadata（`output_quants` 等），但 H1 的数值变换（dequant 与激活
  语义）由 postprocess 或其明确委托模块执行——infer 只做容器适配，不改数值语义。
- 多阶段任务（OCR/SAM/Paraformer）：每个 stage 具备**公开**的
  preprocess/infer/postprocess 三步接口（旧名同为实现委托），`pipeline.predict`
  负责显式编排；阶段次序必须在代码中直接可读（检测→裁剪→识别、encoder→decoder），
  禁止为凑三函数把下一阶段推理藏入上一阶段 postprocess。**不强制 pipeline 顶层只有
  三个方法**：stage composer（如 `run_detection`）、逐阶段 helper（如
  `prepare_detection`/`encode_image`）是合法的公开辅助，只要每个 stage 的三步仍公开
  可单独调用。
- LLM/流式任务可声明 generate/stream/reset 接口并写明理由，不机械套单图模型。
  仅有原生 C++ 运行时的 sample（如 `samples/llm/*`）以**原生等价接口**为准——
  例如 `Gemma4TextEngine::Generate/GenerateStream/ResetSession` 与
  `MiniCPM5::Generate` 是其 README 声明的结构约定，不为可读性伪造 Python Runtime
  镜像，也不把两者混写为同一协议。
- 异常归属：跨阶段封装必须保留“哪个 stage 失败”的信息（如
  `recognizer stage failed for crop {i}`），不得吞掉或混淆。

## 4. 文件组织

- 文件组织遵循 [Sample Runtime 代码规范](runtime-code.md)。普通单任务 Sample 默认使用
  `main.py`、`cli.py` 和具名模型文件；模型文件承载初始化和完整推理阶段，CLI 负责参数、
  发布模型选择和结果交付，入口显式构造模型并调用 `predict`。
- 模型类通过构造方法或具名类方法建立 Runtime。公共 SDK 和张量能力直接复用
  `utils/py_utils/`；模型专有算法留在模型文件。复杂绑定、分词器、跟踪器等完整职责
  可独立成文件，不要求每个阶段都拥有一个文件，也不增加纯转发模块。
- 公开 API 的 shape、dtype、布局、值域、坐标约定及异常在 docstring 中精确说明；
  README 只给摘要与使用例（见 readme-contract §5.6）。

## 5. 必需测试（sample tests）

每个单模型 sample 至少覆盖：

1. **predict 与显式三步一致**：`predict(img)` 的 Result 与
   `postprocess(infer(preprocess(img).tensors))` 逐字段相等（对保留旧名的
   sample，旧名委托与 canonical 结果一致也须覆盖）。
2. **infer 纯度**：注入的 runner 返回固定 fixture 时，infer 的返回值就是该 fixture
   的校验后形态——无解码、无激活、无文件读写、无坐标变换。
3. **context 不串扰**：两种尺寸输入交错调用（A、B、A），每次 `prepared` 的
   context/transform 只描述当次输入。
4. **原始输出不被未声明变换**：binding 声明之外的数值变换（如未声明的 softmax/dequant）
   使测试失败。

多阶段 sample 另需覆盖：零检测结果短路（不触发识别）、多 crop 逐个识别且次序稳定、
阶段错误归属正确（哪个 stage、哪个 crop）。

异常输入按各 sample 协议验证；不宣称 SDK 并发可用。

## 6. 参照实现

- 单模型：`samples/vision/resnet/runtime/python/classify.py`
  （`ResNetClassifier`；`PreparedInput.tensors + .transform` 即 Tensors+Context，
  旧名 `pre_process`/`forward`/`post_process` 为薄 alias）。
- 多阶段：`samples/vision/paddle_ocr/runtime/python/pipeline.py`
  （`OCRPipeline`；每 stage 的 preprocess/infer/postprocess 与 `predict` 编排）。

两参照在 Q2 落地时按本契约调整并在 Q4 双视角验收；2026-10-05 起全仓 51 个本仓
sample 按同一门槛推广（映射见 `docs/migration/
2026-10-05-all-sample-readable-runtime.md`，逐样例验收状态以
`docs/releases/unified-migration/2026-10-05-all-sample-coverage.json` 为准），
本轮源码架构和主机验证已由 `docs/releases/unified-migration/
2026-10-05-all-sample-codex-review.md` 收口；真实板端执行与发布验收另行记录。
