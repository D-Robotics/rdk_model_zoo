# 全仓可读 Runtime 迁移说明（2026-10-05）

分支：`codex/readable-model-examples-20261001`（基线 `eba4adbe`，沿用本地开发
worktree；`develop` 与远端未动）。方案见
[全仓设计](../superpowers/specs/2026-10-05-all-sample-readable-runtime-design.md)，
架构说明见 [docs/architecture/model-examples.md](../architecture/model-examples.md)，
ResNet/YOLO 首轮映射见 [2026-09-30-model-examples.md](2026-09-30-model-examples.md)
（不受本文影响）。本文记录 2026-10-05 起把已认可的可读 Runtime 形态推广到
**全部 51 个本仓样例**的旧新映射。

状态声明：以下映射由各批次执行器在共享 worktree 内实现并自测（日志在
`local-execution/20261005-all-sample-readable-runtime/<batch>/`，批次报告在
`docs/releases/unified-migration/2026-10-05-all-sample-<batch>.md`）；编写本文时
部分批次仍在收尾。**逐样例行为验收、全仓回归与干净 checkout 复现由 Codex 统一
评审收口**，结构化状态见
[2026-10-05-all-sample-coverage.json](../releases/unified-migration/2026-10-05-all-sample-coverage.json)
（`implementation_evaluation: pending_final_review`）。本文不宣称全仓已完成。

范围：`samples/{vision,speech,robotics,llm}` 下 51 个本仓维护样例（45 vision、
3 speech、1 robotics、2 llm）。ACT/Pi0 为 pinned gitlink，按用户要求继续排除。
两个 LLM 样例只有原生 C++ 运行时，保留原生接口，不伪造 Python Runtime。

## 1. 通用旧新映射（各类别共用）

### 1.1 阶段方法名（全部 Python 样例）

| 旧名（保留为兼容委托） | 首选名（canonical） | 说明 |
| --- | --- | --- |
| `pre_process` | `preprocess` | 同一实现，旧名为薄 alias，不维护第二份代码 |
| `forward` | `infer` | 同上 |
| `post_process` | `postprocess` | 同上；消费 context 的任务参数形态不变 |
| — | `predict` | 显式串联 `preprocess`/`infer`/`postprocess`，返回既有 Result 契约 |

两个**合法例外**（同一实现、两个名字，但方向相反，不是"旧方法反向委托新方法"）：

- 共享 SAM stage（`samples/_shared/sam_stages.py` 的 `EncoderStage`/`DecoderStage`）
  的公共实现保留旧名；efficient_sam/mobile_sam 的本地 `*Encoder`/`*Decoder` 以
  canonical 名为公开视图、委托继承到的共享数值实现——本地 canonical 视图与
  共享旧名表面是**同一实现的兼容表面**，共享模块不为了改名而复写。
- 21 分类的共享 `classification.ClassificationTask` 原样保留（兼容出口，公共面
  只有旧名 + `predict`，不带 canonical 方法）；canonical 主线在各样例本地
  `classify.py` 新类中，`infer` 同时接受 `PreparedInput` 与裸 tensors 映射、
  `forward` 只委托 `infer`（2026-10-05 委托修正见
  [classifier-alias-fix](../releases/unified-migration/2026-10-05-all-sample-classifier-alias-fix.md)）。

静态检查器 `tools/sample_contract/check.py` 自 2026-10-05 起对两套拼写（含
`preprocess_*`/`infer_*`/`postprocess_*` 与旧前缀、`run_*`）同等做纯度扫描；
`cli.py`/`yolo_cli.py` 的模块级应用函数（如 `run_prepare` 下载、`run_list_models`）
记为 skip 的应用边界，类方法与其余文件照常检查。

### 1.2 单模型样例形态

| 旧形态 | 新形态 | 兼容保留 |
| --- | --- | --- |
| `main.py` 内联完整运行流程（或藏于 `_run`） | 薄入口：解析参数 → model-free 模式委托 → 显式构造模型对象 → `predict` → 展示 | CLI 参数/默认值/返回码不变；`build_parser` 等仍可从 `main` 导入 |
| 入口内选项声明/展示/文件 IO | 样例本地 `cli.py`（YOLO 为 `yolo_cli.py`） | 不搬整个旧流程，入口仍可见构造与 predict |
| 共享 `ClassificationTask` 转出（分类 21 样例） | 本地具名分类类 `classify.py:<Name>Classifier`，完整三步主线可见 | `classification.py` 共享类再导出保留，旧导入不变 |
| 任务模块内旧阶段名 | canonical 名为实现，旧名薄 alias | 旧调用方零改动 |

### 1.3 多阶段样例形态

每个 stage 公开三步接口（canonical 名 + 旧名委托），`pipeline.predict` 显式编排
（检测→裁剪→识别、encoder→decoder 等）；**不强制 pipeline 顶层只有三个方法**，
`run_detection` 这类 stage composer 与 `prepare_*`/`encode_image` helper 合法。
阶段错误归属（哪个 stage、哪个 crop）保持。

### 1.4 原生（C++）样例

`samples/llm/gemma4-e2b`：模型执行接口是真实引擎类 `gemma4::TextEngine`
（`gemma4_text_engine.hpp`：`Generate`/`GenerateStream`/
`GenerateWithPromptEmbeddings`/`ContinueGenerate(Stream)`/`ResetSession`）与
`gemma4::VisionEngine`（`gemma4_vision_engine.hpp`：`Infer`）；不存在名为
`Gemma4TextEngine` 的类。本轮把交互式入口收薄：`src/main.cpp` 为薄入口，新增
应用 facade `gemma4::chat::InteractiveChatApp`（`gemma4_chat_app.hpp/.cpp`）
承接 REPL/终端 IO——它是**应用类而非模型类**，不存在带控制台 IO 的 predict
型引擎 API，引擎从不隐式打印，模型工作全部委托真实引擎。详见
[speech_policy_native 批次报告](../releases/unified-migration/2026-10-05-all-sample-speech_policy_native.md)。
`samples/llm/minicpm5-2b`：`minicpm5::MiniCPM5::Generate(prompt, new_chat)`
（`pre_process`/`infer`/`post_process` 串联、会话重置语义）经本轮复核已符合
原生标准，零改动保留；`runtime/legacy` 照旧。两者均不新增 Python Runtime。
首轮文档（2026-09-30）所称 C++"未做 Python 式逐方法重构"仅限定首轮范围；
本轮除上述 Gemma 应用边界整理外，其余样例的 C++/legacy 运行时维持既有能力
与构建方式，本轮未改后端语义（共享 `samples/_shared/runtime.py:RuntimeSession`
仍只做薄 SDK 加载与目标身份检查）。

## 2. 分类样例（21，批次 classifiers；ResNet 为既有范例）

各样例新本地类（文件统一为 `runtime/python/classify.py`，入口 `main.py` 薄化、
新增 `cli.py`；`classification.py` 保留共享类兼容导入）：

| 样例 | 本地类 | 样例 | 本地类 |
| --- | --- | --- | --- |
| convnext | `ConvNeXtClassifier` | mobilenetv3 | `MobileNetV3Classifier` |
| edgenext | `EdgeNeXtClassifier` | mobilenetv4 | `MobileNetV4Classifier` |
| efficientformer | `EfficientFormerClassifier` | mobileone | `MobileOneClassifier` |
| efficientformerv2 | `EfficientFormerV2Classifier` | repghost | `RepGhostClassifier` |
| efficientnet | `EfficientNetClassifier` | repvgg | `RepVGGClassifier` |
| efficientvit | `EfficientViTClassifier` | repvit | `RepViTClassifier` |
| fasternet | `FasterNetClassifier` | resnext | `ResNeXtClassifier` |
| fastvit | `FastViTClassifier` | vargconvnet | `VargConvNetClassifier` |
| googlenet | `GoogLeNetClassifier` | vit | `ViTClassifier` |
| hgnetv2 | `HGNetV2Classifier` | （resnet | `ResNetClassifier`，2026-09-30 范例） |
| mobilenetv1 | `MobileNetV1Classifier` | mobilenetv2 | `MobileNetV2Classifier` |

预处理/量化/输出契约按各 binding 实际协议复用共享实现（不假设全部 NV12 或同一
softmax 策略）；`labels`/`top_k`/output transform 语义不变，未新增自训练导出支持。

## 3. 视觉任务样例（13，批次 vision_tasks；其中 13 样例的入口抛光见 §6，第 14 个 yoloe 见 §4）

| 样例 | 模型类（文件） | 特殊输入/状态 |
| --- | --- | --- |
| 3dresnet | `VideoClassificationTask`（classification.py） | 16 帧视频片段输入 |
| depth_anything_v2 | `DepthAnythingV2Task`（depth_anything_v2.py） | 逐调用几何；`return_details`（`DepthPredictionDetails`）可选携带 raw 张量（`raw_depth.npy` 归档来源），默认返回不变 |
| diffusiondrive | `DiffusionDriveTask`（diffusiondrive.py） | prepared-feature 规划；npz 应用输出；`return_details` |
| dinov2 | `DINOv2Task`（embedding.py） | 特征嵌入 |
| fcos | `FCOSTask`（fcos.py） | NMS 仅在 postprocess |
| lanenet | `LaneNetTask`（lanenet.py） | 车道嵌入+二值分支；raw 输出经 `LanePredictionDetails` 保留 |
| lprnet | `LPRNetTask`（lprnet.py） | 车牌字符解码 |
| modnet | `MODNetTask`（modnet.py） | 抠图 alpha |
| pointnet | `PointNetTask`（pointnet.py） | 点云输入（非图像）；归一化张量/centroid/radius 上下文经 `PointNetPredictionDetails` 保留 |
| pp_liteseg | `PPLiteSegTask`（pp_liteseg.py） | 语义分割 |
| unet | `UNetTask`（unet.py） | 语义分割 |
| unetmobilenet | `UnetMobileNetTask`（unetmobilenet.py） | 语义分割 |
| yolo26_depth | `Yolo26DepthTask`（yolo26_depth.py） | 默认 `warmup=0`、返回旧 `DepthResult` 不变；`warmup` 为显式 predict 参数（独立于 details），单次前向 latency 每次调用都计时、仅经 `return_details=True` 的 `DepthPredictionDetails(result, prepared, raw, warmup, latency_ms)` 暴露，CLI 以 details 一次调用取齐（已实施，评估待终审） |

## 4. 检测/跟踪样例（5，批次 detection_tracking）

| 样例 | 模型类（文件） | 特殊语义 |
| --- | --- | --- |
| ultralytics_yolo | `YoloDetect`（detect.py）及 `yolo_cls/seg/pose`、`yolo26_det/seg/pose/obb`、`yolo_v10detect` | 家族×任务分派；DFL/LTRB/NMS-free 协议各自保留；`YOLO26OBB` 复用共享图像传输的 `preprocess`/`infer` |
| yoloe | `YOLOE`（yoloe.py） | 免提示实例分割 |
| yolov5 | `YOLOv5Task`（detection.py） | 单家族检测 |
| yoloworld | `YOLOWorldTask`（yoloworld.py） | 开放词表（文本+图像） |
| bytetrack | `ByteTrackTask`（tracking.py） | 跨视频帧的有状态跟踪 |

## 5. 多阶段/语音/策略/原生（multistage + speech_policy_native）

| 样例 | 编排主体 | 特殊语义 |
| --- | --- | --- |
| clip | `CLIPTask`（matching.py）+ cli.py | 双编码器图文匹配（tokenizer 文本路径） |
| siglip | `SigLIPTask`（embedding.py）+ cli.py | 双编码器图文嵌入/匹配 |
| efficient_sam | `EfficientSAMEncoder`/`EfficientSAMDecoder` + `EfficientSAMPipeline`（pipeline.py） | encoder→提示 decoder；共享 SAM 实现的本地 canonical 视图 |
| mobile_sam | `MobileSAMEncoder`/`MobileSAMDecoder` + `MobileSAMPipeline` | 同上（mobile 变体） |
| paddle_ocr | `OCRPipeline`（pipeline.py）+ cli.py | 检测→有序裁剪→识别；逐 crop 错误归属；零检测短路 |
| paraformer | `EncoderStage`/`PredictorStage`/`DecoderStage`（stages.py）+ `ParaformerPipeline`（pipeline.py）+ cli.py | **离线逐 utterance** 三模型转录：encoder→predictor→CPU CIF（`cif_numpy`）→decoder，零 token 跳过 decoder；无跨 chunk 流状态（流式 chunk 模式属 asr 样例，两者不混称）。main 薄入口正由 speech/policy 入口批次改成显式 `pipeline.predict` 小循环 + application 的 prepare/record 助手（当前快照已可见该形态）；最终形态与行为验收以 Codex 终审为准 |
| asr | `ASR`（asr.py） | 分块流式（chunk 间无模型状态）：与 Paraformer 的 CIF 三模型编排是两种不同模式。当前源码快照 main 为薄入口、逐 chunk 单次 `predict(..., return_details=True)`；该批进行中，实施状态待评估 |
| kws | `KWS`（kws.py） | 滑窗关键词检出 |
| himloco | `HimLocoTask`（policy.py） | 六帧观测+本体历史→动作；无机器人控制；main 薄入口显式逐观测 `task.predict`（application 承接 prepare/记录，入口重整同批进行中、待评估） |
| gemma4-e2b | `gemma4::TextEngine`/`gemma4::VisionEngine`（runtime/cpp 引擎）+ `gemma4::chat::InteractiveChatApp`（应用 facade，非模型类） | 原生 Generate/GenerateStream/ContinueGenerate(Stream)/ResetSession 与 VisionEngine::Infer；控制台 IO 只在 facade；无 Python Runtime |
| minicpm5-2b | `minicpm5::MiniCPM5`（runtime/cpp，legacy 保留） | 原生 `Generate(prompt, new_chat)` 串联 pre_process/infer/post_process；本轮复核已达标、零改动；无 Python Runtime |

## 6. 入口抛光（entrypoint_polish_vision，评审驱动）

Codex 入口评审（`local-execution/.../entrypoint-polish-review.md`）要求：main 必须
显式构造模型并调用 `predict`；对有合法中间输出需求的任务（DepthAnything 的
raw_depth.npy、DiffusionDrive 的 npz、PointNet 的归一化上下文、LaneNet 的 raw
输出、YOLO26Depth 的 warmup/单次时延）采用**可选 `return_details`** 模式：默认
`predict` 返回与旧版完全一致的结果，一次请求仍只跑一次生产推理，不做第二次推理
链来补输出。按当前源码：五个 details 类（Depth/DiffusionDrive/Lane/PointNet/
YoloDepth 的 `*Details` typed 记录）均已在各自模型文件实施，14 个 main 均为
显式 `predict` 入口（8 个新增本地 `cli.py`，6 个既有薄入口零改动复核；yoloe 自
detection_tracking 移交本批）。Yolo26Depth 口径（见 §3）：默认 `warmup=0` 且返回
旧 Result；`warmup` 为显式参数，latency 每次计时、仅 details 暴露——CLI 旧行为
不变。speech/policy 侧（asr 的逐 chunk `return_details`、himloco/paraformer 入口）
由 speech_policy 入口批次进行中（asr 的 details 已出现在当前源码快照），实施状态
待评估。以上均为源码快照事实；逐样例行为等价与全仓回归属 Codex 终审，本文不
预支结论，批次报告见
[entrypoint_polish_vision](../releases/unified-migration/2026-10-05-all-sample-entrypoint_polish_vision.md)
（批次快照，最终状态以 Codex 终审报告为准）。

## 7. 验证边界

- 各批次：先跑所属样例基线，新增接口先写失败测试再实现（failing-first 日志在
  批次目录）；每样例独立进程 discover。
- 本文/契约/检查器批次（integration_docs_checker）：checker 自身 32 项 fixture
  测试通过（含 canonical 纯度新规则与 cli 边界）；ResNet 与 ultralytics_yolo
  契约检查 0 violations。未跑全仓契约检查（其他样例仍在被并行编辑）。
- 全仓逐样例回归、干净 checkout 入口复现、Catalog build/check、板端 SDK 推理：
  未执行（not-run），待 Codex 统一集成安排；本文不以其通过为前提。
