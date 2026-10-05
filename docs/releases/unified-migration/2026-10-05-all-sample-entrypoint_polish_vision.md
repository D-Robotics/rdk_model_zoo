# 全部本仓 Sample 可读 Runtime — entrypoint_polish_vision 批次报告

日期：2026-10-05
执行器：本地 Claude Code 2.1.276 + GLM `glm-5.3[1m]`；主机 Python 为
`rdk_model_zoo/.venv/bin/python`（3.14.7）。
输入评审：`local-execution/20261005-all-sample-readable-runtime/entrypoint-polish-review.md`
（Codex 入口评审：3dresnet/dinov2 以 `_run` 隐藏构造；depth_anything_v2、
diffusiondrive、lanenet、pointnet、yolo26_depth 在 main 内组合阶段而非调用
`predict`；yoloe 入口 101 行、CLI/展示占比过高）。
范围：14 个样例独占所有权（vision_tasks 原 13 个 + yoloe）；原执行器已结束，
本批次在其阶段改名成果之上完成真正的入口可读性。代码与测试未提交，由 Codex
按路径评估提交。外部证据日志目录：
`local-execution/20261005-all-sample-readable-runtime/entrypoint_polish_vision/`
（baseline-/failing 阶段见 after-/final-/sdkfree-/contract- 日志与 progress.json）。

> **批次快照声明**：本报告是本批次的执行快照；其中的测试计数、日志与
> "源码完成"表述均以本批时点工作树为准，**最终状态以 Codex 终审报告为准**，
> 本文不构成行为验收结论。

## 本轮改动模式（对应评审要求）

1. **入口形态**：8 个样例的 `main.py` 重写为薄入口——解析参数、处理 model-free
   模式、显式构造 runner/task/model、调用 `predict()`、展示结果；参数声明、
   list/dry-run、报告组装、文件保存移至各样新增的本地 `cli.py`（每样本一个，
   不建框架）。没有把旧 `_run` 整体搬进 cli 再让 main 只调用 `_run`；main 中
   不再出现业务 preprocess/infer/postprocess 串联。
2. **details opt-in（评审批准的具体方案）**：5 个需要中间数据/计时的样例在各
   **现有模型文件**内新增小的本地 typed 记录并在 `predict` 上增加
   `return_details=True` 关键字（默认 `False`，默认返回值与旧 Result 完全一致；
   不在模型上存 last_result/last_image/last_output 字段）：
   - `depth_anything_v2.py DepthPredictionDetails(result, prepared, raw)` — raw
     即 raw_depth.npy 的 `[1,518,686]` 原始张量；
   - `diffusiondrive.py DiffusionDriveDetails(result, physical, raw)` —
     physical_inputs.npz / raw_outputs.npz 归档来源；
   - `lanenet.py LanePredictionDetails(result, prepared, raw)` — raw_outputs.npz
     的具名原始输出；
   - `pointnet.py PointNetPredictionDetails(labels, prepared)` — prepared 含
     精确 `(1,3,N)` 归一化张量与冻结 centroid/radius 上下文；
   - `yolo26_depth.py DepthPredictionDetails(result, prepared, raw, warmup,
     latency_ms)` — `warmup` 为显式预热次数（执行前校验非负），`latency_ms`
     只覆盖单次 infer（含传输校验与 owned 拷贝），不含 pre/postprocessing；
     warmup+1 即 runner 总调用数，无第二条 predict 链。
   五者的 CLI 均改为一次 `predict(..., return_details=True)`，中间产物来自该次
   调用；输出文件 keys/shapes/dtypes、report 字段、返回码与原 CLI 完全等价
   （现有 fake-runner CLI 测试未削弱并全部通过）。
3. **兼容面保持**：`main.build_parser`（contract checker 依赖）、
   `main.RuntimeModelRunner`（现有 CLI 测试 patch 面）、diffusiondrive 的
   `main.add_runtime_arguments`/`main.sha256_file`（run_cases.py 与测试依赖）、
   3dresnet 的 `main.parse_args` 均从 main 再导出；旧阶段名别名、构造签名、
   CLI flags/默认值/返回码、量化/预处理/输出契约与 source provenance 不变。
   `--warmup`/`--priority` 等校验次序与原 main 一致（dry-run 前拒绝非法值）。
4. **yoloe**：main 由 101 行降为 ~55 行薄入口（Config 构造与 `YOLOE`
   构造、`model.predict(image)` 可见）；parser/list/dry-run/结果 JSON 报告移至
   `cli.py`。算法文件 `yoloe.py` 未改。
5. **已合规入口不动**：fcos、lprnet、modnet、pp_liteseg、unet、unetmobilenet
   的 main 已显式构造 task 并调用 `predict`，本轮零改动（仅复核并记录证据）。

## 新增测试（先失败后实现，红绿日志见外部目录）

每个 details 样例新增 `PredictionDetailsTests`：

- 默认 `predict` 仍返回旧 Result 类型且 runner 恰好一次；
- `return_details=True` 与显式三阶段逐张量相等（result/raw/prepared），
  runner 增量恰一次；
- depth/pointnet：A/B/A 不同几何下 details 逐调用独立，模型无残留字段
  （`set(vars(task))` 断言）；
- yolo26_depth：`warmup=-1` 在任何执行前抛 ValueError（runner 与 preprocess
  均未发生）；warmup=2 时 runner 总调用恰 3 次；`latency_ms` 与被测单次前向
  窗口一致（±20ms），且严格不覆盖 preprocess/postprocess 窗口与 warmup 前向
  （时间窗重叠断言）；
- depth CLI 测试补充 raw_depth.npy shape `(1,518,686)`/float32 断言；
  diffusiondrive/lanenet/yolo26_depth/pointnet 的既有 CLI 归档测试
  （physical_inputs.npz、raw_outputs.npz、outputs.npz、raw_tensor_keys、
  warmup==2、report.json 字段）作为回归原样通过。

红绿记录：实现前运行为 `TypeError: predict() got an unexpected keyword argument
'return_details'`（yolo26_depth 另有 `warmup` 关键字），逐样例的失败输出摘录
保存在 `failing-first-details.log`；实现后全部转绿（final-<sample>.log）。
3dresnet/dinov2/yoloe 为纯入口重构，无新模型接口，其既有 CLI/入口测试
（含 subprocess SDK-free 断言）作为行为等价回归，改写前后均通过。

## 逐样例状态

### samples/vision/3dresnet

* 入口：`runtime/python/main.py` 重写——显式构造 `VideoClassificationTask`
  并调用 `predict(clip)`；`_run` 移除。parser/list/dry-run/报告在新增
  `cli.py`（`build_parser`/`parse_args`/`report` 自 main 再导出）。包名以数字
  开头，importlib 为既定导入方式（cli.py 内部用相对导入）。
* 模型类：`classification.py VideoClassificationTask`（未改动，沿用上一批
  阶段名成果）；视频 clip 契约与逐调用 `VideoContext` 不变。
* 实际后端：`RuntimeModelRunner`（共享会话）或注入 runner；softmax/Top-K 在
  `samples._shared.classification`。
* 兼容接口：`pre_process`/`forward`/`post_process` 别名、`top_k`、
  `main.parse_args`、`main.build_parser`。
* 已运行测试：基线 20 OK → 终态 20 OK（含 CLI fixture pipeline JSON 输出、
  未知板卡先于 runner 拒绝）；`--help/--list-models/--dry-run --target s100`
  subprocess exit 0；contract checker 0 violations。
* 边界：输入为已准备 `.npy` clip，不做视频解码；板端 not-run。

### samples/vision/depth_anything_v2

* 入口：`main.py` + 新增 `cli.py`；main 显式构造 `DepthAnythingV2Task` 并调用
  `task.predict(image, return_details=True)` 一次，raw_depth.npy 取
  `details.raw`；目标路径校验/报告/五文件写入在 cli.py。
* 模型类：`DepthAnythingV2Task` + 新增 `DepthPredictionDetails`；逐像素 RGB
  z-score、INTER_NEAREST stretch/letterbox gray127、恢复几何不变。
* 实际后端：`RuntimeModelRunner`（S100 已发布制品）或注入 runner。
* 兼容接口：旧阶段名别名；`PreparedInput`/`DepthResult`/`ImageContext` 不变；
  `main.RuntimeModelRunner` patch 面保留。
* 已运行测试：基线 19 OK → 新增 3（先 TypeError 红）→ 终态 22 OK；CLI 测试
  增补 raw_depth.npy 断言；contract 0 violations。
* 边界：仅 S100 制品；相对深度非米制；板端 not-run。

### samples/vision/diffusiondrive

* 入口：`main.py` + 新增 `cli.py`；main 显式构造 `DiffusionDriveTask` 并调用
  `predict(features, return_details=True)` 一次，physical/raw/decoded 三份
  npz 全部来自该次调用；`started`/`finished` UTC 戳语义保持（推理前后）。
* 模型类：`DiffusionDriveTask` + 新增 `DiffusionDriveDetails`；四输入量化、
  四输出反量化、agent sigmoid（clip ±60）、BEV argmax、固定 caller 噪声不变。
* 实际后端：`RuntimeModelRunner`（S100P/S600）或注入 runner。
* 兼容接口：`main.add_runtime_arguments`（run_cases.py 构造 parser 用）、
  `main.sha256_file`、`main.RuntimeModelRunner`、旧阶段名别名、六数组结果键。
* 已运行测试：基线 27 OK → 新增 2（先红）→ 终态 29 OK；`run_cases.py --dry-run`
  exit 0；contract 0 violations。
* 边界：确定性示例特征，不做传感器输入/噪声再生；板端 not-run。

### samples/vision/dinov2

* 入口：`main.py` 重写——显式构造 `DINOv2Task`，主图与可选第二图各调用一次
  `predict`（两次独立请求，非重复推理链）；`_run`/`_summary`/`_cosine` 移至
  新增 `cli.py`（summary/cosine/save_feature/read_bgr_image）。
* 模型类：`embedding.py DINOv2Task`（未改动）。
* 实际后端：`RuntimeModelRunner`（S100/S100P/S600 各自制品）或注入 runner。
* 兼容接口：`main.build_parser`；输出 JSON 字段与 `Feature tensor saved:` 行
  不变。
* 已运行测试：基线 22 OK → 终态 22 OK（含 CLI 双图 cosine + output-file 保存、
  未知板卡先于 runner 拒绝）；SDK-free 三模式 exit 0；contract 0 violations。
* 边界：嵌入不做 softmax/L2/池化；板端 not-run。

### samples/vision/fcos（零改动复核）

* 入口：`main.py` 已显式构造 `FCOSTask` 并调用 `predict`，本轮未改。
* 模型类：`fcos.py FCOSTask`（上一批成果）；实际后端 `RuntimeModelRunner`。
* 兼容接口：旧阶段名别名、`--classes-num==80` 固定、自训练标签路径。
* 已运行测试：基线/终态均 OK（test_fcos_contract.py）；`--help/--list-models/
  --target x5 --dry-run` exit 0；contract 0 violations。
* 边界：X5 目标；板端 not-run。

### samples/vision/lanenet

* 入口：`main.py` + 新增 `cli.py`；main 显式构造 `LaneNetTask` 并调用
  `predict(image, return_details=True)` 一次，raw_outputs.npz 取
  `details.raw`（SDK 名→`output_N` 键映射在 cli.py，语义不变）。
* 模型类：`LaneNetTask` + 新增 `LanePredictionDetails`；RGB/ImageNet 归一化、
  0/1 二值标签、model-grid（256×512）语义不变。
* 实际后端：`RuntimeModelRunner`（S100）或注入 runner。
* 兼容接口：旧阶段名别名；`main.RuntimeModelRunner` patch 面。
* 已运行测试：基线 25 OK → 新增 2（先红）→ 终态 28 OK（含 raw_outputs.npz
  键值回归）；contract 0 violations。
* 边界：不做聚类、不恢复原图尺寸；板端 not-run。

### samples/vision/lprnet（零改动复核）

* 入口：`main.py` 已显式构造 `LPRNetTask` 并调用 `predict`，本轮未改。
* 模型类：`lprnet.py LPRNetTask`；实际后端 `RuntimeModelRunner`（X5）。
* 已运行测试：基线/终态 OK；SDK-free 三模式 exit 0；contract 0 violations。
* 边界：输入为预打包 float32 .dat；CTC 解码契约不变；板端 not-run。

### samples/vision/modnet（零改动复核）

* 入口：`main.py` 已显式构造 `MODNetTask` 并调用 `predict`，本轮未改。
* 模型类：`modnet.py MODNetTask`；实际后端 `RuntimeModelRunner`（X5）。
* 已运行测试：基线/终态 OK；SDK-free 三模式 exit 0；contract 0 violations。
* 边界：ref-size 固定 512；matting 输出契约不变；板端 not-run。

### samples/vision/pointnet

* 入口：`main.py` + 新增 `cli.py`；main 显式构造 `PointNetTask` 并调用
  `predict(points, return_details=True)` 一次，绘图用
  `details.prepared.tensors`（归一化点云），report 的 normalization 用
  `details.prepared.context`（centroid/radius）。
* 模型类：`PointNetTask` + 新增 `PointNetPredictionDetails`；质心/最大半径
  归一化、点序不变、整数 float64 解码不变。
* 实际后端：`RuntimeModelRunner`（S100）或注入 runner。
* 兼容接口：旧阶段名别名；`main.build_parser`。
* 已运行测试：基线 30 OK → 新增 3（先红）→ 终态 33 OK（含默认绘图路径的
  normalized cloud 断言、README 示例 exec）；SDK-free 三模式 exit 0；
  contract 0 violations。
* 边界：点数须与编译 metadata 一致；matplotlib 仅绘图用；板端 not-run。

### samples/vision/pp_liteseg（零改动复核）

* 入口：`main.py` 已内联构造 `PPLiteSegTask` 并调用 `predict`，本轮未改。
* 模型类：`pp_liteseg.py PPLiteSegTask`；实际后端 `RuntimeModelRunner`（X5）。
* 已运行测试：基线/终态 OK；SDK-free 三模式 exit 0；contract 0 violations。
* 边界：输入几何固定 1024×512；int32 类 ID 语义不变；板端 not-run。

### samples/vision/unet（零改动复核）

* 入口：`main.py` 已显式构造 `UNetTask` 并调用 `predict`（elapsed_ms 为整条
  predict 口径，历史语义保持），本轮未改。
* 模型类：`unet.py UNetTask`；实际后端 `RuntimeModelRunner`（X5）。
* 已运行测试：基线/终态 OK；SDK-free 三模式 exit 0；contract 0 violations。
* 边界：NV12 单通道输入；Pascal VOC 20+1 类；板端 not-run。

### samples/vision/unetmobilenet（零改动复核）

* 入口：`main.py` 已内联构造 `UnetMobileNetTask` 并调用 `predict`，本轮未改。
* 模型类：`unetmobilenet.py UnetMobileNetTask`；实际后端
  `RuntimeModelRunner`（S100/S600）。
* 已运行测试：基线/终态 OK；`--target s100 --dry-run` exit 0（x5 无制品，
  dry-run 正确 exit 2）；contract 0 violations。
* 边界：双输入（image NV12 + scale/offset）；原始分辨率类 ID；板端 not-run。

### samples/vision/yolo26_depth

* 入口：`main.py` + 新增 `cli.py`；main 显式构造 `Yolo26DepthTask` 并调用
  `predict(image, warmup=args.warmup, return_details=True)` 一次；report 的
  `warmup`/`latency_ms` 来自 details，`latency_scope` 文案不变。
* 模型类：`Yolo26DepthTask` + 新增 `DepthPredictionDetails`；NV12 letterbox
  (pad 114)/lite scale-fill、lite 校准（clip[-4,5]×a+b）、恢复几何不变。
* 实际后端：`RuntimeModelRunner`（X5/S100/S100P/S600，profile 按变体）或注入
  runner。
* 兼容接口：旧阶段名别名；CLI flags（`--warmup` 默认 3、`--priority`/`
  --bpu-cores` 的 X5/S 差异默认）与校验次序不变。
* 已运行测试：基线 42 OK → 新增 5（先红）→ 终态 47 OK（含 CLI warmup=2 →
  恰 3 次 runner 调用、raw_logit.npy、report 字段）；SDK-free 三模式 exit 0；
  contract 0 violations。
* 边界：相对深度非标定米制；计时为端侧单次前向（含传输校验/拷贝，非纯 BPU）；
  板端 not-run。

### samples/vision/yoloe（detection_tracking 移交）

* 入口：`main.py` 重写为薄入口——Config 构造/校验、`build_runner`、
  `model = YOLOE(selection, config, runner=runner)`、`model.predict(image)`、
  `save_result` + JSON 报告；parser/list/dry-run/调度校验/报告在新增
  `cli.py`。
* 模型类：`yoloe.py YOLOE`（未改动；11 的 DFL/NMS 与 26 的插值二值化协议、
  X5 full mask / S ROI mask 语义保持）。
* 实际后端：`build_runner`（X5 已发布浮点输出模型；S 需另行转换的浮点输出
  模型，SHA-256 校验）。
* 兼容接口：`main.build_parser`；`from ...yoloe import Config` 不变；全部
  CLI flags/默认值不变。
* 已运行测试：基线 48 OK → 终态 48 OK（含 dry-run config 矩阵、S11 morph
  默认、词表校验）；SDK-free 三模式 exit 0；contract 0 violations。
* 边界：S 目标的浮点输出模型须用户自行转换（publisher hash 政策不变）；
  板端 not-run。

## 汇总证据

* 基线：14 样例全部 exit 0（baseline-<sample>.log）。
* 终态：14 样例全部 exit 0（final-<sample>.log；测试计数 20/22/29/22/–fcos/28/
  –lprnet/–modnet/33/–pp_liteseg/–unet/–unetmobilenet/47/48，零改动样例以
  after-*.log 记录）。
* SDK-free：每样例 `--help`、`--list-models`、适用目标的 `--dry-run` 主机
  exit 0（sdkfree-<sample>.log；unetmobilenet 的 x5 dry-run 正确拒绝 exit 2，
  s100 exit 0）。
* contract checker：14 样例全部 0 violations（contract-<sample>.log）。
* 未运行（边界）：板端 SDK 链接/推理、真实权重下载、OE/Mapper/HMCT 编译、
  `samples/_shared/tests` 与全仓回归（其他执行器并行编辑中，本批不越界）。
  日志均以工作树当前源码运行；未提交任何代码。
