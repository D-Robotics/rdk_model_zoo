# 全部本仓 Sample 可读 Runtime — vision_tasks 批次报告

日期：2026-10-05
基线：`c1510ede652d30d83eeb691ff0e90b3f735a70a4`（分支 `codex/readable-model-examples-20261001`）
执行器：本地 Claude Code 2.1.276 + GLM `glm-5.3[1m]`；主机 Python 为 `rdk_model_zoo/.venv/bin/python`（3.14.7）。
设计依据：`docs/superpowers/specs/2026-10-05-all-sample-readable-runtime-design.md`；范例为 ResNet `classify.py` 与 Ultralytics `detect.py`。
本报告只覆盖本批次十三个样例；代码与测试未提交，由 Codex 按路径评估提交。外部证据日志目录：`local-execution/20261005-all-sample-readable-runtime/vision_tasks/`（baseline/failing/final 日志、contract-check.log、entries-sdk-free.log、pre-change-canonical-absent.log、progress.json）。开始时工作树内已有其他批次并行修改（ultralytics_yolo 等），本批次未触碰。

> **批次快照声明（2026-10-05 补记）**：本报告记录的是**阶段改名批次**时点的事实，
> 早于 Codex 入口评审驱动的入口修正。下文"main.py 层面的两种形态"中第二种
> （main 内手工规范名组合三阶段）只是本批次的过渡形态，**不再符合最终入口
> 架构**；后续 entrypoint_polish_vision 批次已把 14 个 main（本批 13 个 + yoloe）
> 全部改为显式构造模型并调用 `predict`（五个中间输出样例经 `return_details`），
> 源码层面的状态见该批报告与 coverage.json 快照。本报告所有测试计数与结论均为
> 批次时点快照，最终状态以 Codex 终审报告为准。

## 统一改动模式

每个任务类将既有 `pre_process` / `forward` / `post_process` 实现更名为规范名
`preprocess` / `infer` / `postprocess`（单一实现，主体搬运不加改），旧名保留为薄
兼容别名；`predict` 在类内可见地串联 `preprocess → infer → postprocess`（显式传递
`prepared.tensors` / `prepared.context`）。CLI 参数、默认值、返回码、构造参数、
结果结构、逐目标预处理/量化/输出契约与 source provenance 均未改动。所有 SDK 仍为
懒加载；`--help` 在主机实测 13 条命令全部 exit 0 且无 stderr（`entries-sdk-free.log`），
list/dry-run 主机可用性由各样例既有 CLI 测试覆盖（final 日志全绿）。

main.py 层面的两种形态（本批次时点的快照；第二种为过渡形态，已被后续
entrypoint_polish_vision 批次的显式 `predict` + `return_details` 取代，见顶部
批次快照声明。未把旧运行流程搬入 cli.py，本批未新增 cli.py——既有 main 已经是
薄入口结构）：

* **结果即终态的样例（8/13）**：main 显式构造 runner/binding/task 后直接调用
  `task.predict(...)`（3dresnet、dinov2、fcos、lprnet、pp_liteseg、unet、
  unetmobilenet 原样已是如此；modnet 本轮由手工三段调用改为 `predict`）。
* **需要中间张量证据/计时的样例（5/13）**：depth_anything_v2、diffusiondrive、
  lanenet、yolo26_depth、pointnet 的既有规范输出契约包含 raw/中间产物归档
  （raw_depth.npy、physical_inputs.npz + raw_outputs.npz、raw_outputs.npz、
  warmup + 单次计时 forward、report/绘图的 prepared context）。它们的 main 在
  本批次以规范名单次组合同一组公开阶段并保留中间张量，代码内注释指明
  `predict` 是等价单调用 API；每组新测试都断言 `predict` 与手工规范名组合
  逐张量一致、runner 恰好调用一次。未改变任何输出文件、字段或计时口径。

新增测试统一为每样例一个 `ReadableInterfaceTests` 类（fcos/pp_liteseg/unet/
unetmobilenet 为 4 项、其余为 4 项左右）：规范名存在且旧名行为等价、`predict`
经 spy 证明路由到 `preprocess/infer/postprocess` 且 runner 恰一次、旧名同样路由
到同一规范实现、连续不同输入下 `predict` 与手工组合一致（几何/状态逐调用独立）。
3dresnet 记录了完整红绿循环（`failing-3dresnet.log`：实现前 4 项
`AttributeError: no attribute 'preprocess'`）；其余样例以
`pre-change-canonical-absent.log` 记录 HEAD 版本任务类中规范名全部 ABSENT 作为
实现前缺接口证据（AST 级补充证据，不替代 3dresnet 的行为级红绿日志）。
depth_anything_v2、diffusiondrive、lanenet、pointnet、yolo26_depth 既有的
AST 方法集合断言更新为「规范名 + 兼容别名」集合（同一实现两个名字）。

## 逐样例状态

### samples/vision/3dresnet

* 入口：`runtime/python/main.py`（`_run` 内显式构造 `VideoClassificationTask`
  并调用 `predict(clip)`；包名以数字开头，importlib 为既定导入方式）。
* 模型类：`classification.py VideoClassificationTask`（别名 `ResNet3DTask`）—
  规范阶段名 + 兼容别名；视频 clip 契约（已归一化 `(1,3,16,112,112)`、不解码
  视频）与逐调用 `VideoContext` 不变。
* 实际后端：`RuntimeModelRunner`（共享会话）或注入 runner；softmax/Top-K 仍在
  `samples._shared.classification`。
* 兼容接口：`pre_process`/`forward`/`post_process`、`__call__`、`top_k` 参数。
* 测试：基线 16 OK → 新增 4 项（先失败，见 failing-3dresnet.log）→ 终态 20 OK；
  contract 0 violations。
* 边界：视频解码/抽帧不在运行时范围内（输入为准备好的 .npy clip）；板端 not-run。

### samples/vision/depth_anything_v2

* 入口：`runtime/python/main.py`（显式构造 task；因 raw_depth.npy 规范输出以规范
  名单次组合三阶段，`predict` 为等价单调用 API）。
* 模型类：`depth_anything_v2.py DepthAnythingV2Task` — 逐像素 RGB z-score（非
  ImageNet 常数）、INTER_NEAREST stretch / letterbox gray127、恢复几何全部不变。
* 实际后端：`RuntimeModelRunner`（S100 已发布制品）或注入 runner。
* 兼容接口：旧名别名；`PreparedInput`/`DepthResult`/`ImageContext` 结构不变。
* 测试：基线 15 OK → 新增 4 项 → 终态 19 OK；AST 方法集合断言更新；contract 0
  violations。
* 边界：S100 以外目标按 manifest 拒绝；相对深度非米制；板端 not-run。

### samples/vision/diffusiondrive

* 入口：`runtime/python/main.py`（显式构造 task；physical_inputs.npz 与
  raw_outputs.npz 证据需要中间张量，规范名单次组合三阶段）。
* 模型类：`diffusiondrive.py DiffusionDriveTask` — 四输入量化（camera/lidar/
  status/noise）与四输出反量化、agent sigmoid（clip ±60）、BEV argmax、固定
  caller 噪声语义不变。
* 实际后端：`RuntimeModelRunner`（S100P/S600）或注入 runner。
* 兼容接口：旧名别名；六数组结果结构与键不变。
* 测试：基线 23 OK → 新增 4 项 → 终态 27 OK；AST 方法集合断言更新；contract 0
  violations。
* 边界：无驱动/执行控制声明；噪声由调用方固定提供；板端 not-run。

### samples/vision/dinov2

* 入口：`runtime/python/main.py`（`_run` 显式构造 `DINOv2Task`，逐图调用
  `predict`，第二图余弦相似度为展示层逻辑）。
* 模型类：`embedding.py DINOv2Task`（别名 `Dinov2Task`）— bicubic 短边 256 /
  中心 crop 224 / ImageNet 归一化、双输出（cls/patch）选择与量化变换不变。
* 实际后端：`RuntimeModelRunner`（S100/S100P/S600 各自 manifest HBM）或注入
  runner。
* 兼容接口：旧名别名；`image_format` 参数保留。
* 测试：基线 18 OK → 新增 4 项 → 终态 22 OK；contract 0 violations。
* 边界：不做 softmax/L2；板端 not-run。

### samples/vision/fcos

* 入口：`runtime/python/main.py`（显式构造 `FCOSTask` 并调用 `predict`）。
* 模型类：`fcos.py FCOSTask` — packed NV12、十五输出按名集合匹配、
  `sqrt(sigmoid(cls)*sigmoid(center))` 置信、stride 缩放、源 NMS、direct-resize
  独立宽高比 / letterbox 冻结 padding 反算均不变。
* 实际后端：`RuntimeModelRunner`（X5 三个 EfficientNet 变体）或注入 runner。
* 兼容接口：旧名别名；`DetectionResult.as_tuple`、conf/iou 覆盖参数不变。
* 测试：基线 38 OK → 新增 4 项 → 终态 42 OK；contract 0 violations。
* 边界：80 类契约不可放宽；板端 not-run。

### samples/vision/lanenet

* 入口：`runtime/python/main.py`（显式构造 task；raw_outputs.npz 证据需要原始
  输出映射，规范名单次组合三阶段）。
* 模型类：`lanenet.py LaneNetTask` — RGB/ImageNet 预处理、模型网格 256×512、
  二值标签必须已是 0/1、无聚类声明等语义不变。
* 实际后端：`RuntimeModelRunner`（S100）或注入 runner。
* 兼容接口：旧名别名；`LaneResult`（embedding/binary）不变。
* 测试：基线 22 OK → 新增 4 项 → 终态 26 OK（含 cpp 契约 fixture 构建测试）；
  AST 方法集合断言更新；contract 0 violations。
* 边界：不做车道 ID 聚类/原图尺寸恢复；C++ runtime 与转换配方未动；板端 not-run。

### samples/vision/lprnet

* 入口：`runtime/python/main.py`（显式构造 `LPRNetTask` 并调用 `predict(test_bin)`）。
* 模型类：`lprnet.py LPRNetTask`（别名 `LPRNet`）— 预打包 float32 .dat 输入、
  单元素轴裁剪到 `(68,18)`、源 CTC 去重去 blank 解码不变。
* 实际后端：`RuntimeModelRunner`（X5 lpr.bin）或注入 runner。
* 兼容接口：旧名别名；`ctc_logits`/`decode_plate` 模块级函数不变。
* 测试：基线 23 OK → 新增 4 项 → 终态 27 OK；contract 0 violations。
* 边界：无图像预处理（输入即网络张量）；板端 not-run。

### samples/vision/modnet

* 入口：`runtime/python/main.py`（本轮把手 `pre_process/forward/post_process`
  三段调用改为 `matte = task.predict(image)`；CLI 参数/默认值/返回码不变）。
* 模型类：`modnet.py MODNetTask`（别名 `MODNet`）— BGR→RGB、`[-1,1]` 归一化、
  长边 512 resize + 居中零 padding、去 padding 恢复原图几何不变。
* 实际后端：`RuntimeModelRunner`（X5 手动获取制品）或注入 runner。
* 兼容接口：旧名别名；`GeometryContext`/`resize_with_padding` 不变。
* 测试：基线 13 OK → 新增 4 项 → 终态 17 OK；contract 0 violations。
* 边界：手工资产无公开 URL/hash；合成展示为应用层；板端 not-run。

### samples/vision/pointnet

* 入口：`runtime/python/main.py`（report/原始视图绘图需要本调用 prepared
  context，规范名单次组合三阶段；`predict` 为经测试等价的单调用 API）。
* 模型类：`pointnet.py PointNetTask` — 点云质心/最大半径归一化（不重采样不
  重排）、整数输出 float64 SCALE 反量化后 argmax、平局取最小 ID 不变。
* 实际后端：`RuntimeModelRunner`（S100）或注入 runner。
* 兼容接口：旧名别名；`PointContext` 不变。
* 测试：基线 26 OK → 新增 4 项 → 终态 30 OK；AST 方法集合断言更新；contract 0
  violations。
* 边界：点数必须等于编译元数据；matplotlib 绘图为应用层懒加载；板端 not-run。

### samples/vision/pp_liteseg

* 入口：`runtime/python/main.py`（显式构造 `PPLiteSegTask` 并调用 `predict`）。
* 模型类：`pp_liteseg.py PPLiteSegTask` — INTER_LINEAR 拉伸 1024×512 packed
  NV12、部署边界已是类别图（无 argmax/反量化/恢复原图）不变。
* 实际后端：`RuntimeModelRunner`（X5）或注入 runner。
* 兼容接口：旧名别名；`ImageContext` 不变。
* 测试：基线 18 OK → 新增 4 项 → 终态 22 OK；contract 0 violations。
* 边界：S 系无资产不回退 X5；类别 0..18 校验不变；板端 not-run。

### samples/vision/unet

* 入口：`runtime/python/main.py`（显式构造 `UNetTask` 并调用 `predict`）。
* 模型类：`unet.py UNetTask` — packed NV12 `(1,768,512,1)`、NCHW/NHWC 双布局、
  整数 SCALE 反量化、float32 不重复反量化、模型分辨率 21 类 mask 不变。
* 实际后端：`RuntimeModelRunner`（X5 五个 ResNet 变体）或注入 runner。
* 兼容接口：旧名别名；`ImageContext` 不变。
* 测试：基线 20 OK → 新增 4 项 → 终态 24 OK；contract 0 violations。
* 边界：VOC 21 类、512×512 几何固定；板端 not-run。

### samples/vision/unetmobilenet

* 入口：`runtime/python/main.py`（显式构造 `UnetMobileNetTask` 并调用 `predict`）。
* 模型类：`unetmobilenet.py UnetMobileNetTask` — split NV12（Y/UV 双输入）、
  INTER_AREA 拉伸 2048×1024、float64 反量化保序、INTER_NEAREST 恢复原图分辨率
  不变。
* 实际后端：`RuntimeModelRunner`（S100/S600）或注入 runner。
* 兼容接口：旧名别名；`ImageContext` 不变。
* 测试：基线 19 OK → 新态 23 OK（新增 4 项）；contract 0 violations。
* 边界：Cityscapes 19 类；C++ runtime 与 launcher 未动；板端 not-run。

### samples/vision/yolo26_depth

* 入口：`runtime/python/main.py`（warmup + 单次计时 forward 要求围绕计时的显式
  阶段调用，规范名组合三阶段；`predict` 为无计时的等价单调用 API）。
* 模型类：`yolo26_depth.py Yolo26DepthTask` — NV12 profile（letterbox 114）与
  lite profile（RGB /255）双路线、lite clip [-4,5] + 标定、exp 恢复与 192 方形
  输出契约不变。
* 实际后端：`RuntimeModelRunner`（X5/S100/S100P/S600 二十个资产变体、
  `--converted-model` 自转换边界）或注入 runner。
* 兼容接口：旧名别名；`DepthResult`（log_depth/depth_native/raw_logit/context）
  不变。
* 测试：基线 38 OK → 新增 4 项 → 终态 42 OK（含 cpp 契约 fixture 构建测试）；
  AST 方法集合断言更新；contract 0 violations。
* 边界：相对深度非标定米制；计时口径为单次 forward 含传输校验；板端 not-run。

## 汇总

| 样例 | 基线 | 终态 | 新增测试 | contract | SDK-free 入口 |
| --- | --- | --- | --- | --- | --- |
| 3dresnet | 16 OK | 20 OK | ReadableInterfaceTests×4（红绿循环已录） | 0 violations | --help exit 0 无 stderr |
| depth_anything_v2 | 15 OK | 19 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| diffusiondrive | 23 OK | 27 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| dinov2 | 18 OK | 22 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| fcos | 38 OK | 42 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| lanenet | 22 OK | 26 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| lprnet | 23 OK | 27 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| modnet | 13 OK | 17 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| pointnet | 26 OK | 30 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| pp_liteseg | 18 OK | 22 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| unet | 20 OK | 24 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| unetmobilenet | 19 OK | 23 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |
| yolo26_depth | 38 OK | 42 OK | ReadableInterfaceTests×4 | 0 violations | 同上 |

每个样例以独立进程 discover（`final-<sample>.log`），未与其他样例混跑；未运行
其他批次或全仓回归（其他执行器正在并行修改）。批次内修改文件共 58 个，全部位于
上述 13 个样例目录（`git status` 核对，无共享模块/其他批次文件改动；本报告为本
批次唯一的 docs 写入）。

## 未执行（not-run）

* 板端 SDK 链接与推理、真实权重下载、真实 ONNX 导出、OE/Mapper/HMCT 编译、
  量化精度与数据集评测；主机注入 runner 的通过不构成以上证据。
* lanenet / unetmobilenet / yolo26_depth 的 C++ runtime 与 launcher 只做了既有
  主机测试回归（含 fake-SDK fixture 编译），行为未改动。
* Catalog build/check、全仓 shared 测试与干净 checkout 入口复现属于最终集成
  阶段，不在本批次范围。
