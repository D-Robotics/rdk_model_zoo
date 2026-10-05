# 全部本仓 Sample 可读 Runtime — detection_tracking 批次报告

日期：2026-10-05
基线：`c1510ede652d30d83eeb691ff0e90b3f735a70a4`（分支 `codex/readable-model-examples-20261001`）
执行器：本地 Claude Code 2.1.276 + GLM `glm-5.3[1m]`；主机 Python 为 `rdk_model_zoo/.venv/bin/python`。
设计依据：`docs/superpowers/specs/2026-10-05-all-sample-readable-runtime-design.md`；范例为 ResNet `classify.py` 与 Ultralytics `detect.py`。
本报告只覆盖本批次五个样例；代码与测试未提交，由 Codex 按路径评估提交。外部证据日志目录：`local-execution/20261005-all-sample-readable-runtime/detection_tracking/`。

> **批次快照声明**：本报告是本批次的执行快照（yoloe 的入口其后由
> entrypoint_polish_vision 批次继续重写）；测试计数与结论均为批次时点记录，
> **最终状态以 Codex 终审报告为准**，本文不构成行为验收结论。

## 统一改动模式

每个任务类将既有 `pre_process` / `forward` / `post_process` 实现更名为规范名
`preprocess` / `infer` / `postprocess`（单一实现），旧名保留为薄兼容别名；
`predict` 在类内可见地串联 `preprocess → infer → postprocess`（显式传递
`prepared.transform` / `prepared.context`），不再经由共享 composer。CLI 参数、
默认值、返回码、构造参数、结果结构、协议数学与 source provenance 均未改动。
所有 SDK 仍为懒加载，`--help` / `--list-models` / `--dry-run` 保持主机可用
（实测 11 条命令全部 exit 0，见 `entries-sdk-free.log`）。

## 逐样例状态

### samples/vision/ultralytics_yolo

* 入口：`runtime/python/main.py`（薄入口，`yolo_cli.py` 承载参数/清单/dry-run/渲染；
  `yolo_dispatch.create_runtime_model` 构造任务模型后 inline `predict`）。
* 模型类：`detect.py YoloDetect`（范例，本轮复核未改）；本轮补齐
  `yolo_cls.YoloCls`、`yolo_seg.YoloSeg`、`yolo_pose.YoloPose`、
  `yolo26_det.YOLO26Detect`、`yolo26_obb.YOLO26OBB` 的规范阶段名 + 兼容别名；
  `YoloV10Detect` / `YOLO26Seg` / `YOLO26Pose` 经继承自动获得新接口。
  `YOLO26OBB` 保留对 `YoloDetect.preprocess/infer` 的显式共享（图像传输与原始
  runner 调用确为协议无关），旋转框解码留在本类 `postprocess`，`predict` 在本类
  内可见编排；未做跨协议同一假解码。`detection_io._predict_task` 已无调用方并移除。
* 实际后端：X5/S 分派的 `ModelRunner`（共享 SDK 会话）或注入 runner；DFL/LTRB/
  NMS-free 协议和解码器未改。
* 兼容接口：`pre_process` / `forward` / `post_process`、`pre_process_with_transform`、
  `yolo_detect` 等旧导入面、元组结果形状全部保留（既有 168 项测试全绿）。
* 测试：基线 168 OK → 新增 `tests/test_task_readable_stages.py`（先对 HEAD 失败：
  五个非检测任务 `AttributeError: no attribute 'preprocess'/'postprocess'`）→ 终态 173 OK；
  contract checker 0 violations。
* 边界：板端/SDK 推理、真实权重下载 not-run；主机 fixture 不替代板测。

### samples/vision/yoloe

* 入口：`runtime/python/main.py`（显式构造 `YOLOE(selection, config, runner=...)`
  并调用 `predict`）。
* 模型类：`yoloe.py YOLOE` — `preprocess`/`infer`/`postprocess` 为主实现，
  旧名兼容别名；`predict` 串联新方法并显式携带 `prepared.context`
  （YOLOE-11 ImageTransform / YOLOE-26 PFGeometry）。
* 实际后端：`build_runner(selection)`（共享 runner，浮点输出契约校验）或注入 runner。
* 兼容接口：`pre_process` / `forward` / `post_process` 别名；`Config` 导入面不变。
* 测试：基线 44 OK → 新增 `tests/test_readable_stages.py`（先对 HEAD 失败）→
  终态 48 OK（含 test_documentation 对 README 示例的真实执行）；contract 0 violations。
* 边界：YOLOE-26 固定 letterbox/无 NMS、S 量化路线拒绝、4585 词表校验等语义未改；
  板端 not-run。

### samples/vision/yolov5

* 入口：`runtime/python/main.py`（显式 `YOLOv5Task(runner, binding, ...)` + `predict`）。
* 模型类：`detection.py YOLOv5Task` — 阶段更名 + 兼容别名；`predict` 串联新方法，
  逐调用 context 与显式 0 阈值语义保留。
* 实际后端：`RuntimeModelRunner`（X5 packed NV12 640 / S split NV12 672）或注入 callable。
* 兼容接口：旧名别名；X5 OpenCV `NMSBoxes` 历史 quirk 与 S 按类 NMS 保留。
* 测试：基线 79 OK → 新增 `tests/test_readable_stages.py`（先对 HEAD 失败）→
  终态 83 OK（含 fixed-source 数值回归）；contract 0 violations。
* 边界：板端 not-run。

### samples/vision/yoloworld

* 入口：`runtime/python/main.py`（显式 `YOLOWorldTask(runner, binding, vocabulary,
  ...)` + `predict(image, prompts)`）。
* 模型类：`yoloworld.py YOLOWorldTask` — 阶段更名 + 兼容别名；文本 prompt 协议
  （≤32 prompts、离线词表只读快照、槽位填充）、RGB F32 输入与懒加载依赖未改。
* 实际后端：`RuntimeModelRunner`（X5 RGB+text 双输入）或注入 runner。
* 兼容接口：旧名别名；`parse_prompts`、词表 JSON 结构、结果结构不变。
* 测试：基线 19 OK → 新增 `tests/test_readable_stages.py`（先对 HEAD 失败）→
  终态 23 OK（含 evaluator 对照与 fixed-source 协议测试）；contract 0 violations。
* 边界：板端 not-run；X5 以外目标仍按 manifest 拒绝。

### samples/vision/bytetrack

* 入口：`runtime/python/main.py`（显式 `ByteTrackTask(YOLOv5Task(...), config=cfg)`，
  逐帧 `predict`，视频/记录 IO 在 CLI）。
* 模型类：`tracking.py ByteTrackTask` — `preprocess`/`infer`/`postprocess` 为主实现
  （内部经 detector 的既有公开阶段名委托），旧名兼容别名；`predict` 串联新方法。
  每帧恰好一次 `tracker.update`（含空检测），`reset` 重建流但进程级 ID 计数不清零，
  后端异常后要求先 reset 的语义不变。
* 实际后端：S100P 消费的 YOLOv5 检测器（`consumer='bytetrack'` 选择）+ CPU
  BYTETracker（lap/cython-bbox 懒加载）。
* 兼容接口：旧名别名；`TrackingConfig`、`Track`、JSONL 记录格式不变。
* 测试：基线 12 OK → 新增 `tests/test_readable_stages.py`（先对 HEAD 失败）→
  终态 16 OK（含真实 CPU tracker 与 fixed-source 序列回归）；contract 0 violations。
* 边界：板端与真实视频推理 not-run。

## 汇总

| 样例 | 基线 | 终态 | 新测试（先失败） | contract | SDK-free 入口 |
| --- | --- | --- | --- | --- | --- |
| ultralytics_yolo | 168 OK | 173 OK | test_task_readable_stages.py | 0 violations | help/list/dry-run exit 0 |
| yoloe | 44 OK | 48 OK | test_readable_stages.py | 0 violations | list/dry-run exit 0 |
| yolov5 | 79 OK | 83 OK | test_readable_stages.py | 0 violations | list/dry-run exit 0 |
| yoloworld | 19 OK | 23 OK | test_readable_stages.py | 0 violations | list/dry-run exit 0 |
| bytetrack | 12 OK | 16 OK | test_readable_stages.py | 0 violations | list/dry-run exit 0 |

每个样例按批次要求以独立进程 discover（`final-<sample>.log`），未与其他样例混跑；
未运行其他批次或全仓回归（其他执行器正在并行修改）。

## 未执行（not-run）

* 板端 SDK 链接与推理、真实权重下载、真实 ONNX 导出、OE/Mapper/HMCT 编译、
  量化精度与数据集评测；主机注入 runner 的通过不构成以上证据。
* C++ runtime、evaluator 板端脚本只做了既有主机测试回归，行为未改动。
* Catalog build/check、全仓 shared 测试与干净 checkout 入口复现属于最终集成阶段，
  不在本批次范围。
