# 可读模型示例架构

当前代码采用 [Sample Runtime 代码规范](../sample-standards/runtime-code.md)。
本文回答三个问题：示例代码各层放什么、调用链长什么样、自训练模型从哪里接入。

## 1. 文件职责

普通单任务 Sample 默认三个 Python 文件；多任务家族按任务划分模型文件
（Ultralytics YOLO 覆盖 detect / segment / pose / classify / obb 五类任务，
目标形态约 7–8 个 Python 文件，详见规范 §1）。

- **main.py（薄入口）**：解析参数 → 显式构造模型对象 → 调用 `predict` → 交付结果。
  入口可直接定位模型构造与 `predict`，不把构造藏进通用 dispatch。
- **cli.py**：参数声明、发布模型选择、`--list-models`/`--dry-run`、结果展示与保存。
  同一选项只声明一次；帮助与列表不触发图像库或板端 SDK 加载。
- **模型文件（按任务命名）**：具名模型类，初始化与
  `preprocess` / `infer` / `postprocess` / `predict` 主线在一个文件内可读。
  复杂算法（NV12 打包、DFL/LTRB 解码、NMS、量化变换）继续留在共享模块，
  不为了"文件完整"复制。

## 2. 调用链

以 ResNet 为例，调用链为：`main.py` 解析参数 → `cli.resolve_selection`
按目标选定制品 → 以 `selection.model_path`、`target` 及契约参数
（输入尺寸、插值、分数策略等）构造 `ResNetClassifier` → `predict` →
CLI 输出与保存。真实入口见
[main.py](../../samples/vision/resnet/runtime/python/main.py)，使用方式见
[ResNet README](../../samples/vision/resnet/README.md)。

- `predict` 显式串联三阶段；既有 `pre_process` / `forward` / `post_process`
  名称是同一实现的薄别名，不维护第二份实现。
- Context（尺寸、scale、padding 等几何信息）显式存放在 `prepared` 返回值中，
  不放入会被下一次调用覆盖的实例字段。
- 多阶段任务（OCR/SAM/Paraformer）：每个 stage 公开三步接口，
  `pipeline.predict` 显式编排；阶段次序在代码中直接可读。

## 3. 公共 Runtime

`utils/py_utils/runtime.py` 只承接 SDK 导入、模型实例创建与按目标身份检查。
集成的 runner 复用会话的加载/身份边界后，直接在同一个已加载 SDK 对象上执行
自身已校验的调用；运行时元信息读取与调度参数（`set_scheduling_params` 及其
校验）保留在各 runner。公共库不隐藏模型前后处理，也不依赖某个 Sample 的入口。

## 4. 板卡与模型选择

- 各 Sample 通过 `--target` 或 `--platform` 显式选择目标板卡，支持的目标
  组合以该 Sample 的 `--help` 与 README 声明为准；提供 `auto` 的 Sample
  读取板卡身份。各 Sample 的 README 与支持矩阵说明其发布
  制品实际支持的板卡。
- 模型准备通过 `model/download_model.sh`（或各 sample 的下载脚本）完成；
  manifest（`docs/release/{x5,s}/models.yaml`）是制品与 URL 的事实源。
- 自定义模型路径通过 CLI 显式传入，经实际张量契约校验。

## 5. 边界

- 薄 SDK 会话不是自研推理引擎；真实执行依赖板端安装的 `hbm_runtime`。
- ONNX 导出、工具链量化编译与板端推理是独立环节，其执行状态在各 Sample
  文档中单独记录；主机测试不替代板端证据。
- 仅 C++ 运行时的 Sample（如 `samples/llm/*`）以其原生 generate/stream/reset
  接口为明确等价形态，不伪造 Python Runtime。
