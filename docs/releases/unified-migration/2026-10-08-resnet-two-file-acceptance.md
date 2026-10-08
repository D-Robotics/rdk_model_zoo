# ResNet Runtime 职责精简验收单

日期：2026-10-08。状态：本地主机验证完成，待用户验收代码组织与使用体验。
当前采用按职责划分的规范，不设文件数量或行数指标。

## 打开位置

- 工作目录：`/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk-resnet-two-file-runtime`
- 分支：`codex/resnet-two-file-runtime-20261008`
- 基线：develop 的 `dc237a332232fe9e8a40913b71c989b4ee56ac64`
- 改动保留在独立工作树，未提交、未合入 develop、未推送。
- [编写规范](../../sample-standards/python-runtime.md)
- [main.py](../../../samples/vision/resnet/runtime/python/main.py)
- [classify.py](../../../samples/vision/resnet/runtime/python/classify.py)
- [cli.py](../../../samples/vision/resnet/runtime/python/cli.py)
- [中文使用说明](../../../samples/vision/resnet/runtime/python/README_cn.md)
- [接口调整说明](../../migration/2026-10-08-resnet-two-file-runtime.md)

## 职责验收

| 文件 | 内容 |
| --- | --- |
| main.py | 参数传递、构造模型、predict、调用展示函数 |
| classify.py | 模型初始化、前处理、一次推理调用、后处理、predict |
| cli.py | 参数定义、发布模型选择、列表/dry-run、结果展示 |
| _shared/image.py | read_bgr_image；既有 NV12 转换 |
| _shared/labels.py | load_labels、validate_labels |
| _shared/model_runner.py | from_file 构造本地分类 Runtime，加载及元数据校验 |
| _shared/runtime.py | 既有 SDK 会话与真实执行板卡身份校验 |

没有新增公共模块文件；图片读取、标签检查和
本地分类加载可用于其他模型。ResNet 模型类没有发布目录或单独的 custom_selection。
发布模型的特有配置留在本地 CLI。旧兼容类和纯转发文件不再保留。

建议重点看 main 的调用主线、classify 的三个阶段，以及公共函数的独立职责。
本轮只调整 ResNet；YOLO 后续按 detect、segment 等任务各自组织文件。

## 已执行验证

下表为职责重构阶段的验证记录，对应 `shared-boundaries/validation.json` 的源码哈希。
后续 Google docstring 修正的独立验证见下一节，不将旧哈希记录覆盖成新版本。

| 检查 | 结果 |
| --- | --- |
| ResNet unittest | 81 项通过 |
| 共享模块 unittest | 189 项通过；新增图片读取、标签校验和通用本地分类 Runtime 测试 |
| 共享 runner 调用方 | 另 27 个 Sample、919 项通过；总计 29 个套件、1,189 项通过 |
| Sample 静态契约 | 51 个 Sample、0 违规、85 项按既有策略跳过、0 新豁免 |
| 前后版本数值比较 | 14 组完全一致：7 个板卡/模型组合 × stretch/letterbox |
| 比较内容 | 入口发布配置、NV12 张量形状/类型/字节、原始输出、Top-K ID/分数/标签 |
| CLI | 帮助/列表/dry-run、模块入口、外部 cwd 的 run.sh 正常 |
| 依赖边界 | 帮助/列表/dry-run 阻断 NumPy、OpenCV、SDK 导入仍通过 |
| 入口模拟 | X5/S100/S600 调度、预测、打印、保存图片通过；错误板卡在 SDK 加载前拒绝 |
| 文档和编译 | 运行时 API 示例已同步；编译配方保留，导出脚本仅改接入提示文字 |
| 本版本真实板端执行 | not-run |

测试使用本地 fixture/mock，未执行真实 ONNX 导出、工具链编译、模型下载或 BPU 推理。

## 验证记录

本地目录：
`/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/local-execution/20261008-resnet-two-file-runtime/shared-boundaries`

包含 validation.json（源码 SHA-256）、suites.json、各套件日志、compare_runtime.py、
after.json。基线 before.json 位于其上一级目录。之前两版记录保留在各自目录。

## Google docstring 修正（2026-10-08）

- 依据 `docs/Model_Zoo_Repository_Guidelines.md` 的 Python 注释规范补全文件职责、
  类属性、函数 Args/Returns/Raises，以及模型各阶段的 shape/dtype/布局/值域。
- 范围：ResNet `main.py`、`classify.py`、`cli.py`；公共 `image.py`、`labels.py`、
  `model_runner.py` 的公开接口，以及 `cls_binding.bind_model`。不声明全仓注释已达标。
- 7 个 Python 文件去除 docstring 后 AST 与本轮修改前完全一致。
- Sphinx Napoleon GoogleDocstring 检查 32 个函数，全部实参和 Returns 均可解析。
- ResNet 主机测试重新执行：81 项通过。静态契约：51 Sample，0 违规，85 策略跳过。
- Sphinx 配置包含构造函数说明，并把 Attributes 渲染为字段，避免重复对象定义。
- 本轮接口独立 HTML 预览使用 `sphinx -W --keep-going` 构建通过，0 警告/错误。
- 全仓 API 文档尝试构建时产生 110 条警告、5 条 docutils 错误；错误来自
  ByteTrack `kalman_filter`、DINOv2 `conversion/mapper` 和 YOLOv5
  `evaluator/native/compare_native`。未修改这些模块，未替换完整发布文档包。

本轮证据目录：
`/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/local-execution/20261008-resnet-two-file-runtime/google-docstrings`

其中 `ast-audit.json` 记录前后源码哈希和 AST 比较，`docstring-audit.json` 记录函数检查，
`resnet-tests.log` / `sample-contract.log` 为主机检查，`sphinx-final.log` 保留全仓诊断；
`preview/html/index.html` 为接口预览，`preview.log` 为严格构建日志。

公共目录建议统一到根目录 `utils/`；本轮仍使用当前路径，尚未实施全仓搬迁。
参见[公共目录统一方案](../../superpowers/plans/2026-10-08-common-utilities-consolidation.md)。

## 本地复核

在该工作目录和包含 NumPy、OpenCV、PyYAML 的 Python 环境中运行：

```bash
python -m unittest discover -s samples/vision/resnet/tests
python -m unittest discover -s samples/_shared/tests
python tools/sample_contract/check.py --scope migration
python samples/vision/resnet/runtime/python/main.py --list-models
python samples/vision/resnet/runtime/python/main.py --dry-run --target x5
```
