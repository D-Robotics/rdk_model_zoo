# ResNet Runtime 职责与接口调整

本次只调整 ResNet。规范见 [Python Runtime](../sample-standards/python-runtime.md)。

## 当前调用

```python
from samples.vision.resnet.runtime.python.classify import ResNetClassifier

model = ResNetClassifier("model.bin", target="x5", top_k=5)
result = model.predict("image.jpg")
```

自训练模型直接传 `input_size=(224, 224)`、`class_count=4`、`top_k=2` 和标签。
已经是概率的输出用 `score_policy="none"`，量化输出用 `output_transform="dequant"`。
库调用接收明确的 target；命令行仍支持 `--target auto`。

## 职责与导入变化

| 原接口或文件 | 当前方式 |
| --- | --- |
| `ResNetClassifier(selection, ...)`、上一版 `ResNetClassifier(target=...)` | `ResNetClassifier(model_path, target=..., ...)`；不再让模型类解析发布目录 |
| `model_binding.resolve_selection/list_available_assets` | cli.py 的模型选择工具；复用原有下载器的制品引用 |
| `custom_selection(...)` | 删除；构造函数直接接收输入尺寸、类别数和输出策略 |
| 本地 `model_runner` | 删除；分类器直接使用共享 runner |
| 本地 `classification/tensor_io/labels` 转发模块 | 删除；底层工具直接从 `samples._shared` 对应模块导入 |
| `legacy.ResNet/Resnet18` | 改用 `ResNetClassifier` 与 `ClassificationResult` |
| 本地 `cli` | 聚合参数、列表、dry-run、发布模型选择和展示，main.py 保持短入口 |

`pre_process/forward/post_process` 仍分别转调 `preprocess/infer/postprocess`。
`predict` 返回类型和数值处理保持一致。

公共图片读取移到 `_shared.image.read_bgr_image`；分类标签检查移到
`_shared.labels.validate_labels`；本地分类加载使用
`_shared.model_runner.RuntimeModelRunner.from_file`。不新增通用模块文件。

共享绑定保留已有概念的入口支持：`custom=True` 的本地模型可不传发布目录表，
仍执行完整元数据检查；发布模型的选择仍要求目录表。没有新共享文件或新注册框架。
模型内部 binding 记录的是本地文件契约，发布制品身份由入口 selection 保留并展示。

编译、下载命令不变。conversion README 的运行时接入示例及导出脚本的提示文字
更新为 ResNetClassifier；导出算法和参数行为不变。
历史验证脚本仍保留原始源码版本，复现时使用它们记录的提交。
本次本地模拟验证不代表已完成新版板端测试。

最新约定：取消两文件数量约束。ResNet 为 main/classify/cli；YOLO 按 detect、segment 等任务组织，公共能力复用 _shared。
