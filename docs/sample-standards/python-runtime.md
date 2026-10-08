# Python Runtime 编写规范

目标：人和 Agent 都能顺着入口读懂模型流程，并复用稳定的公共能力。
详细要求见 [Sample Runtime 代码规范](runtime-code.md)。普通单任务 Sample 默认采用 `main.py`、`cli.py` 和模型类三个 Python 文件；复杂算法按完整职责拆分。ResNet 展示单模型分类，YOLO 按检测、分割、姿态等任务组织。

## 职责划分

| 位置 | 职责 |
| --- | --- |
| `main.py` | 获取参数、构造模型、调用 predict、把结果交给展示函数 |
| `classify.py` / `detect.py` / `segment.py` | 模型初始化与 preprocess、infer、postprocess、predict；保留模型特有算法和默认参数 |
| `cli.py` | 较多的参数声明、发布模型选择、列表/dry-run、结果展示和保存 |
| `utils/py_utils/` | SDK 调用、读取图片、标签校验、通用张量转换等多个模型可复用的能力 |

ResNet 的组织方式：

```text
utils/py_utils/
├── runtime.py        # SDK 会话与板卡识别
├── model_runner.py   # 分类模型加载及元数据绑定
├── image.py          # BGR 图片读取、通用像素转换
├── labels.py         # 标签读取与校验
├── tensor_io.py      # 分类输入准备
└── classification.py # 分类结果和 Top-K
samples/vision/resnet/runtime/python/
├── main.py
├── classify.py
└── cli.py
```

公共函数按职责归入根目录 `utils/py_utils/`，C++ 公共函数归入 `utils/c_utils/`。

YOLO 可按任务设置 detect.py、segment.py、pose.py、classify.py 等文件；
不同任务的后处理保持可读，不为凑文件数量合在一起，也不为每个函数另建文件。

## 模型接口

```python
from samples.vision.resnet.runtime.python.classify import ResNetClassifier

model = ResNetClassifier("model.bin", target="x5", top_k=5)
result = model.predict("image.jpg")
```

- 初始化调用公共 Runtime 加载模型和绑定张量；模型类不承担发布目录管理。
- preprocess 说明该模型如何准备输入；infer 调用一次 Runtime；postprocess 展示该模型如何处理输出。
- predict 在本文件内显式串联三阶段，不转交给隐藏的整套推理流程。
- 公共函数直接调用，模型类不复制图片读取、标签校验或 SDK 包装的实现。
- 图片路径与 BGR uint8 数组都可输入；结果返回调用者，预测本身不打印、保存或下载。
- 自训练模型直接传 input_size、class_count、score_policy 等参数，无需 custom_selection。
- CLI 按 target/variant 选择发布制品；模型类接收明确路径和目标。实际执行仍校验板卡身份。

## 抽取与新增文件的判断

已有相同公共能力时直接复用。多个模型都适用的函数进入职责匹配的公共模块；
模型专有解码保留在任务文件。同一目录新增文件应承担完整且可说明的职责，
不建立只转发几个 import 的空壳，不引入无需求的基类、注册器或插件框架。

ResNet 中，通用本地分类模型的契约建立随 Runtime 加载封装在共享 runner；
ResNet 制品目录和模型默认参数留在本地 CLI。公共模块不写入 ResNet 专用制品名。

## 注释要求

遵循[仓库规范的 Python 注释要求](../Model_Zoo_Repository_Guidelines.md#python-注释规范)，
统一使用 Google-style docstring，不以单行职责描述替代完整接口说明。

- 文件顶部放 module docstring，先写职责摘要；保留版权和许可证声明。
- 公开类写职责、构造参数说明或构造方法引用，以及读者关心的 `Attributes:`。
- 每个自编函数和方法写摘要、所有实参的 `Args:` 和 `Returns:`；无实参时不写空 Args，
  无返回值时明确 `None`。`self` / `cls` 不作为调用参数重复说明。
- 张量接口写明 shape、dtype、布局、值域；分类结果写明排序、标签和分数语义。
- `Raises:` 写实际可能发生且影响调用者的异常；`Notes:` 只补充必要的副作用或约束。
- 行内注释使用英文，解释关键逻辑；复杂流程需要步骤概览，简单转发不逐行解释。
- 注释内容与实现核对，通过 Sphinx Napoleon 解析并同步源码 API 文档。

## 保持的行为与验收

1. main 中能直接看到构造模型和 predict；模型文件能读到完整三阶段。
2. 图片读取、标签检查、SDK 调用有明确公共入口，独立测试可在其他模型中复用。
3. 保留现有 CLI、默认值、错误报告、NV12 排列、插值、输出变换和 Top-K 数值。
4. 模型帮助、列表和 dry-run 按声明的依赖执行，不加载板卡 SDK；轻量的参数查看和模型列举不连带加载图像库。
5. 导出/编译配方保持；运行时中英文 README 直接介绍使用方法，迁移和测试记录另存。
6. 主机注入 Runtime 与真实板卡验证分别记录，不混称。

文件数和行数只作为阅读参考，不作为验收目标。
