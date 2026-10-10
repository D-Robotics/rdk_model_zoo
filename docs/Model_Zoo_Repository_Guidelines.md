# Model Zoo 仓库规范

本文档定义 Model Zoo 仓库的目录组织、命名约定与通用编码规则。具体规范链接各专门文档，不在本文重复：

- [Runtime 代码规范](sample-standards/runtime-code.md)：main/cli/模型文件结构、阶段契约、公共 Runtime、测试要求
- [README 契约](sample-standards/readme-contract.md)：各级 README 必答内容、双语配对、模板
- [模型示例架构](architecture/model-examples.md)：可读模型示例的当前形态

适用对象：仓库开发者、维护者与文档贡献者。

## 1. 仓库分层

`develop` 是统一源：X5 与 S（S100/S100P/S600）的 sample 统一存放于 `samples/`，
硬件通过每个 sample 的目标参数（`--target` 或 `--platform`）选择。平台身份只认板卡事实
（`/sys/class/boardinfo/soc_name` → socinfo → device-tree），未知即报错、不静默回退。

## 2. 目录组织

新增文件按功能属性放置到对应层级，不允许随意新增顶层目录或跨职责混放：

```text
.
├── CHANGELOG.md                           # 交付版本变化记录
├── LICENSE                                # Apache-2.0
├── README.md / README_cn.md               # 顶层项目说明（双语）
├── datasets/                              # 示例数据 + 数据集下载脚本
├── docs/
│   ├── release/                           # 发布数据：platforms.json（SoC 身份）+ {x5,s}/ 清单
│   ├── sample-standards/                  # README 契约、Runtime 代码规范与模板
│   ├── source_reference/                  # 源码参考文档构建（Doxygen+Sphinx）
│   ├── Model_Zoo_Repository_Guidelines.md # 本规范
│   ├── Python_API_User_Guide.md           # hbm_runtime Python 接口指引
│   └── UCP_User_Guide.md                  # libdnn/libucp 接口指引
├── samples/                               # 统一 sample
│   └── {vision,speech,robotics,llm,vla}/  # 每样本见下方 Sample 布局
├── utils/
│   ├── py_utils/                          # Python 公共工具
│   ├── c_utils/                           # C/C++ 公共工具
│   └── tools/                             # 编译、评测、目录发布与仓库检查工具
└── skills/                                # Agent Skills（独立版本线）
```

每个 sample 目录：

```text
samples/<领域>/<模型>/
├── model/                  # 编译后模型的获取、制品和板卡对应关系
├── conversion/             # 原始权重、导出和编译配方
├── runtime/{python,cpp}/   # 运行时实现
├── evaluator/              # 数据集评估及指标
├── test_data/              # 可直接运行的输入示例
└── tests/                  # Sample 的行为与数值回归
```

## 3. 命名规范

- **Sample 目录**：小写字母与数字，与上游模型名对应（`resnet`、`yolov5`、
  `efficient_sam`）。多词用下划线（`paddle_ocr`、`yolo26_depth`）。
- **Python 源文件**：小写蛇形命名。入口 `main.py`；CLI 辅助 `cli.py`；模型文件按任务
  命名（`classify.py`、`detect.py`、`segment.py`、`policy.py` 等）。
- **C/C++ 源文件**：与现有平台指南一致；头文件 `.h`/`.hpp`，源文件 `.c`/`.cc`/`.cpp`。
- **测试**：`tests/test_<行为>.py`；数值回归命名描述所锁定的行为。
- **模型二进制**（`.bin`/`.hbm`/`.onnx`）不入库；`model/` 提供下载与校验。

## 4. 编码通用规则

- Python 遵循 PEP 8；C/C++ 遵循对应平台指南。Google-style docstring 说明职责、
  参数、返回值与异常（详见 [Runtime 代码规范 §8](sample-standards/runtime-code.md)）。
- 行内注释用英文，解释关键算法或平台约束。
- 板卡选择显式传递；`utils/py_utils/platforms.py` 负责身份识别，未知目标报错。
- 公共能力只维护一份：Python 在 `utils/py_utils/`，C/C++ 在 `utils/c_utils/`。
  公共库不依赖某个 Sample 的入口。
- 模型二进制、数据集产物不入库；下载通过 `model/` 中的脚本完成并带哈希校验。

## 5. 跨平台注意

- 涉及板卡的代码用平台抽象（`--target`/`--platform`、`platforms.py`），不硬编码 SoC 名。
- 模型制品的目录与文件名以 manifest（`docs/release/{x5,s}/models.yaml`）声明的
  路径为准，不自行约定制品目录。

## 6. 验收要点

- 目录、命名与本规范一致；新 Sample 按 [README 契约](sample-standards/readme-contract.md)
  编写双语文档并通过 `utils/tools/sample_contract/check.py`。
- 新增或修改的代码通过对应 Sample `tests/` 中的主机测试。
- 保留有意义的数值与行为断言，不通过删除测试消除回归。
