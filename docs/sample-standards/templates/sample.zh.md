<!-- 模板：sample 根 README（中文）。契约：docs/sample-standards/readme-contract.md §4.1。
     规则：所有 <a id="…"></a> 锚点保持原样；替换 ⟪…⟫ 占位符；完成后删除引导块引用；
     章节只能按契约 §1 的 not-applicable 理由删除。必须与英文 README.md 成对。 -->

# ⟪模型名称⟫（⟪任务⟫）

<a id="overview"></a>
## 算法与来源

> **必须回答：** 模型做什么（一句话）；算法简述；官方论文/仓库链接；在本仓的定位。

⟪一句话任务描述，如“面向 RDK 板卡的实时目标检测”。⟫

- 算法：⟪一段话，不展开实现细节⟫
- 官方来源：⟪论文/仓库链接⟫
- 本仓分类：`samples/⟪domain⟫/⟪name⟫`

<a id="support-matrix"></a>
## 支持与实测矩阵

> **必须回答：** 哪些 target × variant × 语言**支持**，其中哪些**实际上板验证过**。
> 单元格只有三态：`supported-verified` / `supported-not-run` / `not-supported`。
> 缺 C++ 必须在此可见——未提供 cpp 时任何位置不得声称双语言支持。

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| ⟪variant⟫ | ⟪状态⟫ | ⟪状态⟫ | ⟪状态⟫ | ⟪状态⟫ | ⟪有/无⟫ | ⟪有/无⟫ |

板端验证证据：⟪链接 evidence/批次评审，或写 “not-run”⟫。

<a id="prerequisites"></a>
## 环境前提

> **必须回答：** 板卡与系统镜像版本；工具链要求；Python 依赖；磁盘/内存约束。
> 版本号必须具体（“最新版”不是版本）。

- 板卡：⟪如 RDK X5（8GB/4GB），系统镜像 ≥ ⟪版本⟫⟫
- Python：⟪版本⟫，依赖⟪清单或“板端镜像自带”⟫
- 需预先准备模型制品（见[快速体验](#quickstart)）。

<a id="quickstart"></a>
## 快速体验

> **必须回答：** **一条**从模型准备到看到结果的完整路径。每条命令给出 cwd、前置文件
> 来源、参数、输出与成功判断。模型准备必须显式（`model/download.sh --target …`），
> 不依赖隐式自动下载。

```bash
# cwd：仓库根目录
bash samples/⟪domain⟫/⟪name⟫/model/download.sh --target ⟪target⟫
# 预期：制品位于 ⟪path⟫（下载日志打印 sha256 校验结果）

# cwd：仓库根目录
python3 samples/⟪domain⟫/⟪name⟫/runtime/python/main.py --target ⟪target⟫ ⟪input⟫
# 预期：⟪可观察的成功判据，如打印 Top-5 / 生成结果文件 ⟪path⟫⟫
```

⟪若存在 run.sh 快捷脚本，在显式路径之后并列给出。⟫

<a id="expected-results"></a>
## 预期结果

> **必须回答：** 正确运行应看到什么——test_data 的真实输出形态/数值、输出文件路径与
> 命名。禁止虚构精度数字。

⟪如 test_data/⟪image⟫ 的 Top-5 = […]；结果图写入 ⟪path⟫⟫

<a id="directory"></a>
## 目录职责

> **必须回答：** 每项一句话职责；与实际目录一致（本地路径会被机器校验）。

```text
⟪name⟫/
├── conversion/    # 模型转换（ONNX → BPU 制品，按 target）
├── model/         # 制品下载/准备与制品 README
├── runtime/       # python/（及提供时的 cpp/）推理实现
├── evaluator/     # 精度/性能评估
├── test_data/     # 示例输入与预期参照
└── README.md      # 本文件
```

<a id="entry-points"></a>
## 入口索引

> **必须回答：** 每个入口的链接＋一句话说明；缺失的入口写理由而不是链接。

- 模型准备：[`model/README.md`](model/README.md) —— ⟪一句话⟫
- Python 运行：[`runtime/python/README.md`](runtime/python/README.md) —— ⟪一句话⟫
- ⟪C++ 运行：[`runtime/cpp/README.md`](runtime/cpp/README.md) —— ⟪一句话⟫⟫
- 模型转换：[`conversion/README.md`](conversion/README.md) —— ⟪一句话或缺失理由⟫
- 模型评估：[`evaluator/README.md`](evaluator/README.md) —— ⟪一句话或缺失理由⟫

<a id="license"></a>
## 许可说明

> **必须回答：** 模型权重与示例代码的许可；与仓库顶层 LICENSE 的关系。

⟪如：代码遵循仓库 LICENSE；权重为 ⟪许可⟫（来源：⟪链接⟫）。⟫
