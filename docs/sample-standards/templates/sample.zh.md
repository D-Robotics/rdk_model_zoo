[English](README.md) | 简体中文

<!-- 使用模板时保留章节锚点，填写 ⟪…⟫，完成后删除引导文字。与英文版本同步维护。 -->

# ⟪模型名称⟫（⟪任务⟫）

<a id="overview"></a>
## 算法与来源

> **必须回答：** 模型做什么（一句话）；算法简述；官方论文/仓库链接；在本仓的定位。

⟪用一两句话介绍模型的用途及主要方法，例如“ResNet 通过残差连接训练深层图像分类网络。”⟫

来源：⟪论文/上游仓库链接⟫

⟪板卡与变体写入支持矩阵，输入输出和 API 写入运行章节。不要在正文解释文档的写作方式。⟫

<a id="directory"></a>
## 目录结构

⟪列出当前目录的实际文件及用途。⟫

<a id="support-matrix"></a>
## 支持矩阵

> **必须回答：** 提供哪些 target × variant × 语言组合、对应制品及所需 SDK 或模型配置。
> 按实际提供的 Python 和 C++ 入口填写。

| Target | Variant | 语言 | 制品及环境要求 |
| --- | --- | --- | --- |
| ⟪x5 / s100 / s100p / s600⟫ | ⟪variant⟫ | ⟪Python / C++⟫ | ⟪实际制品和 SDK⟫ |

⟪需要额外模型制品或 SDK 的组合，在此写明具体前提。⟫

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
> 来源、参数、输出与成功判断。使用实际脚本和目标选择参数显式准备模型。

```bash
# cwd：仓库根目录
⟪使用实际脚本和支持的参数准备模型⟫
# 预期：制品位于 ⟪path⟫；根据下载器实际行为注明校验结果或本地 SHA-256

# cwd：仓库根目录
python3 samples/⟪domain⟫/⟪name⟫/runtime/python/main.py ⟪实际目标选择参数⟫ ⟪input⟫
# 预期：⟪输出内容，如打印 Top-5 / 生成结果文件 ⟪path⟫⟫
```

⟪若存在 run.sh 快捷脚本，在显式路径之后并列给出。⟫

<a id="expected-results"></a>
## 预期结果

> **必须回答：** 正确运行应看到什么——test_data 的真实输出形态/数值、输出文件路径与
> 命名。禁止虚构精度数字。

⟪如 test_data/⟪image⟫ 的 Top-5 = […]；结果图写入 ⟪path⟫⟫

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
