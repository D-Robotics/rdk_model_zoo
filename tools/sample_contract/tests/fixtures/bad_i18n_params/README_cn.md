# good_sample（正例 fixture）

仅供 `tools/sample_contract/check.py` 测试使用的最小合规样例。

<a id="overview"></a>
## 概述

仅用于检查器测试的 fixture 分类器；来源为本仓库测试夹具，无外部上游。

<a id="support-matrix"></a>
## 支持与实测矩阵

| Target | Variant | 语言 | 状态 |
| --- | --- | --- | --- |
| x5 | fixture1 | python | supported-not-run |
| s100 | fixture1 | python | supported-not-run |

未提供 C++，本样例不声称双语言支持。

<a id="prerequisites"></a>
## 环境前提

主机 Python 3.10+ 运行单元测试；推理需要带 `hbm_runtime` 的 RDK 板，
本 fixture 不执行推理。

<a id="quickstart"></a>
## 快速体验

在仓库根目录先准备制品再运行：

```bash
bash samples/tools/fixture/good_sample/model/download.sh --target x5
python3 samples/tools/fixture/good_sample/runtime/python/main.py \
  --target x5 --test-img samples/tools/fixture/good_sample/test_data/input.jpg
```

成功判据：退出码为 0 且打印 top-5 列表。

<a id="expected-results"></a>
## 预期结果

运行打印 fixture top-5 列表并退出 0；runtime 自身不写任何输出文件。

<a id="directory"></a>
## 目录职责

- `model/` — 制品准备（[README](README.md#overview) 之外的中文说明见 [model/README_cn.md](model/README_cn.md)）
- `runtime/python/` — Python 入口（[runtime/python/README_cn.md](runtime/python/README_cn.md)）
- `test_data/` — 随仓输入（[input.jpg](test_data/input.jpg)）

<a id="entry-points"></a>
## 入口索引

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行：[runtime/python/README_cn.md](runtime/python/README_cn.md)

本 fixture 不提供转换配方与评估实现；适用范围见样例根 README。

<a id="license"></a>
## 许可

fixture 内容遵循仓库顶层 LICENSE，无额外模型许可。
