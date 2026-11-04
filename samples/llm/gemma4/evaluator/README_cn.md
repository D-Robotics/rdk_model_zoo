[English](README.md) | 简体中文

# 官方性能参考与本地验证

[Sample](../README_cn.md)

<a id="dataset"></a>
## 评估输入与来源

`../test_data/prompts.json` 提供手动文本检查提示；官方未发布性能测试的完整输入语料。

数字保存于 `official_benchmarks.json`，来源为 [SDK 官方 S600 基准](https://developer.d-robotics.cc/oe_llm_doc/guide/benchmark/s600/gemma_benchmark.html)。

<a id="directory"></a>
## 目录

`official_benchmarks.json` 保存官方参考；`board_results/<profile>.json` 单独保存本地板端记录。

<a id="environment"></a>
## 环境

官方条件为 S600/Linux、SDK 2.0.0、4 核、chunk 512、context 4096。主机文件检查仅依赖 Python 3 标准库。

<a id="command"></a>
## 检查命令

```bash
python3 -m unittest discover -s samples/llm/gemma4/tests
```

从仓库根目录运行。生成检查提示见 [Runtime](../runtime/cpp/README_cn.md#run)。

<a id="metrics"></a>
## 指标与官方参考

| Profile | TTFT (ms) | Decode (token/s) | Memory (GB) |
| --- | ---: | ---: | ---: |
| e2b-w4 | 74 | 61.2 | 9.9 |
| e2b-w8 | 79 | 47.8 | 13.5 |
| e4b-w4 | 137 | 38.3 | 15.4 |
| e4b-w8 | 139 | 27.9 | 15.5 |

TTFT 采用纯文本列。条件为 S600/Linux、4 核 BPU、chunk 512、context 4096、标准单请求模式。
内存为官方完整模型的 Linux+ION 总占用，不是峰值统计；不要写入 `peak_memory_bytes`。
CPU 线程数量未公布，网页的单路指请求并发。

<a id="outputs"></a>
## 本地输出

`board_results/<profile>.json` 是本项目的板端记录，目前全部为 `not-run` 与 null。
官方参考与本项目实测分开保存；没有公开 PPL、两路并发或端到端 C++ 分段数据的项目保持为空。
不沿用旧自转换模型的主机 PPL，也不把官方数据改写为本地测量。

<a id="reference-results"></a>
## 参考结果

上述表格为官方公开数据，未提供对应二进制 SHA256 绑定。主机校验结果与板端测量分开记录。

<a id="boundaries"></a>
## 适用范围

单请求参考用于选择官方配置；本项目未运行板端编译、加载、生成和性能测试。
