[English](README.md) | 简体中文

# 官方参考与本地记录

<a id="dataset"></a>
## 输入与来源

`../test_data/prompts.json` 给出手动输入示例。官方公开表未提供完整基准输入集及二进制 SHA256 绑定；条件与原始表行保存于 `official_benchmarks.json`。

- [S600 LLM](https://developer.d-robotics.cc/oe_llm_doc/guide/benchmark/s600/llm_benchmark)
- [S600 VLM](https://developer.d-robotics.cc/oe_llm_doc/guide/benchmark/s600/vlm_benchmark)
- [S600 ASR](https://developer.d-robotics.cc/oe_llm_doc/guide/benchmark/s600/asr_benchmark)

<a id="directory"></a>
## 目录

`official_benchmarks.json` 保存逐配置官方数字与条件；`board_results/` 保存独立的本地空记录。

<a id="environment"></a>
## 参考环境

S600/Linux。标准 LLM 为 4 核、prefill 256、context 2048；VLM 图像为 448×448、context 2048，chunk 与核心数按各行保存，Qwen3.5 4B 为 1 核。投机模式保留其单独表条件。ASR 使用 4 核，采样与窗口参数见配套 SDK 配置。CPU 线程数未公布。

<a id="command"></a>
## 主机检查

```bash
python3 -m unittest discover -s samples/llm/oellm/tests
```

生成或转写命令见 [Runtime](../runtime/cpp/README_cn.md#run)。

<a id="metrics"></a>
## 指标

TTFT 为首字延迟；VLM 的 TTFT 已包含视觉与 prefill 总耗时，不能再次加上 `vision_ms`。Decode 为 token/s；投机速度随请求变化，保留官方 min–max 范围。内存为 Linux+ION 总占用，不是峰值。RTF 为处理耗时/音频时长。

<a id="outputs"></a>
## 本地记录

全部 `board_results/<profile>.json` 为 `not-run`，PPL、TTFT、decode、峰值内存、RTF、生成检查与环境均为 null。

<a id="reference-results"></a>
## 官方参考值

| Profile | TTFT (ms) | Decode (token/s) | Memory (GB) | RTF |
| --- | ---: | --- | ---: | ---: |
| internvl2-2b-w4 | 94 | 97.1 | 2.9 | — |
| internvl2-5-1b-w4 | 125 | 163.6 | 1.7 | — |
| internvl3-5-1b-w4 | 103 | 115.1 | 2.2 | — |
| minicpm5-2b-w4 | 57 | 75.5 | 2.6 | — |
| phi4-mini-w8 | 55 | 36.5 | 5.9 | — |
| qwen2-5-3b-w4 | 56 | 64.6 | 3.4 | — |
| qwen2-5-vl-3b-w4 | 156 | 62.7 | 5.5 | — |
| qwen2-5-vl-7b-w4 | 221 | 36.8 | 9.3 | — |
| qwen3-0-6b-w8 | 29 | 105.5 | 2 | — |
| qwen3-1-7b-w4 | 34 | 82.3 | 2.7 | — |
| qwen3-4b-instruct-2507-w4 | 71 | 48.2 | 5.4 | — |
| qwen3-4b-thinking-2507-w4 | 71 | 48.3 | 5.4 | — |
| qwen3-4b-w4 | 72 | 48.3 | 5.4 | — |
| qwen3-4b-w4-dflash | 162 | 40.0–193.5 | 7.7 | — |
| qwen3-4b-w4-eagle3 | 88 | 48.9–89.3 | 5.5 | — |
| qwen3-5-2b-w4 | 150 | 84.0 | 4.3 | — |
| qwen3-5-4b-w4 | 251 | 30.4 | 5 | — |
| qwen3-5-9b-w4 | 234 | 30.2 | 9.8 | — |
| qwen3-8b-w4 | 92 | 33.3 | 9.1 | — |
| qwen3-8b-w4-dflash | 204 | 27.0–124.4 | 14.1 | — |
| qwen3-8b-w4-eagle3 | 107 | 48.5–77.1 | 8.4 | — |
| qwen3-8b-zh-w4-eagle3 | 107 | 34.2–82.0 | 8.4 | — |
| qwen3-asr-0-6b-w8 | 126 | 114.8 | 2.7 | 0.05 |
| qwen3-asr-1-7b-w8 | 123 | 66.6 | 4.6 | 0.05 |
| qwen3-vl-2b-w4 | 87 | 81.0 | 3.6 | — |
| qwen3-vl-2b-w4-eagle3 | 145 | 72.8–108.8 | 5.2 | — |
| qwen3-vl-4b-w4 | 139 | 45.5 | 6.5 | — |
| qwen3-vl-4b-w4-eagle3 | 197 | 48.1–68.6 | 7.4 | — |
| qwen3-vl-8b-w4 | 169 | 31.9 | 10.2 | — |
| qwen3-vl-8b-w4-eagle3 | 258 | 33.9–68.2 | 10.7 | — |
| whisper-medium-w8 | 66 | 52.3 | 2.2 | 0.09 |
| whisper-small-w8 | 27 | 135.5 | 0.6 | 0.04 |


<a id="boundaries"></a>
## 适用范围

官方参考与本地测试分开。源码 revision 或二进制绑定未知时不补造，也不把邻近尺寸、不同 Runtime 或不同加速模式的测量混用。当前接入未进行板端测试。
