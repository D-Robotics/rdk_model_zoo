English | [简体中文](README_cn.md)

# Vendor references and local records

<a id="dataset"></a>
## Inputs and sources

`../test_data/prompts.json` supplies manual inputs. Vendor tables publish no complete benchmark corpus or binary SHA256 binding; conditions and original rows are preserved in `official_benchmarks.json`.

- [S600 LLM](https://developer.d-robotics.cc/oe_llm_doc/guide/benchmark/s600/llm_benchmark)
- [S600 VLM](https://developer.d-robotics.cc/oe_llm_doc/guide/benchmark/s600/vlm_benchmark)
- [S600 ASR](https://developer.d-robotics.cc/oe_llm_doc/guide/benchmark/s600/asr_benchmark)

<a id="directory"></a>
## Directory

`official_benchmarks.json` stores per-profile vendor values and conditions; `board_results/` stores separate empty local records.

<a id="environment"></a>
## Reference environment

S600/Linux. Standard LLM uses four cores, prefill 256 and context 2048. VLM uses 448×448 images/context 2048; per-row chunk and cores are preserved, including one-core Qwen3.5 4B. Speculative modes retain their separate table conditions. ASR uses four cores with sampling/window parameters from its SDK configuration. CPU thread count is unspecified.

<a id="command"></a>
## Host checks

```bash
python3 -m unittest discover -s samples/llm/oellm/tests
```

For generation/transcription commands see [Runtime](../runtime/cpp/README.md#run).

<a id="metrics"></a>
## Metrics

TTFT is first-token latency. VLM TTFT already includes vision and prefill, so do not add `vision_ms` again. Decode is token/s; speculative speed varies by request and retains the vendor min–max range. Memory is Linux+ION total rather than peak. RTF is processing time divided by audio duration.

<a id="outputs"></a>
## Local records

Every `board_results/<profile>.json` is `not-run`; PPL, TTFT, decode, peak memory, RTF, generation checks and environment are null.

<a id="reference-results"></a>
## Vendor reference values

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
## Scope

Vendor references remain separate from local tests. Unknown revisions/binary bindings remain unknown, and measurements from neighboring sizes, Runtime versions or acceleration modes are not mixed. Board tests have not been run for this integration.
