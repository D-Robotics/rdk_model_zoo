English | [简体中文](README_cn.md)

# Official references and local validation

[Sample](../README.md)

<a id="dataset"></a>
## Evaluation inputs and sources

`../test_data/prompts.json` supplies manual text-check prompts; the vendor does not publish its full performance input corpus.

`official_benchmarks.json` records the [vendor S600 reference](https://developer.d-robotics.cc/oe_llm_doc/guide/benchmark/s600/gemma_benchmark.html).

<a id="directory"></a>
## Directory

`official_benchmarks.json` stores vendor references; `board_results/<profile>.json` independently stores local board records.

<a id="environment"></a>
## Environment

The official conditions are S600/Linux, SDK 2.0.0, four cores, chunk 512 and context 4096. Host file checks use only the Python 3 standard library.

<a id="command"></a>
## Check command

```bash
python3 -m unittest discover -s samples/llm/gemma4/tests
```

Run from the repository root. For manual generation prompts see [Runtime](../runtime/cpp/README.md#run).

<a id="metrics"></a>
## Metrics and official reference

| Profile | TTFT (ms) | Decode (token/s) | Memory (GB) |
| --- | ---: | ---: | ---: |
| e2b-w4 | 74 | 61.2 | 9.9 |
| e2b-w8 | 79 | 47.8 | 13.5 |
| e4b-w4 | 137 | 38.3 | 15.4 |
| e4b-w8 | 139 | 27.9 | 15.5 |

TTFT uses the text-only column. Conditions: S600/Linux, four BPU cores, chunk 512, context 4096, standard single-request generation.
Memory is the full vendor model's Linux+ION total, not a peak-memory statistic; it must not populate `peak_memory_bytes`.
CPU thread count is unspecified. The website's single-stream selector denotes request concurrency.

<a id="outputs"></a>
## Local outputs

Local `board_results/<profile>.json` entries remain `not-run` and null.
Published references are stored separately from local measurements. PPL, two-stream concurrency and C++ stage timings are not fabricated.
Host PPL from the retired self-converted package is not attributed to these official profiles.

<a id="reference-results"></a>
## Reference results

The table above contains published vendor data without a binary-SHA binding. Host integrity checks remain separate from board measurements.

<a id="boundaries"></a>
## Scope

Single-request references help select an official profile. This project has not run board build, load, generation or performance tests for this publication.
