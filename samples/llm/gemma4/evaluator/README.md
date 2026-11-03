# Accuracy and performance records

[简体中文](README_cn.md) · [Sample](../README.md)

`host_reference.json` records full WikiText2 test evaluation: 142 sequences of
2048 tokens. LightCompress float/fakequant PPL is 41.21096802/43.29276657
(+5.05%). HPP float/fakequant PPL is 41.2612/43.0350 (+4.30%). Compare each pair
within its own path. HPP meets the reference 5% gate, but not the 3% stretch goal.
Host SDK generation produced `4` and `你好` for the recorded prompts.

These are host fake-quant references. `board_result.json` keeps board PPL,
TTFT, decode token/s, peak memory, generation checks and environment null, with
status `not-run`. This integration does not execute board tests.

Future board evaluation can use the SDK's `examples/ppl_demo` and its README,
pointing at the prepared runtime config. Align tokenizer, template, prefixes,
segmentation and label masks before comparing PPL, and preserve full NLL/token
counts. Generation prompts are in `test_data/prompts.json`. Record prompt/output
lengths, concurrency, cache state, runtime, OS and hardware frequencies for
performance; host CUDA timings are not board measurements.
