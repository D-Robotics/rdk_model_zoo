# Generation reference data

[简体中文](README_cn.md) | **English**

These prompts and measurements come from the source S release.

`legacy-prompts.json` supplies SDK 1.0.0 single/two-turn generation cases.

`prompts.json` contains six deterministic comparison prompts. `generation-reference.json` records official HF greedy text/token IDs and observed S600 outputs. EOS is removed from the reference token list because OELLM returns generated content tokens separately. This set is a deterministic generation check that supplements the full WikiText2 PPL evaluation. JSON and Python answers retain Markdown fences produced by the original model; do not rewrite them into a bare format the model did not produce.

`legacy-long-prompts.json` reproduces the approximately 2000/3750-token retrieval checks for SDK 1.0.0. Legacy generation evidence is recorded separately under `evaluator/results/s100-generation-full.json` and `s100p-generation-full.json`: only 2/6 reference texts match. The legacy SDK does not expose generated token IDs, so these are text comparisons. The six comparison cases above are S600 references.
