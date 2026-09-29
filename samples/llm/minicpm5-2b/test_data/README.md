# Generation reference data

[简体中文](README_cn.md) | **English**

These prompts and measurements come from pinned S source `380e1a2`; none were regenerated or board-tested this round.

`legacy-prompts.json` supplies SDK 1.0.0 single/two-turn generation cases.

`prompts.json` contains six deterministic acceptance prompts. `generation-reference.json` records official HF greedy text/token IDs and observed S600 outputs. EOS is removed from the reference token list because OELLM returns generated content tokens separately. All six comparisons passed. This set supplements full WikiText2 PPL; it is not a general language-capability benchmark. JSON and Python answers retain Markdown fences produced by the original model.

`legacy-long-prompts.json` reproduces the approximately 2000/3750-token retrieval checks for SDK 1.0.0. Legacy generation evidence is recorded separately under `evaluator/results/s100-generation-full.json` and `s100p-generation-full.json`: only 2/6 reference texts match. The six passing comparisons above belong to S600.
