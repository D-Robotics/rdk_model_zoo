# Generation reference data

`prompts.json` contains six deterministic acceptance prompts. `generation-reference.json` records official HF greedy text/token IDs and observed S600 outputs. EOS is removed from the reference token list because OELLM returns generated content tokens separately. All six comparisons passed. This set supplements full WikiText2 PPL; it is not a general language-capability benchmark. JSON and Python answers retain Markdown fences produced by the original model.

`legacy-long-prompts.json` reproduces the approximately 2000/3750-token retrieval checks for SDK 1.0.0. Legacy generation evidence is recorded separately under `evaluator/results/s100-generation-full.json` and `s100p-generation-full.json`: only 2/6 reference texts match. The six passing comparisons above belong to S600.
