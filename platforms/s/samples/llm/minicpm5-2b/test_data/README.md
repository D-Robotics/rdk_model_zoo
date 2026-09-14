# Generation reference data

`prompts.json` contains six deterministic acceptance prompts. `generation-reference.json` records official HF greedy text/token IDs and observed S600 outputs. EOS is removed from the reference token list because OELLM returns generated content tokens separately. All six comparisons passed. This set supplements full WikiText2 PPL; it is not a general language-capability benchmark. JSON and Python answers retain Markdown fences produced by the original model.
