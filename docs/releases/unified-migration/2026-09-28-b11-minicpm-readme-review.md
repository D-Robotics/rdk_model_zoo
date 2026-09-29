# B11 MiniCPM all-layer README organization — author record

Base `f42a31d5`. Documentation organization has progressed; native core migration
and whole-branch independent acceptance remain open. No quantization recipe,
model download, evaluation or board test was executed.

## Documentation changes

Twenty English/Chinese README files now cover root, model, runtime overview,
S600 native runtime, legacy native runtime, both conversion/evaluation paths and
test_data. The missing Chinese test-data guide is supplied. Root guides expose the
actual target/SDK/precision matrix and direct links to every task. Model guides
retain archive sizes/digests and explain environment overrides, companion files
and integrity checks. Native guides distinguish build/run, wrapper/native flags,
SDK package vs runtime versions, session ownership and RESULT interpretation.

Conversion documentation separates source/toolchain, preparation, calibration,
compile, validation and packaging without adding fictitious standalone ONNX steps.
Original executable shell lines and ordering are preserved across all eight
conversion/evaluator guides; added section breaks do not rewrite recipes. Legacy
instructions remain complete. Evaluation guides separate full-run completeness,
precision thresholds, text/token matching and performance scope.

Historical measurements remain explicitly attributed to fixed S source `380e1a2`:
S100/S100P PPL +27.83% fails the <=3% target and only 2/6 reference texts match;
S600's +1.60% and six token matches belong to its own artifact. Source SDK release
forecasts are not represented as newly verified current release information.

## Host evidence and scope

[Evidence](evidence/2026-09-28-b11-minicpm-readme/) includes all 20 README hashes,
115 existing relative-link targets, exact original shell-line comparison for eight
conversion/evaluator guides, S100P run/S600 build previews and contract output.
The migration checker now reports 51 samples / 0 violations / 51 policy skips /
0 exemptions; the former 70 MiniCPM section gaps are resolved without exemptions.

This is documentation verification only. Source implementation and recipe scripts
were not changed, and code tests were not redundantly rerun for prose/structure.
Two documented dry-run examples execute the host launcher only; they do not load
models or launch SDK/build processes. `git diff --check` passes.

README contract success is not full semantic or implementation acceptance. MiniCPM
native stage/resource refactoring, corresponding final API documentation, manifest
integration and final independent review remain open. Gemma's ledger also now
reflects its previously committed ownership/preparation tests rather than its older
13-test snapshot. H0–H9 remain active; board status is not-run.
