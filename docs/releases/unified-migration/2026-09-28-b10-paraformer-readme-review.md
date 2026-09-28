# Paraformer README contract and content review

Base: `ec65fefcc8d4d16d8df6312a3f6b1e330907b668`.
Status: author documentation correction, not independent whole-sample acceptance.
No runtime behavior or board evidence was changed in this step.

Direct sample checking, previously outside migration-scope checking while pending,
found 78 violations: missing standard navigation sections, Python section ordering
and CLI default-table drift. These were fixed in content rather than adding a
checker exemption. The current direct sample check reports 0 violations, 1 existing
CLI-layer policy skip, 0 exemptions.

## Customer-visible changes

- Python installation/usage/parameters/results now precede embedding examples and
  detailed stage mathematics. All numerical, binding and frontend explanations stay.
  The table separates argparse `null` from the application's effective defaults and
  uses repository-relative spellings for sample-local CMVN/vocabulary locations.
- Model instructions explain why vocabulary, CMVN and source config are needed,
  where preparation places them, and why the runtime uses bundled `model/am.mvn`
  while package preparation also copies `s100/am.mvn`. An alternate download root
  does not silently change runtime model paths. Unknown publisher HBM hashes remain
  unknown, explicitly separate from locally pinned auxiliary-file hashes.
- Native documentation adds the actual support matrix, puts host build instructions
  before run instructions and locates the parameter anchor at the launcher table.
  Existing numerical, SDK lifetime, preflight and feature-reader details remain.
- Conversion distinguishes source, environment, export, calibration, compile,
  validation, artifacts and remaining gaps. Host numerical/array checks, compiler
  success, host simulation and board output verification are different claims.
- Evaluator instructions describe custom dataset manifest preparation, separate CER
  normalization from report fields and put historical metrics beside their limits.
  Its model paths now use `outputs/paraformer_export`, matching the conversion
  quickstart. Previously the evaluator used a hyphen while conversion used an
  underscore, so copying both commands literally would not find the export.

## Evidence and limits

[Direct contract result](evidence/2026-09-28-b10-paraformer-readme/contract.json)
checks section/order/CLI facts, not complete usability or inference correctness.
[Preservation check](evidence/2026-09-28-b10-paraformer-readme/content-preservation.json)
compares all 14 bilingual README files to the base: 72 fenced examples preserved,
except the intentional evaluator path correction; 132 local file links resolve.
These are link existence checks, not automatic anchor or external-web verification.
No runtime tests were repeated for this documentation-only edit; the preceding
75-test sample result still applies to unchanged code.

## Remaining API finding

The runtime currently exposes `ParaformerPipeline.predict` and raw callable stage
runners. The inference contract requires public pre/forward/post interfaces at
each model stage, plus clear stage attribution on errors. The checker does not
prove that requirement: the pipeline presently calls raw runners directly and
some errors contain only a tensor name or SDK exception. The next implementation
step must expose these boundaries without hiding the next model call inside a
postprocess method, preserve CPU CIF explicitly between predictor and decoder,
and test raw-output purity, explicit-stage/predict equivalence, owned state and
failure attribution. Runtime README/API examples must then follow the real API.

Keep Paraformer Refactor/Docs pending until that code/document audit is complete;
do not promote it merely because standard anchors and default tables now pass.
H0–H9 and final independent whole-branch review remain open. Actual OE/HMCT/SDK
validation and board execution remain separately not-run.
