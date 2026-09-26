# B8 LaneNet — source contract audit and migration decisions

Status: source audit and shared dtype prerequisite only; LaneNet implementation
is not migrated or accepted. Source is S
`380e1a2bf42041af54be6f34935e50197cfadff9`. The
[host capture](evidence/2026-09-26-b8-lanenet-audit/capture.py) verifies 21 source
files byte-for-byte and records reproducible numerical counterexamples in
[result.json](evidence/2026-09-26-b8-lanenet-audit/result.json).
No board/HP/SSH/model-download action occurred.

## Source capability and published identity

One S100 artifact: `s:lanenet:s100/lanenet256x512.hbm`, publisher SHA-256 unknown.
Both Python and native UCP/DNN implementations exist. Source main/run wrappers
warn on other SoCs but continue with S100; canonical execution must reject
S100P/S600/X5 before model loading or native build. Preparation must be explicit.
Source default `/opt/hobot/model/s100/basic/lanenet256x512.hbm` becomes a sample
model path; explicit external copies remain possible with exact contract identity.

Input is RGB float32 NCHW [1,3,256,512]. Both implementations use INTER_AREA,
`/255` and ImageNet mean/std; unlike Depth Anything V2 this is actual ImageNet
normalization. Python swaps channels before resize; native swaps after resize.
These channel-independent operations must be fixture-compared. No NV12 input is
implemented despite the conversion output prefix `lanenet256x512_nv12`.

Python consumes named `instance_seg_logits` and `binary_seg_pred`; native assumes
indices0/1 and documents F32 [1,3,H,W] plus S64 [1,1,H,W]. Python reshapes away
rank details without validating type/shape. The conversion prose claims three
outputs, but supplies neither an output manifest nor the referenced accuracy
image. Do not invent a third output name, require an unproven exact output count,
or silently treat output0 as the embedding in a reordered model. Required roles
must be bound explicitly; auxiliary outputs, if present, remain raw and identified
by observed metadata. Actual HBM metadata remains unobserved.

## Important discrepancies

1. **There is no lane clustering.** The three-channel embedding is directly
   scaled into a color image; the implementation does not produce stable lane
   IDs, curves, instance assignments or TuSimple predictions. Source root claims
   that colors identify separate lanes are stronger than the code supports.
   Preserve raw embeddings and binary labels, explain display meaning, and do
   not invent a new clustering algorithm as if it were source behavior.
2. **Python and native displays disagree.** Python multiplies by255 then casts
   to uint8 (truncation and out-of-range wrap); native clips to[0,1] then uses
   OpenCV saturating conversion/rounding. Even .5 differs (127 versus128).
   The audit records negative, fractional and >1 counterexamples. Canonical
   visualization should unify on explicit clipping/round-to-nearest, separately
   from raw float output. This deliberately changes Python display bytes and
   must not be sold as bit-identical source output.
3. **Binary invalid labels are inconsistently accepted.** Python multiplies
   arbitrary values by255 then casts, while native maps every nonzero value to255.
   The public result should validate discrete0/1 labels, preserve integer values,
   and leave display scaling outside postprocess. Source native S64 requires
   explicit int64 metadata-token support; internal quantization prose alone is
   not grounds for dequantizing a discrete label output.
4. **Native safety is incomplete.** Initialization macros can return after
   acquiring resources; buffer-preparation returns are ignored; inference task
   release is skipped on submit/wait failure. Metadata count/rank/layout/dtype
   and capacity are not validated. Embedding reads use row/channel strides but
   assume contiguous width; binary reads use width stride. Canonical SDK ownership
   needs RAII and checked stride-aware copy, while task stages remain SDK-free.
5. **Source output stays at256×512.** Do not silently restore to the original
   image and call it source parity. Retain model-grid embeddings and binary labels;
   any future overlay/resize must be explicit and distinct from raw results.

## Conversion and README recovery

A real YAML exists and must be byte-preserved. It sets nash-e, featuremap NCHW
[1,3,256,512], float32 calibration, set_all_nodes_int16, latency mode and O2.
Its ONNX path `../log/best_model.onnx` and calibration path `../cal_data` are
unprepared external prerequisites. The nv12 filename suffix does not establish
an NV12 tensor boundary.

Source documentation links a checkpoint URL and commands `test.py` and
`get_calibration_data.py`, but those scripts are absent. The compile example
`source/yaml/config.yaml` does not point to the local file. The accuracy figure
`test_data/readme_img/result.jpg` and linked Chinese conversion README are absent.
Preserve these as disclosed source gaps, rather than provide broken runnable
steps or an invented figure. A useful canonical guide must distinguish a retained
config, validated caller-supplied data/ONNX and actual OE compilation.

Preserve the source's 200-frame14.245ms/69.894FPS performance reference as
historical, without invented dataset, firmware, artifact digest or thread context.
The source evaluator is only a placeholder. Preserve the four Python/native
reference PNGs plus lane.jpg as historical visual examples. Add all six directory
levels of bilingual documentation, including native build/dependencies/API/resource
lifetime and a truthful evaluation boundary. The root must acknowledge native
capability despite its opening sentence naming Python only.

## Shared prerequisite

`canonicalise_dtype` previously left S64 unknown; that was explicitly tested
because no migrated runtime needed it. LaneNet's documented native label output
now supplies a concrete need: canonicalize s64/i64/int64/SDK-S64 spellings to int64.
This is spelling normalization, not global acceptance of int64. Existing sample
bindings retain their dtype allowed sets, and tests preserve unknown S128/U64
behavior and ensure classification allowed sets do not expand.

Native and Python stage implementations, conversion handling, twelve READMEs,
full host verification and independent acceptance are still required. This audit
must not update the migration row to done or close B8.
