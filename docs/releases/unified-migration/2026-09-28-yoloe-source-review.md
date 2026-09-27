# YOLOE source, artifact and vocabulary audit

2026-09-28 implementation preparation; base `32b9c780e867b0f4e85e45fafcd910f2542e36eb`.
This report advances H5/H8 source completeness. Unified YOLOE implementation,
customer documentation and independent acceptance remain pending.

## Preserved source capabilities

| Source at fixed commit | Published scope in source | Decode contract | Interface/content to preserve |
| --- | --- | --- | --- |
| X5 `ac115717197920355fc390bb04299b20e6436864`, `samples/vision/yoloe` | YOLOE-11 s/m/l, packed NV12, 640 square | 4585 classes, DFL16, NMS, 32 mask coefficients | Python, full-image masks, export patch and PTQ YAML, bilingual instructions, three historical runtime-only performance rows |
| S `380e1a2bf42041af54be6f34935e50197cfadff9`, `samples/vision/yoloe11_seg` | 11s nash-e, S100; no published S100P/S600 asset | DFL16, NMS, 32 coefficients, ROI masks | Python and OpenMP C++, conversion configuration and intermediate/final compiler logs, example image and vocabulary |
| Same S commit, `samples/vision/yoloe26_seg` | n/s/m/l/x on nash-e and nash-m; S600 excluded | direct LTRB (`reg_max=1`), end-to-end Top-K, **no NMS**, 32 coefficients | Python/C++, single-label or multiple anchor-class selection, release JSON/names identity, conversion/export/mapper, source performance tables and tests |

These are prompt-free samples with a checkpoint-ordered 4585-class vocabulary.
They do not provide arbitrary text/visual prompting. Ordinary Ultralytics YOLO26
segmentation is not a substitute for YOLOE-26's candidate selection contract.

The absent 30 YOLOE-26 files have been restored **byte-for-byte** into
`platforms/s/samples/vision/yoloe26_seg`, with original modes, licenses, tests,
images and every README. [Source manifest](evidence/2026-09-28-yoloe-source-audit/restored-source.json)
records full commit, blob, SHA-256 and size. The 42 existing X5/S11 source files
also match their fixed commits byte-for-byte; see `existing-source.json` in the
same evidence directory. This is historical source restoration,
not a completed canonical `samples/vision/yoloe` migration. Source dates and board
claims remain historical; no new board verification is implied.

## Output precision: do not infer float HBM from an intermediate report

X5 Python checks floating output arrays. S YOLOE-11's intermediate
`hb_model_info_yoloe_11s_seg.txt` lists ten FLOAT32 outputs, while its final
`hb_combine_yoloe_11s_seg.txt` records FLOAT32 class heads, INT32 box/coefficient
heads and an INT16 prototype. The source removes selected Dequantize nodes during
compilation. A similarly named `hrt_model_exec_model_info_yolow_11s_seg.txt`
actually describes a v8s intermediate graph; filenames alone are not contracts.
The published 11s HBM has not been independently downloaded/inspected this round.

For YOLOE-26, the two public manifests and twenty JSON/names sidecars were captured
and the sidecars verified against manifest sizes/hashes. All ten JSONs state
`hbm_output_quantized=true`. Seven also specify nine INT32 outputs plus INT8 proto;
three nash-e sidecars (s/l/x) omit the dtype list, so no extra field is invented.
The mapper explicitly removes `Quantize;Dequantize`. Current floating-only
Ultralytics binding **cannot load these published quantized artifacts**.

Maintain the user's floating-output preference: do not silently enable a second
manual dequantization path in the task postprocessor or pretend these assets are
float-compatible. Migration must retain dequantization nodes in the float-output
conversion route and validate the resulting descriptors; any still-unbuilt or
unpublished float asset must be stated as such. Retain original source/asset
provenance while this work proceeds. This is a concrete conversion and asset gap,
not justification for dropping YOLOE or declaring its migration complete.

## Asset identity and labels

[Captured records](evidence/2026-09-28-yoloe-source-audit/release-capture.json)
retain exact public URLs, timestamps, lengths and hashes. The ten HBM digests and
sizes agree between each manifest and corresponding JSON. Those publisher digests
now populate the previously-null entries in **active** `docs/release/s/models.yaml`;
summary counts are 45 recorded / 327 unrecorded. Archival platform manifests are
unchanged. The HBM bytes themselves were not fetched: this establishes expected
download checksums, not independent validation of model bytes or inference.

All ten names files have SHA-256
`1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3`.
Their 4585 ordered strings match X5's dataset vocabulary, S11's sample vocabulary,
S26's source vocabulary and each JSON's `names`. A shared vocabulary is therefore
supported by evidence; runtime selection must still bind it to the chosen asset.

## Validation and next implementation work

[Offline verifier](evidence/2026-09-28-yoloe-source-audit/verify_source.py) checks
restored bytes against Git blobs, captured bytes against digests, sidecars against
manifests, active HBM hashes, labels and the intermediate/final S11 dtype distinction.
The four restored S26 Python source tests passed with only unavailable hardware
imports substituted. Its C++ regression explicitly needs a real HBM/image and SDK;
it was not presented as a host test and was not run.

Publisher source validation, 121 tests, catalog build/typecheck and reproducibility
check passed (57 families / 812 benchmark rows). Two initial source-validation
failures caught stale recorded/unrecorded summary counts; both logs are preserved
and the counts corrected. Generated catalog is not a publication or new release.
The final host run also passed 408 current migration tests, the four source tests,
44-sample contract checking (0 violations / 46 policy skips / 0 exemptions),
128 Ultralytics README links and 16 bilingual examples. The offline source
verifier was additionally rerun after including all 42 existing source files.
Full migration completion remains open. Next work must implement canonical stages,
separate E11/E26 selection/math policies, conversion/asset contracts and complete
bilingual sample/model/runtime/conversion/evaluator instructions with source
performance provenance and host-executed examples.

Board, real SDK/model inspection, actual OE compilation, new performance and
accuracy evaluation: **not-run**. No HBM download or hardware connection occurred.
