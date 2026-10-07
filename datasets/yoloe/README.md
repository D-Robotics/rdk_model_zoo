English | [简体中文](./README_cn.md)

# YOLOE Dataset Resources

This directory carries the fixed prompt-free (PF) class vocabulary shared by
the [YOLOE sample](../../samples/vision/yoloe/README.md). It contains **no
images and no dataset download** — YOLOE evaluation datasets are COCO-format
annotation sets you prepare separately (see the
[YOLOE evaluator guide](../../samples/vision/yoloe/evaluator/README.md)).

<a id="files"></a>
## Files

- `yoloe_seg_pf_classes.names` — class names for the YOLOE-11/26 Seg
  Prompt-Free models: **4585 lines, one class per line, alphabetically
  sorted**, covering the open-vocabulary instance-segmentation vocabulary
  (0-based model output indices; line 1 = index 0).

It is **byte-identical** (SHA-256
`1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3`) to the
canonical copy `samples/vision/yoloe/test_data/classes.names`, which is the
hash-pinned reference the sample's conversion and runtime enforce. Use this fixed class order for PF output interpretation.

<a id="usage"></a>
## Usage

The canonical [YOLOE Python runtime](../../samples/vision/yoloe/runtime/python/README.md)
defaults `--label-file` to its own `test_data/classes.names`; this file is the
same vocabulary and can be passed explicitly instead. Commands run from the
**repository root**:

```bash
# cwd: repository root; prepare the variant first (see the runtime guide)
python3 samples/vision/yoloe/runtime/python/main.py \
  --target x5 --model-path samples/vision/yoloe/model/<prepared-artifact>.bin \
  --label-file datasets/yoloe/yoloe_seg_pf_classes.names
```

Substitute your prepared model path; the runtime guide documents all
parameters. Passing any other vocabulary file fails the sample's hash checks —
never edit the 4585-entry list.

<a id="pf-vs-coco"></a>
## PF class indices are not COCO category IDs

The 4585 indices are the model's open-vocabulary output order, **not** COCO
category IDs and **not** the 80-class Ultralytics COCO order. Example positions
verified in the file: index 2163 = `person`, index 821 = `chair`. Mapping PF
indices to dataset categories for scoring is an explicit, name-checked step in
the [YOLOE evaluator](../../samples/vision/yoloe/evaluator/README.md)
(`mapping.example.json` demonstrates person → COCO 1 and chair → COCO 62 and
is not a complete COCO-80 mapping).

For naming model outputs at runtime, the 80-class files live in
[datasets/coco](../coco/README.md) and the sample's `test_data/`; do not
substitute this 4585-entry vocabulary for them.

<a id="provenance"></a>
## Provenance and generation

The vocabulary is generated and checked during the PF export/preparation flow:

- [conversion/export.py](../../samples/vision/yoloe/conversion/README.md)
  exports a local PF checkpoint and writes `yoloe_<variant>_seg_pf.onnx`
  plus a `yoloe_<variant>_seg_pf.names` copy of the vocabulary next to it;
- conversion `prepare.py` then requires `--names` to match
  `samples/vision/yoloe/test_data/classes.names` **byte for byte** before it
  builds any conversion configuration.

Treat the exporter-generated `.names` as a derived copy; this directory and
the sample's `test_data/classes.names` are the fixed references.
