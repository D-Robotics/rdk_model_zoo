English | [简体中文](README_cn.md)

# ImageNet Dataset Resources

**ImageNet ILSVRC-2012 (ImageNet-1k)** is an image-classification dataset with
1,000 classes, about 1.28 million training images and 50,000 validation images.
Classification samples use this class order. This directory bundles the label
list and one example image; acquire the dataset separately under the official
terms (the validation set requires registration).

<a id="files"></a>
## Bundled files

```text
imagenet/
├── README.md                    # this guide
├── README_cn.md                 # Chinese guide
├── imagenet_classes.names       # 1000-class label list (dict-literal format)
└── asset/
    └── zebra_cls.jpg            # one example classification image
```

### imagenet_classes.names

The file is a **Python dict literal** spanning all 1000 classes,
keys are the model output class indices 0–999 in the standard ILSVRC-2012
order (index 0 = `tench, Tinca tinca`, index 999 = `toilet tissue, toilet
paper, bathroom tissue`); each value is a comma-separated list of synonymous
English names; the file has no trailing newline. Excerpt:

```python
{0: 'tench, Tinca tinca',
 1: 'goldfish, Carassius auratus',
 ...
 999: 'toilet tissue, toilet paper, bathroom tissue'}
```

This is **not** a one-name-per-line list, although the runtime loader accepts
that format too. `utils/py_utils/labels.py::load_labels` (re-exported by each
classification sample, and mirrored by the legacy
`utils/py_utils/file_io.py::load_labels`) detects the leading `{` and parses it
with `ast.literal_eval` into `{index: name}`; one-label-per-line display-name
text is also accepted. Each value is a comma-separated list of human-readable
synonym phrases — **display names only**: they are not WordNet synset IDs
(`n########`) and cannot substitute for them.

The keys are **class indices, not synset IDs**. ImageNet-1k has no separate
sparse category-ID space like COCO; index order and naming are fixed by this
file, and accuracy comparisons are only valid against the same file. Note that
`--label-file` does **not** mean the same thing everywhere: see the two
consumer rows below.

### asset/zebra_cls.jpg

The bundled zebra image is a classification example with ImageNet class index 340 (`zebra`). It is byte-identical to `samples/vision/resnet/test_data/zebra_cls.jpg` and `samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg`.

<a id="usage"></a>
## Where these resources are used

| Consumer | Use |
| --- | --- |
| All classification samples (e.g. [ResNet](../../samples/vision/resnet/README.md), [EfficientFormer](../../samples/vision/efficientformer/README.md), [RepViT](../../samples/vision/repvit/README.md)) | `--label-file` here means **runtime display names**: this dict-literal file (or one-label-per-line display-name text) is parsed by the shared loader to name Top-1/Top-5 results (path relative to the repository root) |
| [Ultralytics YOLO classification evaluator](../../samples/vision/ultralytics_yolo/evaluator/README.md) | Its `--label-file` means something different: an ordered list of `n########` synset IDs in model-class order, matched against identifiers in the image filenames. This display-name file **cannot** be passed there. Ground truth comes from `--val-txt` (named format `<relative-image-path> <zero-based-class-index>`, or the ordered variant) — see the evaluator's dataset section for the exact conventions |

Commands in the sample guides run from the **repository root**, so the label
path is exactly `datasets/imagenet/imagenet_classes.names`. If you run from
another directory, pass the corrected relative or absolute path.

<a id="acquisition"></a>
## Acquiring the dataset

There is **no download script** in this directory (unlike
[COCO](../coco/README.md)). Prepare the validation set yourself, for example:

```text
datasets/imagenet/val_images/   # .gitignore excludes this path
```

`.gitignore` already excludes `datasets/imagenet/val_images/*`; never commit
dataset images or ground-truth lists. The Ultralytics classification evaluator
documents its expected image/label-list layout in its
[dataset section](../../samples/vision/ultralytics_yolo/evaluator/README.md#dataset).

Official sources — obtain access and use the data under their terms:

- ImageNet: <https://image-net.org/> (registration required for downloads)
- Hugging Face mirror: <https://huggingface.co/datasets/ILSVRC/imagenet-1k>
  (gated dataset)

<a id="provenance"></a>
## Provenance

Dataset source: [ImageNet](https://image-net.org/). `imagenet_classes.names` maps model output indices to display names. Dataset scoring additionally requires an image-to-ground-truth-class mapping in the same 0–999 class order.
