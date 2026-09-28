English | [简体中文](./README_cn.md)

# Pascal VOC Dataset Resources

**PASCAL VOC** (2007/2012) is a classic benchmark for object detection and
segmentation: 20 foreground classes plus background (21 classes for
segmentation), with JPEG images, XML box annotations and palette-indexed
segmentation masks. This directory contains **reference links only — no files
are bundled**, and there is no download script. Acquisition is manual from the
official hosts under their terms.

<a id="files"></a>
## Contents

No datasets, class lists or example images live in this directory. The paired
[Chinese guide](./README_cn.md) and this page carry the source references that
the delivery branches shipped here.

<a id="usage"></a>
## Where Pascal VOC is used in this checkout

| Consumer | Use |
| --- | --- |
| [UNet evaluator](../../samples/vision/unet/evaluator/README.md) | Evaluates **VOC 2012 segmentation**: a manifest of `JPEGImages/<id>.jpg` + `SegmentationClass/<id>.png` path pairs (tab-separated, absolute paths); 21-class logits with palette indices as class indices, `255` is the ignored void label. Masks must stay palette-indexed — do not convert them to grayscale |
| [UNet test_data](../../samples/vision/unet/test_data/README.md) | Bundles `2007_000033.jpg`, a single VOC 2012 validation image, for offline smoke checks |

No current sample performs VOC **detection**; class-index nomenclature for the
segmentation path above is: palette index = class index (`0` background,
`1`–`20` foreground, `255` ignore).

<a id="acquisition"></a>
## Acquiring the dataset

Download manually from the official host and point the UNet evaluator manifest
at your prepared absolute paths, for example:

```text
/data/VOC2012/JPEGImages/2007_000033.jpg
/data/VOC2012/SegmentationClass/2007_000033.png
```

- Official site: <http://host.robots.ox.ac.uk/pascal/VOC/>
- VOC 2012 kit: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/> (see the
  [UNet test_data guide](../../samples/vision/unet/test_data/README.md) for
  the same reference)
- Introductory article (Chinese, inherited source link):
  <https://blog.csdn.net/generalsong/article/details/108471378>

Obey the dataset's own terms of use; images and annotations remain with their
original copyright holders.

<a id="provenance"></a>
## Provenance

The X5 delivery branch (`ac11571`) shipped this directory with the same two
links in a nearly empty English README and a short Chinese README; the S
delivery line (`380e1a2`) carried no `PascalVOC/` directory at all. This guide
expands that inherited content with the current consumer map. Archived copies
remain under `platforms/x5/datasets/PascalVOC/`.
