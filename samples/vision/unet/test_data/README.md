[English](./README.md) | [简体中文](./README_cn.md)

# UNet Test Data

`2007_000033.jpg` is the Pascal VOC 2012 validation image used by the Python
Runtime by default. It is intended only as a smoke input for image loading,
NV12 preprocessing, BPU inference, and result visualization.

- Original size: 500 × 366
- SHA256: `23b51ccd1a19c6f1f75573b1903e19015bf98c159b03d497efa8e912f8ffbe8e`
- Dataset: [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

Full accuracy evaluation requires a separate Pascal VOC download and
`evaluator/eval_unet.py`.
