English | [简体中文](README_cn.md)

# OSNet fixed inputs

<a id="inputs"></a>
## Inputs

No Market1501 images are committed. model/download.py --fixtures prepares a private images.dat containing decoded RGB crops and signed-planar reference bytes, an ordered manifest, and matching 32×512 float references. The Python default reads the first crop; the C++ launcher checks all 32. A user image can instead be passed via --test-img.

S-series HBMs come from target-specific private candidate bundles. Verify SHA256SUMS and artifacts.json, then pass --model-path; the X5 recovery ZIP/TAR does not contain these HBMs. See [S-series Workbench workflow](../evaluator/S_SERIES.md) for commands, calibration and fixed-input requirements.
