English | [简体中文](./README_cn.md)

# Model Evaluation

## Runtime Performance

S100 V1P0, RDK OS 4.0.5-Beta, UCP 3.13.6 / HBRT 4.7.5.
HBM: OE 3.7.0, HBDK 4.7.5, INT8 KL, batch 1, 640x640 NV12, 4585 classes.
Measured on 2026-09-08 with `hrt_model_exec perf`, 200 frames, warmup enabled,
thread_num=1 and core_id=0.

Only **model Runtime** latency and throughput are published. Application
preprocessing, output dequantization, postprocessing, visualization and I/O are
excluded. These are not Python/C++ end-to-end FPS.

| Board | Model | Runtime latency (ms) | Runtime FPS |
|---|---|---:|---:|
| S100 | YOLOE-26n Seg PF | 4.943 | 200.74 |
| S100 | YOLOE-26s Seg PF | 9.944 | 100.08 |
| S100 | YOLOE-26m Seg PF | 11.765 | 84.55 |
| S100 | YOLOE-26l Seg PF | 13.417 | 74.18 |
| S100 | YOLOE-26x Seg PF | 22.013 | 45.31 |
| S100P | YOLOE-26n/s/m/l/x Seg PF | Not measured | Not measured |

```bash
hrt_model_exec perf --model_file /path/model.hbm \
  --thread_num 1 --core_id 0 --frame_count 200 --enable_warmup true \
  --profile_path ./profile
```

## Runtime Contract Regression

From the `samples/vision/yoloe26_seg/` sample root, install Python 3, NumPy,
OpenCV and SciPy, then run:

```bash
python3 evaluator/test_runtime.py
```

The suite checks box-local mask rendering, clipped and degenerate ROI alignment,
the staged preprocessing/forward contract, and scheduling updates. On a
development host that cannot import `hbm_runtime`, it substitutes only the
unavailable hardware binding; on a board it imports the installed binding.
Repository preprocessing and visualization code and the NumPy/OpenCV/SciPy
dependencies remain real. This is a contract regression, not HBM inference,
dataset mAP evaluation, or a board performance benchmark.

On an S100/S100P with the C++ runtime dependencies installed, run the hardware
contract check from the same sample root (replace the HBM path for your board):

```bash
g++ -std=c++17 -O2 evaluator/cpp/test_cpp_contract.cpp \
  runtime/cpp/src/yoloe26seg.cpp -Iruntime/cpp/inc -I../../../utils/c_utils/inc \
  -I/usr/hobot/include -L/usr/hobot/lib -ldnn -lhbucp \
  $(pkg-config --cflags --libs opencv4) -o /tmp/yoloe26_cpp_contract
/tmp/yoloe26_cpp_contract \
  model/nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12.hbm test_data/office_desk.jpg
```

This check passed on S100P with the n model. It covers prediction before
initialization, failed initialization and retry, repeated initialization,
box-local binary masks, and rejection of malformed output tensor shapes.

## Accuracy Status

The original five S100 Python/C++ runs returned matching detection counts and
labels; boxes/scores agreed within 1e-7 and mask areas matched. Area equality
does not establish pixel-wise mask identity.

S100P n-model Python/C++ regression was recorded on 2026-09-17 with
UCP 3.13.6 / HBRT 4.7.5. The released nash-m n HBM (SHA256
`58922c669bcccec7a00c11719db6c3b9c4eaa4e0388b58bd120fda914fe9d048`)
processed the bundled `office_desk.jpg`. Both zero-argument `main.py` and the
CMake-built C++ executable completed with 14 detections. Boxes, scores, classes
and every pixel of every box-local mask matched the baseline at commit
`855a2b176fd215e4e62e73372c85da33f4f076bc` exactly, in both languages and across
Python/C++. Baseline full-image masks were cropped with the same clipped,
integer-truncated box bounds before comparison. All four Python runtime
contract tests passed on both the development host and this S100P.
This evidence covers the n model and bundled image only. It does not verify
s/m/l/x, dataset mAP, or S100P Runtime performance.

Calibration used 100 representative COCO train2017 images. Dataset box/mask mAP
has not been accepted, and the INT8 baseline shows numerical differences from
float inference. Validate on a held-out labeled dataset before production use.
Map the PF vocabulary explicitly to dataset categories: PF IDs are not COCO IDs.
No invented mAP or S100P Runtime numbers are included.
