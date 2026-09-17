English | [简体中文](./README_cn.md)

# YOLOE-26 Segmentation C++ Runtime

This sample runs the released YOLOE-26 prompt-free instance-segmentation HBM
on S100 or S100P with UCP, OpenCV, CMake, C++17, and gflags.

Install the board development packages before building:

```bash
sudo apt install -y libgflags-dev libopencv-dev
```

From this `runtime/cpp/` directory, download the n model and run the complete
example:

```bash
cd samples/vision/yoloe26_seg/runtime/cpp
bash ../../model/download_model.sh auto n
bash run.sh
```

`run.sh` resolves its own source directory, so it can also be invoked by its
absolute path from another working directory. It downloads and verifies the
matching board model, builds the executable, and passes named flags. Optional
arguments are `SIZE`, `IMAGE`, and `OUTPUT`:

```bash
bash run.sh x /path/to/image.jpg /path/to/result.jpg
```

To build directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

The executable uses gflags and is zero-argument runnable when the downloaded n
model and the bundled 4585-class label file and test image exist at their
repository paths. Override paths and inference options with snake_case flags:

```bash
./build/yoloe26seg \
  --model_path=/path/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm \
  --model_size=n \
  --label_file=/path/yoloe_26n_seg_pf.names \
  --test_img=/path/image.jpg \
  --output_path=result.jpg \
  --score_thres=0.25 --max_det=300 --multi_label=false
```

The model file must use the canonical board suffix: `nashe` for S100 and
`nashm` for S100P. The runtime validates the suffix against
`/sys/class/boardinfo`.

`inc/yoloe26seg.hpp` exposes `YoloE26SegConfig`, the lightweight
`YoloE26Seg` owner, and the staged `pre_process`, `infer`, and `post_process`
functions. `init()` returns zero on success and contains initialization errors;
the destructor releases partially allocated resources safely.

`post_process()` returns the common `InstanceSegResult` from
`utils/c_utils/inc/model_types.hpp`. Boxes remain floating-point xyxy values in
source-image coordinates. Each mask is a `CV_8UC1` matrix containing only 0 or
1 and is local to the clipped, integer-truncated box. Empty boxes keep an empty
mask so detections and masks remain index-aligned. The demo renders these local
masks inside their clipped boxes and draws the corresponding boxes directly;
this keeps the sample independent of the optional hardware-display parts of the
shared visualization implementation.

The raw-v1 decoder keeps the ten-output order, tensor byte strides,
quantization scale/zero-point metadata, and deterministic static top-K behavior
of the released protocol. It does not apply IoU NMS.
