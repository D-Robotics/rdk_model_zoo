English | [简体中文](./README_cn.md)

# C++ UCP Inference

Build on S100/S100P with the installed UCP/DNN SDK, OpenCV development package,
CMake and a C++17 compiler.

```bash
# SIZE [IMAGE] [OUTPUT]; default SIZE=n, bundled image and result.jpg.
bash run.sh n
bash run.sh x /path/image.jpg result.jpg

# Direct build and invocation, after downloading the model:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/yoloe26seg /path/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm \
  /path/yoloe_26n_seg_pf.names /path/image.jpg result.jpg 0.25 300 0
```

The last three optional arguments are confidence, max_det and multi_label (0/1).
The canonical filename must match the board: `nashe` for S100 or `nashm` for
S100P. The download wrapper selects this automatically and verifies SHA256.

`inc/yoloe26seg.hpp` exposes `YoloE26Seg::predict` and result structures.
`src/yoloe26seg.cpp` owns UCP model/tensor resources, preprocessing and output
decoding. `src/main.cpp` only handles the image demo and visualization.

Outputs are read using their byte strides, dequantized using their actual
scale/zero-point metadata and decoded without DFL or NMS. Use one model instance
per inference thread. Resource cleanup also applies to exceptions.

The default build contains no internal benchmark executables or test fixtures.
Use `hrt_model_exec perf` for model Runtime measurements.
