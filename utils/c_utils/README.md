# C++ runtime utilities

[简体中文](README_cn.md)

These helpers provide image and tensor processing, result rendering, board identification and file hashing for native Model Zoo samples.

## Directory structure

```text
c_utils/
├── inc/                   # Public headers
├── src/                   # Implementations
├── platform_identity.h    # Board identity types and target resolution
├── platform_identity.cc   # Read board identity
└── sha256.h               # SHA-256 helper
```

| Header / source | Purpose |
| --- | --- |
| `file_io.hpp` / `.cpp` | Images and labels |
| `model_types.hpp` | Classification, detection and keypoint types |
| `nn_math.hpp` / `.cpp` | Sigmoid, Softmax and normalization |
| `preprocess.hpp` / `.cpp` | Resize, letterbox, color conversion and tensor preparation |
| `postprocess.hpp` / `.cpp` | Top-K, dequantization, decoding, NMS and coordinate mapping |
| `visualize.hpp` / `.cpp` | Classification, boxes, masks, keypoints and text |
| `runtime.hpp` | HB-DNN and HB-UCP return-code handling |

## Usage

Build from a sample's `runtime/cpp` directory with its documented CMake command. Its CMakeLists selects the required headers and implementation files from this directory and links the target SDK and OpenCV libraries. See the [ResNet C++ guide](../../samples/vision/resnet/runtime/cpp/README.md) and [YOLO C++ guide](../../samples/vision/ultralytics_yolo/runtime/cpp/README.md) for target-specific configuration.

Include headers from `inc/` for image and result processing. Include `platform_identity.h` and compile `platform_identity.cc` when the application needs `rdk::read_native_identity()` and `rdk::identify_target()`. Include `sha256.h` for file hashes. Function signatures and tensor descriptions are documented in the [source reference](../../docs/source_reference/README.md).
