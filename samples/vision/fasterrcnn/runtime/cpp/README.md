English | [简体中文](README_cn.md)

# Faster R-CNN C++ runtime

<a id="overview"></a>
## Overview

C++ COCO detection and CPU ByteTrack person tracking using the paired board models.

<a id="directory"></a>
## Directory structure

```text
src/main.cpp  # Construct model and call predict
src/cli.cpp   # Options and image/video IO
src/detect.cpp # Detector, board runtime and person tracker
inc/cli.hpp
inc/detect.hpp
CMakeLists.txt
run.sh
```

<a id="supported-boards"></a>
## Supported boards

| Target | Model | Python | C++ | Runtime SDK |
| --- | --- | --- | --- | --- |
| `x5` | MobileNetV3-Large-320-FPN, PartA + PartB `.bin` | ✓ | ✓ | DNN 1.24.5 |
| `s100` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |
| `s100p` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |
| `s600` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |
Use `.bin` on X5 and the target-specific `.hbm` pair on S boards.

<a id="dependencies"></a>
## Dependencies

CMake 3.16 or newer, a C++17 compiler, OpenCV development headers/libraries and the board DNN SDK. S boards also need `hbucp`. X5 uses `/usr/lib` for SDK libraries; S uses `/usr/hobot/lib`. CMake accepts `FRCNN_HOBOT_LIB_DIR`, `FRCNN_HOBOT_INCLUDE_DIR` and `FRCNN_SYSTEM_HOBOT_INCLUDE_DIR` to select an SDK location. The CPU tracker sources are in the ByteTrack Sample.

<a id="build"></a>
## Build on the board

From the repository root:
```bash
cd samples/vision/fasterrcnn/runtime/cpp
cmake -S . -B build/x5 -DFRCNN_TARGET=x5 -DCMAKE_BUILD_TYPE=Release
cmake --build build/x5 -j2
```
For S100/S100P/S600, replace `x5` in the build directory and `FRCNN_TARGET` with the chosen target. This selects the matching DNN/UCP backend.

<a id="run"></a>
## Run

Download the X5 pair from the repository root using Python 3.10 or newer. For another board, replace `x5` with `s100`, `s100p` or `s600`.
```bash
python3 -m venv .venv-model-download
. .venv-model-download/bin/activate
python3 -m pip install PyYAML==6.0.3
python3 samples/vision/fasterrcnn/model/download.py --target x5
```
Run on the board from `samples/vision/fasterrcnn/runtime/cpp`:
```bash
./build/x5/fasterrcnn
./build/x5/fasterrcnn --target x5 \
  --parta-path ../../model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path ../../model/x5/frcnn_mbv3_320_partb.bin \
  --input ../../test_data/bus.jpg --save-path result.jpg
./build/x5/fasterrcnn --target x5 \
  --parta-path ../../model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path ../../model/x5/frcnn_mbv3_320_partb.bin \
  --video ../../../bytetrack/test_data/track_test.mp4 \
  --save-path people.mp4 --records people.tsv
```
The default command uses the descriptor pair. Supply both paths for your own compiled models. The shell entry combines build and run: `bash run.sh --target x5 -- --list-models`.

<a id="parameters"></a>
## Options

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--target` | `string` | `auto` | `auto`, `x5`, `s100`, `s100p`, `s600` |
| `--parta-path` | `path` | `None` | PartA file |
| `--partb-path` | `path` | `None` | PartB file, together with PartA |
| `--input` | `path` | `samples/vision/fasterrcnn/test_data/bus.jpg` | Image, or video with `--track` |
| `-i` | `path` | `samples/vision/fasterrcnn/test_data/bus.jpg` | Alias for `--input` |
| `--video` | `path` | `None` | Video input; person tracking |
| `--track` | `bool` | false | Track input video |
| `--reset-every` | `int` | `None` | Positive frame interval |
| `--records` | `path` | `None` | Track TSV path |
| `--no-encode` | `bool` | false | Skip output video encoding |
| `--save-path` | `path` | `samples/vision/fasterrcnn/test_data/result.jpg` | Output path, `.mp4` for video |
| `--threads` | `int` | `None` | Image conversion threads: 1, 2, 4 |
| `--label-file` | `path` | `samples/vision/fasterrcnn/model/coco91_labels.txt` | COCO label file |
| `--list-models` | `bool` | false | List targets and local model paths |
| `--dry-run` | `bool` | false | Print selected model paths |

`None` means the option is omitted. Model paths are then selected from the target directory; OpenCV keeps its default thread count. Default file paths above are shown relative to the repository root. Use `--help` or `-h` for command help.

<a id="interface-lifecycle"></a>
## Use in an application

The public interface is in `inc/detect.hpp`. Link `fasterrcnn_core` and OpenCV. This example runs from the repository root:
```cpp
#include "detect.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <iostream>

int main() {
  fasterrcnn::FasterRCNN::Config cfg;
  cfg.target = "x5";
  cfg.parta_path = "samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_parta.bin";
  cfg.partb_path = "samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_partb.bin";
  cfg.layout = fasterrcnn::PartbLayout::kPooled;
  fasterrcnn::FasterRCNN model(cfg);
  cv::Mat image = cv::imread("samples/vision/fasterrcnn/test_data/bus.jpg");
  if (image.empty()) throw std::runtime_error("bus.jpg");
  if (!image.isContinuous()) image = image.clone();
  fasterrcnn::FasterRCNN::Input input;
  input.rows = image.rows;
  input.cols = image.cols;
  input.bgr.assign(image.data, image.data + image.total() * image.elemSize());
  const auto result = model.predict(input);
  std::cout << result.size() << " detections\n";
  return 0;
}
```
Use `preprocess(input)`, `infer(prepared)` and `postprocess(outputs, prepared.context)` for staged calls. Construction loads the model pair; destruction releases runtime resources. Keep the model alive while a `PersonTracker` uses it, and process frames in order. Create one model/tracker pair per independent stream.

<a id="results-interpretation"></a>
## Outputs

Image JSON contains original-pixel xyxy boxes, scores, COCO category IDs, artifact hashes and the image path. Video produces MP4 and optional TSV records with columns `frame, track_id, x1, y1, x2, y2, score`. The process returns 0 on completion and 2 for an input, model or runtime error. Video `predict` time includes detection and tracking; tracker time is the CPU association subset.

