/*
 * Copyright (c) 2024-2025, WuChao && MaChao D-Robotics.
 * Copyright (c) 2026, D-Robotics.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// YOLO detection sample for RDK X5 and RDK S100/S100P/S600. This program
// runs on-board. Two head contracts are supported and selected at load time
// from the model's own output shapes:
//   * YOLO26 direct-LTRB heads (4-channel box maps).
//   * DFL heads (YOLOv5u/v8/v9/yolo11/yolo12/yolov13, 64-channel box maps).
// Two input protocols are supported and likewise probed at load time:
// packed NV12 (X5 .bin) and split Y/UV NV12 (S-series .hbm).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

#include "common/benchmark.h"
#include "common/dnn_io.h"
#include "common/decode.h"
#include "common/tensor_view.h"

namespace {

const char* kDefaultModel = "yolo26n_detect_bayese_640x640_nv12.bin";
const char* kDefaultImage = "bus.jpg";
const char* kDefaultOutput = "cpp_result.jpg";

const int kClasses = 80;
const int kStrides[] = {8, 16, 32};
const int kOutputCount = 6;

const std::vector<std::string> kCocoNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

const std::vector<cv::Scalar> kColors = {
    cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255),
    cv::Scalar(29, 178, 255), cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72),
    cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61), cv::Scalar(52, 147, 26),
    cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0),
    cv::Scalar(255, 56, 132), cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203),
    cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)};

typedef std::chrono::steady_clock Clock;

enum class HeadMode { kAuto, kLtrb, kDfl };

struct Options {
  std::string model_path = kDefaultModel;
  std::string image_path = kDefaultImage;
  std::string output_path = kDefaultOutput;
  std::string json_path;
  HeadMode head_mode = HeadMode::kAuto;
  int resize_type = 1;
  int opencv_threads = 0;
  int pipeline_streams = 1;
  int warmup = 20;
  int runs = 200;
  int rounds = 3;
  float score_threshold = 0.25f;
  float nms_threshold = 0.7f;
  bool benchmark = false;
  bool save_result = true;
};

struct Detection {
  int class_id;
  float confidence;
  cv::Rect2d bbox;
};

bool check_ret(int ret, const std::string& action) {
  if (ret == 0) return true;
  std::cerr << "[ERROR] " << action << " failed, error code: " << ret << std::endl;
  return false;
}

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [model] [image] [result.jpg] [options]\n"
      << "Options:\n"
      << "  --benchmark          Run bounded end-to-end benchmark\n"
      << "  --warmup N           Warmup frames per round (default: 20)\n"
      << "  --runs N             Timed frames per round (default: 200)\n"
      << "  --rounds N           Benchmark rounds (default: 3)\n"
      << "  --json PATH          Write aggregate benchmark JSON\n"
      << "  --head auto|dfl|ltrb\n"
      << "                       Box decode contract (default: auto-detect\n"
      << "                       from output shapes; dfl covers\n"
      << "                       YOLOv5u/v8/v9/yolo11/yolo12/yolov13)\n"
      << "  --score VALUE        Score threshold (default: 0.25)\n"
      << "  --nms VALUE          NMS IoU threshold (default: 0.7)\n"
      << "  --resize-type 0|1    0=resize, 1=letterbox (default: 1)\n"
      << "  --opencv-threads N|all\n"
      << "                       OpenCV CPU threads (default: all online CPUs)\n"
      << "  --pipeline-streams N Complete concurrent E2E pipelines (default: 1)\n"
      << "  --no-save            Do not draw or save the validation result\n"
      << "  --help               Show this message\n";
}

int parse_positive(const std::string& value, const std::string& option) {
  const int parsed = std::stoi(value);
  if (parsed <= 0) throw std::runtime_error(option + " must be positive");
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    const auto next_value = [&](const std::string& option) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(option + " requires a value");
      return std::string(argv[++i]);
    };

    if (arg == "--benchmark") {
      options.benchmark = true;
    } else if (arg == "--no-save") {
      options.save_result = false;
    } else if (arg == "--warmup") {
      options.warmup = parse_positive(next_value(arg), arg);
    } else if (arg == "--runs") {
      options.runs = parse_positive(next_value(arg), arg);
    } else if (arg == "--rounds") {
      options.rounds = parse_positive(next_value(arg), arg);
    } else if (arg == "--json") {
      options.json_path = next_value(arg);
    } else if (arg == "--head") {
      const std::string value = next_value(arg);
      if (value == "auto") {
        options.head_mode = HeadMode::kAuto;
      } else if (value == "dfl") {
        options.head_mode = HeadMode::kDfl;
      } else if (value == "ltrb") {
        options.head_mode = HeadMode::kLtrb;
      } else {
        throw std::runtime_error("--head must be auto, dfl or ltrb");
      }
    } else if (arg == "--score") {
      options.score_threshold = std::stof(next_value(arg));
    } else if (arg == "--nms") {
      options.nms_threshold = std::stof(next_value(arg));
    } else if (arg == "--resize-type") {
      options.resize_type = std::stoi(next_value(arg));
    } else if (arg == "--opencv-threads") {
      const std::string value = next_value(arg);
      options.opencv_threads = value == "all" ? 0 : parse_positive(value, arg);
    } else if (arg == "--pipeline-streams") {
      options.pipeline_streams = parse_positive(next_value(arg), arg);
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.size() > 3) throw std::runtime_error("too many positional arguments");
  if (!positional.empty()) options.model_path = positional[0];
  if (positional.size() > 1) options.image_path = positional[1];
  if (positional.size() > 2) options.output_path = positional[2];

  if (options.resize_type != 0 && options.resize_type != 1) {
    throw std::runtime_error("--resize-type must be 0 or 1");
  }
  if (options.score_threshold < 0.0f || options.score_threshold > 1.0f) {
    throw std::runtime_error("--score must be in [0, 1]");
  }
  if (options.nms_threshold < 0.0f || options.nms_threshold > 1.0f) {
    throw std::runtime_error("--nms must be in [0, 1]");
  }
  return options;
}

cv::Mat preprocess_image(const cv::Mat& image, int input_h, int input_w,
                         int resize_type, yolo::ImageTransform* transform) {
  cv::Mat result;
  if (resize_type == 0) {
    cv::resize(image, result, cv::Size(input_w, input_h));
    transform->scale_x = static_cast<float>(input_w) / image.cols;
    transform->scale_y = static_cast<float>(input_h) / image.rows;
    return result;
  }

  const float scale = std::min(static_cast<float>(input_h) / image.rows,
                               static_cast<float>(input_w) / image.cols);
  const int resized_w = static_cast<int>(image.cols * scale);
  const int resized_h = static_cast<int>(image.rows * scale);
  transform->scale_x = scale;
  transform->scale_y = scale;
  transform->shift_x = (input_w - resized_w) / 2;
  transform->shift_y = (input_h - resized_h) / 2;
  const int right = input_w - resized_w - transform->shift_x;
  const int bottom = input_h - resized_h - transform->shift_y;

  cv::resize(image, result, cv::Size(resized_w, resized_h));
  cv::copyMakeBorder(result, result, transform->shift_y, bottom,
                     transform->shift_x, right, cv::BORDER_CONSTANT,
                     cv::Scalar(127, 127, 127));
  return result;
}

// One detection runtime context. Input protocol and head contract are both
// probed from the model itself.
class DetectRuntime {
 public:
  DetectRuntime() = default;
  ~DetectRuntime() { release(); }

  bool initialize(const std::string& model_path, HeadMode head_mode) {
    const char* model_file = model_path.c_str();
    if (!check_ret(hbDNNInitializeFromFiles(&packed_handle_, &model_file, 1),
                   "hbDNNInitializeFromFiles")) {
      return false;
    }

    const char** model_names = nullptr;
    int model_count = 0;
    if (!check_ret(hbDNNGetModelNameList(&model_names, &model_count, packed_handle_),
                   "hbDNNGetModelNameList") ||
        model_count <= 0) {
      return false;
    }
    model_name_ = model_names[0];
    if (!check_ret(hbDNNGetModelHandle(&model_handle_, packed_handle_, model_names[0]),
                   "hbDNNGetModelHandle")) {
      return false;
    }

    std::string protocol_error;
    input_plan_ = yolo::probe_input_protocol(model_handle_, &protocol_error);
    if (input_plan_.protocol == yolo::InputProtocol::kUnknown) {
      std::cerr << "[ERROR] Unsupported model input: " << protocol_error
                << std::endl;
      return false;
    }
    if (!input_.allocate(model_handle_, input_plan_)) return false;

    int32_t output_count = 0;
    if (!check_ret(hbDNNGetOutputCount(&output_count, model_handle_),
                   "hbDNNGetOutputCount") ||
        output_count != kOutputCount) {
      std::cerr << "[ERROR] Detect models require six outputs, got "
                << output_count << std::endl;
      return false;
    }

    outputs_.resize(output_count);
    output_allocated_.assign(output_count, false);
    std::vector<yolo::OutputShape> shapes(output_count);
    for (int i = 0; i < output_count; ++i) {
      std::memset(&outputs_[i], 0, sizeof(outputs_[i]));
      if (!check_ret(hbDNNGetOutputTensorProperties(&outputs_[i].properties,
                                                     model_handle_, i),
                     "hbDNNGetOutputTensorProperties")) {
        return false;
      }
      const hbDNNTensorProperties& properties = outputs_[i].properties;
      if (properties.tensorLayout != HB_DNN_LAYOUT_NHWC ||
          properties.tensorType != HB_DNN_TENSOR_TYPE_F32 ||
          properties.quantiType != NONE ||
          properties.validShape.numDimensions != 4) {
        std::cerr << "[ERROR] output[" << i
                  << "] must be unquantized FLOAT32 NHWC" << std::endl;
        return false;
      }
      if (!check_ret(hbSysAllocCachedMem(&outputs_[i].sysMem[0],
                                         properties.alignedByteSize),
                     "hbSysAllocCachedMem(output)")) {
        return false;
      }
      output_allocated_[i] = true;

      const hbDNNTensorShape& valid = properties.validShape;
      const hbDNNTensorShape& aligned = properties.alignedShape;
      shapes[i].h = valid.dimensionSize[1];
      shapes[i].w = valid.dimensionSize[2];
      shapes[i].c = valid.dimensionSize[3];
      std::cout << "[INFO] output[" << i << "] valid=(" << valid.dimensionSize[0]
                << ", " << valid.dimensionSize[1] << ", " << valid.dimensionSize[2]
                << ", " << valid.dimensionSize[3] << ") aligned=("
                << aligned.dimensionSize[0] << ", " << aligned.dimensionSize[1]
                << ", " << aligned.dimensionSize[2] << ", "
                << aligned.dimensionSize[3] << ")" << std::endl;
    }

    if (!bind_outputs(shapes, head_mode)) return false;

    std::cout << "[INFO] Model name: " << model_name_ << std::endl;
    std::cout << "[INFO] Input: "
              << (input_plan_.protocol == yolo::InputProtocol::kPackedNv12
                      ? "packed NV12 "
                      : "split Y/UV NV12 ")
              << input_plan_.input_w << "x" << input_plan_.input_h << std::endl;
    std::cout << "[INFO] Head: "
              << (head_ == yolo::BoxDecode::kDirectLtrb ? "direct-LTRB" : "DFL")
              << std::endl;
    std::cout << "[INFO] Output order: [";
    for (int i = 0; i < kOutputCount; ++i) {
      if (i) std::cout << ", ";
      std::cout << output_order_[i];
    }
    std::cout << "]" << std::endl;
    return true;
  }

  int input_h() const { return input_plan_.input_h; }
  int input_w() const { return input_plan_.input_w; }
  yolo::BoxDecode head() const { return head_; }

  bool upload(const cv::Mat& i420) {
    if (i420.empty() || !i420.isContinuous() ||
        static_cast<int>(i420.total()) !=
            input_plan_.input_h * input_plan_.input_w * 3 / 2) {
      std::cerr << "[ERROR] Invalid I420 input" << std::endl;
      return false;
    }
    return input_.upload(input_plan_, i420.ptr<uint8_t>());
  }

  bool infer() {
    hbDNNTaskHandle_t task = nullptr;
    hbDNNInferCtrlParam control;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
    hbDNNTensor* output_ptr = outputs_.data();

    int ret = hbDNNInfer(&task, &output_ptr, input_.tensors(), model_handle_,
                         &control);
    if (ret != 0) return check_ret(ret, "hbDNNInfer");
    ret = hbDNNWaitTaskDone(task, 0);
    const int release_ret = hbDNNReleaseTask(task);
    if (ret != 0) return check_ret(ret, "hbDNNWaitTaskDone");
    return check_ret(release_ret, "hbDNNReleaseTask");
  }

  bool postprocess(const yolo::ImageTransform& transform, int image_w, int image_h,
                   float score_threshold, float nms_threshold,
                   std::vector<Detection>* detections) {
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (!check_ret(hbSysFlushMem(&outputs_[i].sysMem[0],
                                   HB_SYS_MEM_CACHE_INVALIDATE),
                     "hbSysFlushMem(output invalidate)")) {
        return false;
      }
    }

    const float raw_threshold = yolo::raw_logit_threshold(score_threshold);
    std::vector<std::vector<cv::Rect2d> > boxes(kClasses);
    std::vector<std::vector<float> > scores(kClasses);

    for (int scale = 0; scale < 3; ++scale) {
      const int stride = kStrides[scale];
      const yolo::TensorView cls_view = output_view(output_order_[scale * 2]);
      const yolo::TensorView box_view = output_view(output_order_[scale * 2 + 1]);

      for (int y = 0; y < cls_view.h; ++y) {
        for (int x = 0; x < cls_view.w; ++x) {
          const float* logits = cls_view.cell(y, x);
          int class_id = 0;
          for (int c = 1; c < kClasses; ++c) {
            if (logits[c] > logits[class_id]) class_id = c;
          }
          if (logits[class_id] < raw_threshold) continue;

          float ltrb[4];
          const float* box_raw = box_view.cell(y, x);
          if (head_ == yolo::BoxDecode::kDirectLtrb) {
            yolo::decode_box_ltrb(box_raw, ltrb);
          } else {
            yolo::decode_box_dfl(box_raw, ltrb);
          }

          float x1, y1, x2, y2;
          yolo::box_from_distances(x + 0.5f, y + 0.5f, ltrb,
                                   static_cast<float>(stride), &x1, &y1, &x2,
                                   &y2);
          if (!yolo::map_to_source(&x1, &y1, &x2, &y2, transform, image_w,
                                   image_h)) {
            continue;
          }

          boxes[class_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
          scores[class_id].push_back(yolo::sigmoid(logits[class_id]));
        }
      }
    }

    detections->clear();
    for (int class_id = 0; class_id < kClasses; ++class_id) {
      if (boxes[class_id].empty()) continue;
      std::vector<int> keep;
      cv::dnn::NMSBoxes(boxes[class_id], scores[class_id], score_threshold,
                        nms_threshold, keep, 1.0f, 0);
      for (size_t i = 0; i < keep.size(); ++i) {
        const int index = keep[i];
        detections->push_back(
            Detection{class_id, scores[class_id][index], boxes[class_id][index]});
      }
    }
    return true;
  }

 private:
  yolo::TensorView output_view(int index) const {
    const hbDNNTensor& tensor = outputs_[index];
    const hbDNNTensorShape& valid = tensor.properties.validShape;
    const hbDNNTensorShape& aligned = tensor.properties.alignedShape;
    yolo::TensorView view;
    view.data = reinterpret_cast<const float*>(tensor.sysMem[0].virAddr);
    view.h = valid.dimensionSize[1];
    view.w = valid.dimensionSize[2];
    view.channels = valid.dimensionSize[3];
    view.aligned_w = aligned.numDimensions == 4 ? aligned.dimensionSize[2]
                                                : view.w;
    view.aligned_c = aligned.numDimensions == 4 ? aligned.dimensionSize[3]
                                                : view.channels;
    return view;
  }

  bool bind_outputs(const std::vector<yolo::OutputShape>& shapes,
                    HeadMode head_mode) {
    // Try the requested protocol first, then fall back for auto mode.
    std::vector<yolo::BoxDecode> candidates;
    if (head_mode == HeadMode::kLtrb) {
      candidates.push_back(yolo::BoxDecode::kDirectLtrb);
    } else if (head_mode == HeadMode::kDfl) {
      candidates.push_back(yolo::BoxDecode::kDfl);
    } else {
      candidates.push_back(yolo::BoxDecode::kDirectLtrb);
      candidates.push_back(yolo::BoxDecode::kDfl);
    }

    for (size_t candidate = 0; candidate < candidates.size(); ++candidate) {
      const int box_channels = yolo::box_channels(candidates[candidate]);
      bool complete = true;
      for (int scale = 0; scale < 3 && complete; ++scale) {
        const int h = input_plan_.input_h / kStrides[scale];
        const int w = input_plan_.input_w / kStrides[scale];
        const int cls_index = yolo::find_output_by_shape(shapes, h, w, kClasses);
        const int box_index = yolo::find_output_by_shape(shapes, h, w, box_channels);
        if (cls_index < 0 || box_index < 0) {
          complete = false;
          break;
        }
        output_order_[scale * 2] = cls_index;
        output_order_[scale * 2 + 1] = box_index;
      }
      if (complete) {
        head_ = candidates[candidate];
        return true;
      }
    }

    std::cerr << "[ERROR] Missing detect outputs: expected per stride a "
              << kClasses << "-channel class map plus a 4-channel (YOLO26 "
              << "direct-LTRB) or 64-channel (DFL) box map" << std::endl;
    return false;
  }

  void release() {
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (i < output_allocated_.size() && output_allocated_[i]) {
        const int ret = hbSysFreeMem(&outputs_[i].sysMem[0]);
        if (ret != 0) {
          std::cerr << "[WARN] hbSysFreeMem(output) returned " << ret << std::endl;
        }
        output_allocated_[i] = false;
      }
    }
    if (packed_handle_ != nullptr) {
      const int ret = hbDNNRelease(packed_handle_);
      if (ret != 0) std::cerr << "[WARN] hbDNNRelease returned " << ret << std::endl;
      packed_handle_ = nullptr;
      model_handle_ = nullptr;
    }
  }

  hbPackedDNNHandle_t packed_handle_ = nullptr;
  hbDNNHandle_t model_handle_ = nullptr;
  yolo::InputPlan input_plan_;
  yolo::Nv12Input input_;
  std::vector<hbDNNTensor> outputs_;
  std::vector<bool> output_allocated_;
  int output_order_[kOutputCount] = {-1, -1, -1, -1, -1, -1};
  yolo::BoxDecode head_ = yolo::BoxDecode::kUnknown;
  std::string model_name_;
};

bool run_pipeline(DetectRuntime* runtime, const cv::Mat& image,
                  const Options& options, std::vector<Detection>* detections,
                  yolo::StageTiming* timing) {
  const Clock::time_point start = Clock::now();
  yolo::ImageTransform transform;
  const cv::Mat resized = preprocess_image(image, runtime->input_h(), runtime->input_w(),
                                           options.resize_type, &transform);
  cv::Mat i420;
  cv::cvtColor(resized, i420, cv::COLOR_BGR2YUV_I420);
  if (!runtime->upload(i420)) return false;
  const Clock::time_point preprocessed = Clock::now();

  if (!runtime->infer()) return false;
  const Clock::time_point inferred = Clock::now();

  if (!runtime->postprocess(transform, image.cols, image.rows,
                            options.score_threshold, options.nms_threshold,
                            detections)) {
    return false;
  }
  const Clock::time_point finished = Clock::now();

  if (timing != nullptr) {
    const auto ms = [](const Clock::time_point& a, const Clock::time_point& b) {
      return std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(
                 b - a)
          .count();
    };
    timing->preprocess_ms = ms(start, preprocessed);
    timing->runtime_ms = ms(preprocessed, inferred);
    timing->postprocess_ms = ms(inferred, finished);
    timing->end_to_end_ms = ms(start, finished);
  }
  return true;
}

bool run_pipeline_streams(const std::vector<DetectRuntime*>& runtimes,
                          const cv::Mat& image, const Options& options,
                          int frames_per_stream, size_t expected_detections,
                          bool collect_timing, yolo::BenchmarkRound* result) {
  if (runtimes.empty()) return false;

  std::vector<yolo::StageSamples> stream_samples(runtimes.size());
  std::vector<std::thread> workers;
  workers.reserve(runtimes.size());
  std::atomic<bool> failed(false);
  std::mutex start_mutex;
  std::mutex error_mutex;
  std::condition_variable ready_condition;
  std::condition_variable start_condition;
  size_t ready_workers = 0;
  bool start = false;
  std::string error_message;

  const auto fail = [&](const std::string& message) {
    if (!failed.exchange(true)) {
      std::lock_guard<std::mutex> lock(error_mutex);
      error_message = message;
    }
  };

  for (size_t stream = 0; stream < runtimes.size(); ++stream) {
    workers.push_back(std::thread([&, stream]() {
      {
        std::unique_lock<std::mutex> lock(start_mutex);
        ++ready_workers;
        ready_condition.notify_one();
        start_condition.wait(lock, [&]() { return start; });
      }

      try {
        std::vector<Detection> detections;
        for (int frame = 0; frame < frames_per_stream && !failed.load(); ++frame) {
          yolo::StageTiming timing;
          if (!run_pipeline(runtimes[stream], image, options, &detections,
                            collect_timing ? &timing : nullptr)) {
            fail("pipeline stream " + std::to_string(stream) + " failed");
            break;
          }
          if (detections.size() != expected_detections) {
            fail("pipeline stream " + std::to_string(stream) +
                 " detection count changed: expected " +
                 std::to_string(expected_detections) + ", got " +
                 std::to_string(detections.size()));
            break;
          }
          if (collect_timing) stream_samples[stream].add(timing);
        }
      } catch (const std::exception& error) {
        fail("pipeline stream " + std::to_string(stream) +
             " raised: " + error.what());
      } catch (...) {
        fail("pipeline stream " + std::to_string(stream) +
             " raised an unknown exception");
      }
    }));
  }

  Clock::time_point wall_start;
  {
    std::unique_lock<std::mutex> lock(start_mutex);
    ready_condition.wait(lock, [&]() { return ready_workers == runtimes.size(); });
    wall_start = Clock::now();
    start = true;
  }
  start_condition.notify_all();
  for (size_t i = 0; i < workers.size(); ++i) workers[i].join();
  const Clock::time_point wall_end = Clock::now();

  if (failed.load()) {
    std::lock_guard<std::mutex> lock(error_mutex);
    std::cerr << "[ERROR] " << error_message << std::endl;
    return false;
  }
  if (collect_timing && result != nullptr) {
    result->wall_ms = std::chrono::duration_cast<
        std::chrono::duration<double, std::milli> >(wall_end - wall_start)
        .count();
    for (size_t stream = 0; stream < stream_samples.size(); ++stream) {
      result->samples.append(stream_samples[stream]);
    }
    result->completed_frames = result->samples.end_to_end.size();
  }
  return true;
}

void draw_detections(cv::Mat* image, const std::vector<Detection>& detections) {
  for (size_t i = 0; i < detections.size(); ++i) {
    const Detection& detection = detections[i];
    const cv::Scalar color = kColors[detection.class_id % kColors.size()];
    cv::rectangle(*image, detection.bbox, color, 2);
    const std::string label =
        kCocoNames[detection.class_id] + " " +
        std::to_string(static_cast<int>(detection.confidence * 100.0f)) + "%";
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    const int x = static_cast<int>(detection.bbox.x);
    const int y = std::max(static_cast<int>(detection.bbox.y), text_size.height + 8);
    cv::rectangle(*image, cv::Point(x, y - text_size.height - 8),
                  cv::Point(x + text_size.width, y), color, cv::FILLED);
    cv::putText(*image, label, cv::Point(x, y - 4), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  }
}

void print_statistics(const std::string& label, const std::vector<double>& values) {
  const yolo::Statistics stats = yolo::summarize(values);
  std::cout << std::left << std::setw(13) << label << std::right << std::fixed
            << std::setprecision(3) << " mean=" << std::setw(8) << stats.mean
            << " ms  p50=" << std::setw(8) << stats.p50
            << "  p95=" << std::setw(8) << stats.p95
            << "  min=" << std::setw(8) << stats.min
            << "  max=" << std::setw(8) << stats.max << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const int online_cpu_threads = std::max(1, cv::getNumberOfCPUs());
    const int opencv_threads =
        options.opencv_threads == 0 ? online_cpu_threads : options.opencv_threads;
    cv::setUseOptimized(true);
    cv::setNumThreads(opencv_threads);

    std::cout << "[INFO] YOLO detect C++ sample (direct-LTRB + DFL heads)"
              << std::endl;
    std::cout << "[INFO] OpenCV: " << CV_VERSION
              << ", online CPUs: " << online_cpu_threads
              << ", CPU thread policy: "
              << (options.opencv_threads == 0 ? "all-online" : "fixed")
              << ", OpenCV threads: " << cv::getNumThreads() << std::endl;
    std::cout << "[INFO] Loading model: " << options.model_path << std::endl;
    std::cout << "[INFO] Pipeline streams: " << options.pipeline_streams << std::endl;

    const Clock::time_point load_start = Clock::now();
    std::vector<std::unique_ptr<DetectRuntime> > runtime_storage;
    std::vector<DetectRuntime*> runtimes;
    runtime_storage.reserve(options.pipeline_streams);
    runtimes.reserve(options.pipeline_streams);
    for (int stream = 0; stream < options.pipeline_streams; ++stream) {
      std::unique_ptr<DetectRuntime> runtime(new DetectRuntime());
      if (!runtime->initialize(options.model_path, options.head_mode)) return 1;
      if (!runtimes.empty() &&
          (runtime->input_h() != runtimes[0]->input_h() ||
           runtime->input_w() != runtimes[0]->input_w() ||
           runtime->head() != runtimes[0]->head())) {
        std::cerr << "[ERROR] Runtime contexts expose different contracts"
                  << std::endl;
        return 1;
      }
      runtimes.push_back(runtime.get());
      runtime_storage.push_back(std::move(runtime));
    }
    std::cout << "[INFO] Runtime context load: " << std::fixed
              << std::setprecision(3)
              << std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(
                     Clock::now() - load_start)
                     .count()
              << " ms" << std::endl;

    const cv::Mat image = cv::imread(options.image_path);
    if (image.empty()) {
      std::cerr << "[ERROR] Cannot read image: " << options.image_path << std::endl;
      return 1;
    }
    std::cout << "[INFO] Image: " << image.cols << "x" << image.rows << std::endl;

    std::vector<Detection> detections;
    yolo::StageTiming validation_timing;
    if (!run_pipeline(runtimes[0], image, options, &detections,
                      &validation_timing)) {
      return 1;
    }
    std::cout << "[INFO] Validation detections: " << detections.size() << std::endl;
    std::cout << "[INFO] Validation timing: preprocess=" << validation_timing.preprocess_ms
              << " ms, runtime=" << validation_timing.runtime_ms
              << " ms, postprocess=" << validation_timing.postprocess_ms
              << " ms, end_to_end=" << validation_timing.end_to_end_ms << " ms"
              << std::endl;

    if (options.save_result) {
      cv::Mat rendered = image.clone();
      draw_detections(&rendered, detections);
      if (!cv::imwrite(options.output_path, rendered)) {
        std::cerr << "[ERROR] Cannot write result: " << options.output_path << std::endl;
        return 1;
      }
      std::cout << "[INFO] Result saved: " << options.output_path << std::endl;
    }

    if (!options.benchmark) return 0;

    yolo::StageSamples aggregate;
    double aggregate_wall_ms = 0.0;
    size_t completed_frames = 0;
    const size_t expected_detections = detections.size();
    for (int round = 0; round < options.rounds; ++round) {
      if (!run_pipeline_streams(runtimes, image, options, options.warmup,
                                expected_detections, false, nullptr)) {
        return 1;
      }

      yolo::BenchmarkRound current;
      if (!run_pipeline_streams(runtimes, image, options, options.runs,
                                expected_detections, true, &current)) {
        return 1;
      }
      aggregate.append(current.samples);
      aggregate_wall_ms += current.wall_ms;
      completed_frames += current.completed_frames;

      std::cout << "\n[ROUND " << round + 1 << "/" << options.rounds << "]" << std::endl;
      print_statistics("preprocess", current.samples.preprocess);
      print_statistics("runtime", current.samples.runtime);
      print_statistics("postprocess", current.samples.postprocess);
      print_statistics("end_to_end", current.samples.end_to_end);
      std::cout << "wall_time     " << std::fixed << std::setprecision(3)
                << current.wall_ms << " ms for " << current.completed_frames
                << " frames" << std::endl;
      std::cout << "throughput    " << std::fixed << std::setprecision(3)
                << (current.wall_ms > 0.0
                        ? current.completed_frames * 1000.0 / current.wall_ms
                        : 0.0)
                << " aggregate fps" << std::endl;
    }

    std::cout << "\n[AGGREGATE " << aggregate.end_to_end.size() << " frames]" << std::endl;
    print_statistics("preprocess", aggregate.preprocess);
    print_statistics("runtime", aggregate.runtime);
    print_statistics("postprocess", aggregate.postprocess);
    print_statistics("end_to_end", aggregate.end_to_end);
    std::cout << "wall_time     " << std::fixed << std::setprecision(3)
              << aggregate_wall_ms << " ms for " << completed_frames
              << " frames" << std::endl;
    std::cout << "throughput    " << std::fixed << std::setprecision(3)
              << (aggregate_wall_ms > 0.0
                      ? completed_frames * 1000.0 / aggregate_wall_ms
                      : 0.0)
              << " aggregate fps" << std::endl;

    if (!options.json_path.empty()) {
      yolo::BenchmarkMeta meta;
      meta.model_path = options.model_path;
      meta.image_path = options.image_path;
      meta.implementation = runtimes[0]->head() == yolo::BoxDecode::kDirectLtrb
                                ? "native_cpp_yolo26_ltrb"
                                : "native_cpp_yolo_dfl";
      meta.timing_scope = "in_memory_bgr_to_detections";
      meta.pipeline_streams = options.pipeline_streams;
      meta.cpu_thread_policy =
          options.opencv_threads == 0 ? "all_online" : "fixed";
      meta.online_cpu_threads = online_cpu_threads;
      meta.opencv_threads = cv::getNumThreads();
      meta.warmup_frames_per_round = options.warmup;
      meta.runs_per_round = options.runs;
      meta.rounds = options.rounds;
      meta.score_threshold = options.score_threshold;
      meta.nms_threshold = options.nms_threshold;
      if (!yolo::write_benchmark_json(options.json_path, meta, aggregate,
                                      expected_detections, completed_frames,
                                      aggregate_wall_ms)) {
        return 1;
      }
      std::cout << "[INFO] Benchmark JSON: " << options.json_path << std::endl;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    print_usage(argv[0]);
    return 1;
  }
}
