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

// YOLO26 direct-LTRB detection sample for RDK X5. This program runs on-board.

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

struct Options {
  std::string model_path = kDefaultModel;
  std::string image_path = kDefaultImage;
  std::string output_path = kDefaultOutput;
  std::string json_path;
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

struct ImageTransform {
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  int shift_x = 0;
  int shift_y = 0;
};

struct Detection {
  int class_id;
  float confidence;
  cv::Rect2d bbox;
};

struct StageTiming {
  double preprocess_ms = 0.0;
  double runtime_ms = 0.0;
  double postprocess_ms = 0.0;
  double end_to_end_ms = 0.0;
};

struct StageSamples {
  std::vector<double> preprocess;
  std::vector<double> runtime;
  std::vector<double> postprocess;
  std::vector<double> end_to_end;

  void add(const StageTiming& timing) {
    preprocess.push_back(timing.preprocess_ms);
    runtime.push_back(timing.runtime_ms);
    postprocess.push_back(timing.postprocess_ms);
    end_to_end.push_back(timing.end_to_end_ms);
  }

  void append(const StageSamples& other) {
    preprocess.insert(preprocess.end(), other.preprocess.begin(), other.preprocess.end());
    runtime.insert(runtime.end(), other.runtime.begin(), other.runtime.end());
    postprocess.insert(postprocess.end(), other.postprocess.begin(), other.postprocess.end());
    end_to_end.insert(end_to_end.end(), other.end_to_end.begin(), other.end_to_end.end());
  }
};

struct BenchmarkRound {
  StageSamples samples;
  double wall_ms = 0.0;
  size_t completed_frames = 0;
};

struct Statistics {
  double mean = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double min = 0.0;
  double max = 0.0;
};

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(
             end - start)
      .count();
}

bool check_ret(int ret, const std::string& action) {
  if (ret == 0) return true;
  std::cerr << "[ERROR] " << action << " failed, error code: " << ret << std::endl;
  return false;
}

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [model.bin] [image] [result.jpg] [options]\n"
      << "Options:\n"
      << "  --benchmark          Run bounded end-to-end benchmark\n"
      << "  --warmup N           Warmup frames per round (default: 20)\n"
      << "  --runs N             Timed frames per round (default: 200)\n"
      << "  --rounds N           Benchmark rounds (default: 3)\n"
      << "  --json PATH          Write aggregate benchmark JSON\n"
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
                         int resize_type, ImageTransform* transform) {
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

cv::Mat bgr_to_nv12(const cv::Mat& bgr) {
  if ((bgr.cols & 1) != 0 || (bgr.rows & 1) != 0) {
    throw std::runtime_error("NV12 input dimensions must be even");
  }
  cv::Mat i420;
  cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);

  const int y_size = bgr.rows * bgr.cols;
  const int uv_plane_size = y_size / 4;
  const uint8_t* source = i420.ptr<uint8_t>();
  const uint8_t* u = source + y_size;
  const uint8_t* v = u + uv_plane_size;

  cv::Mat nv12(bgr.rows * 3 / 2, bgr.cols, CV_8UC1);
  uint8_t* target = nv12.ptr<uint8_t>();
  std::memcpy(target, source, y_size);
  uint8_t* uv = target + y_size;
  for (int i = 0; i < uv_plane_size; ++i) {
    uv[2 * i] = u[i];
    uv[2 * i + 1] = v[i];
  }
  return nv12;
}

float sigmoid(float value) {
  if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
  const float exp_value = std::exp(value);
  return exp_value / (1.0f + exp_value);
}

class Yolo26Runtime {
 public:
  Yolo26Runtime() = default;
  ~Yolo26Runtime() { release(); }

  bool initialize(const std::string& model_path) {
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

    int32_t input_count = 0;
    if (!check_ret(hbDNNGetInputCount(&input_count, model_handle_),
                   "hbDNNGetInputCount") ||
        input_count != 1) {
      std::cerr << "[ERROR] YOLO26 sample requires exactly one input, got "
                << input_count << std::endl;
      return false;
    }
    if (!check_ret(hbDNNGetInputTensorProperties(&input_properties_, model_handle_, 0),
                   "hbDNNGetInputTensorProperties")) {
      return false;
    }
    if (input_properties_.tensorType != HB_DNN_IMG_TYPE_NV12 ||
        input_properties_.tensorLayout != HB_DNN_LAYOUT_NCHW ||
        input_properties_.validShape.numDimensions != 4) {
      std::cerr << "[ERROR] Expected one NCHW NV12 input" << std::endl;
      return false;
    }
    input_h_ = input_properties_.validShape.dimensionSize[2];
    input_w_ = input_properties_.validShape.dimensionSize[3];
    if ((input_h_ & 1) != 0 || (input_w_ & 1) != 0) {
      std::cerr << "[ERROR] NV12 input shape must be even" << std::endl;
      return false;
    }

    int32_t output_count = 0;
    if (!check_ret(hbDNNGetOutputCount(&output_count, model_handle_),
                   "hbDNNGetOutputCount") ||
        output_count != kOutputCount) {
      std::cerr << "[ERROR] YOLO26 detect requires six outputs, got "
                << output_count << std::endl;
      return false;
    }

    std::memset(&input_, 0, sizeof(input_));
    input_.properties = input_properties_;
    const int valid_input_bytes = input_h_ * input_w_ * 3 / 2;
    input_bytes_ = input_properties_.alignedByteSize;
    if (input_bytes_ < valid_input_bytes) input_bytes_ = valid_input_bytes;
    if (!check_ret(hbSysAllocCachedMem(&input_.sysMem[0], input_bytes_),
                   "hbSysAllocCachedMem(input)")) {
      return false;
    }
    input_allocated_ = true;
    std::memset(input_.sysMem[0].virAddr, 0, input_bytes_);

    outputs_.resize(output_count);
    output_allocated_.assign(output_count, false);
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
      std::cout << "[INFO] output[" << i << "] valid=(" << valid.dimensionSize[0]
                << ", " << valid.dimensionSize[1] << ", " << valid.dimensionSize[2]
                << ", " << valid.dimensionSize[3] << ") aligned=("
                << aligned.dimensionSize[0] << ", " << aligned.dimensionSize[1]
                << ", " << aligned.dimensionSize[2] << ", "
                << aligned.dimensionSize[3] << ")" << std::endl;
    }

    for (int scale = 0; scale < 3; ++scale) {
      const int h = input_h_ / kStrides[scale];
      const int w = input_w_ / kStrides[scale];
      output_order_[scale * 2] = find_output(h, w, kClasses);
      output_order_[scale * 2 + 1] = find_output(h, w, 4);
      if (output_order_[scale * 2] < 0 || output_order_[scale * 2 + 1] < 0) {
        std::cerr << "[ERROR] Missing YOLO26 direct-LTRB outputs for stride "
                  << kStrides[scale] << std::endl;
        return false;
      }
    }

    std::cout << "[INFO] Model name: " << model_name_ << std::endl;
    std::cout << "[INFO] Input: NV12 " << input_w_ << "x" << input_h_ << std::endl;
    std::cout << "[INFO] Output order: [";
    for (int i = 0; i < kOutputCount; ++i) {
      if (i) std::cout << ", ";
      std::cout << output_order_[i];
    }
    std::cout << "]" << std::endl;
    return true;
  }

  int input_h() const { return input_h_; }
  int input_w() const { return input_w_; }

  bool copy_input(const cv::Mat& nv12) {
    const int valid_bytes = input_h_ * input_w_ * 3 / 2;
    if (nv12.empty() || !nv12.isContinuous() ||
        nv12.total() != static_cast<size_t>(valid_bytes)) {
      std::cerr << "[ERROR] Invalid packed NV12 input" << std::endl;
      return false;
    }
    std::memcpy(input_.sysMem[0].virAddr, nv12.ptr<uint8_t>(), valid_bytes);
    return check_ret(hbSysFlushMem(&input_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
                     "hbSysFlushMem(input clean)");
  }

  bool infer() {
    hbDNNTaskHandle_t task = nullptr;
    hbDNNInferCtrlParam control;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
    hbDNNTensor* output_ptr = outputs_.data();

    int ret = hbDNNInfer(&task, &output_ptr, &input_, model_handle_, &control);
    if (ret != 0) return check_ret(ret, "hbDNNInfer");
    ret = hbDNNWaitTaskDone(task, 0);
    const int release_ret = hbDNNReleaseTask(task);
    if (ret != 0) return check_ret(ret, "hbDNNWaitTaskDone");
    return check_ret(release_ret, "hbDNNReleaseTask");
  }

  bool postprocess(const ImageTransform& transform, int image_w, int image_h,
                   float score_threshold, float nms_threshold,
                   std::vector<Detection>* detections) {
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (!check_ret(hbSysFlushMem(&outputs_[i].sysMem[0],
                                   HB_SYS_MEM_CACHE_INVALIDATE),
                     "hbSysFlushMem(output invalidate)")) {
        return false;
      }
    }

    const float raw_threshold =
        score_threshold <= 0.0f
            ? -std::numeric_limits<float>::infinity()
            : (score_threshold >= 1.0f
                   ? std::numeric_limits<float>::infinity()
                   : -std::log(1.0f / score_threshold - 1.0f));
    std::vector<std::vector<cv::Rect2d> > boxes(kClasses);
    std::vector<std::vector<float> > scores(kClasses);

    for (int scale = 0; scale < 3; ++scale) {
      const int stride = kStrides[scale];
      const int cls_index = output_order_[scale * 2];
      const int box_index = output_order_[scale * 2 + 1];
      const hbDNNTensor& cls_tensor = outputs_[cls_index];
      const hbDNNTensor& box_tensor = outputs_[box_index];
      const int height = cls_tensor.properties.validShape.dimensionSize[1];
      const int width = cls_tensor.properties.validShape.dimensionSize[2];
      const float* cls_data =
          reinterpret_cast<const float*>(cls_tensor.sysMem[0].virAddr);
      const float* box_data =
          reinterpret_cast<const float*>(box_tensor.sysMem[0].virAddr);

      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const float* logits = tensor_cell(cls_tensor, cls_data, y, x);
          int class_id = 0;
          for (int c = 1; c < kClasses; ++c) {
            if (logits[c] > logits[class_id]) class_id = c;
          }
          if (logits[class_id] < raw_threshold) continue;

          const float* ltrb = tensor_cell(box_tensor, box_data, y, x);
          float x1 = (x + 0.5f - ltrb[0]) * stride;
          float y1 = (y + 0.5f - ltrb[1]) * stride;
          float x2 = (x + 0.5f + ltrb[2]) * stride;
          float y2 = (y + 0.5f + ltrb[3]) * stride;

          x1 = (x1 - transform.shift_x) / transform.scale_x;
          y1 = (y1 - transform.shift_y) / transform.scale_y;
          x2 = (x2 - transform.shift_x) / transform.scale_x;
          y2 = (y2 - transform.shift_y) / transform.scale_y;
          x1 = std::max(0.0f, std::min(x1, static_cast<float>(image_w)));
          y1 = std::max(0.0f, std::min(y1, static_cast<float>(image_h)));
          x2 = std::max(0.0f, std::min(x2, static_cast<float>(image_w)));
          y2 = std::max(0.0f, std::min(y2, static_cast<float>(image_h)));
          if (x2 <= x1 || y2 <= y1) continue;

          boxes[class_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
          scores[class_id].push_back(sigmoid(logits[class_id]));
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
  int find_output(int height, int width, int channels) const {
    int match = -1;
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const hbDNNTensorShape& shape = outputs_[i].properties.validShape;
      if (shape.dimensionSize[0] == 1 && shape.dimensionSize[1] == height &&
          shape.dimensionSize[2] == width && shape.dimensionSize[3] == channels) {
        if (match >= 0) return -1;
        match = static_cast<int>(i);
      }
    }
    return match;
  }

  const float* tensor_cell(const hbDNNTensor& tensor, const float* data,
                           int y, int x) const {
    const hbDNNTensorShape& valid = tensor.properties.validShape;
    const hbDNNTensorShape& aligned = tensor.properties.alignedShape;
    const int aligned_width = aligned.numDimensions == 4
                                  ? aligned.dimensionSize[2]
                                  : valid.dimensionSize[2];
    const int aligned_channels = aligned.numDimensions == 4
                                     ? aligned.dimensionSize[3]
                                     : valid.dimensionSize[3];
    return data + (y * aligned_width + x) * aligned_channels;
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
    if (input_allocated_) {
      const int ret = hbSysFreeMem(&input_.sysMem[0]);
      if (ret != 0) {
        std::cerr << "[WARN] hbSysFreeMem(input) returned " << ret << std::endl;
      }
      input_allocated_ = false;
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
  hbDNNTensorProperties input_properties_{};
  hbDNNTensor input_{};
  std::vector<hbDNNTensor> outputs_;
  std::vector<bool> output_allocated_;
  int output_order_[kOutputCount] = {-1, -1, -1, -1, -1, -1};
  int input_h_ = 0;
  int input_w_ = 0;
  int input_bytes_ = 0;
  bool input_allocated_ = false;
  std::string model_name_;
};

bool run_pipeline(Yolo26Runtime* runtime, const cv::Mat& image,
                  const Options& options, std::vector<Detection>* detections,
                  StageTiming* timing) {
  const Clock::time_point start = Clock::now();
  ImageTransform transform;
  const cv::Mat resized = preprocess_image(image, runtime->input_h(), runtime->input_w(),
                                           options.resize_type, &transform);
  const cv::Mat nv12 = bgr_to_nv12(resized);
  if (!runtime->copy_input(nv12)) return false;
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
    timing->preprocess_ms = elapsed_ms(start, preprocessed);
    timing->runtime_ms = elapsed_ms(preprocessed, inferred);
    timing->postprocess_ms = elapsed_ms(inferred, finished);
    timing->end_to_end_ms = elapsed_ms(start, finished);
  }
  return true;
}

bool run_pipeline_streams(const std::vector<Yolo26Runtime*>& runtimes,
                          const cv::Mat& image, const Options& options,
                          int frames_per_stream, size_t expected_detections,
                          bool collect_timing, BenchmarkRound* result) {
  if (runtimes.empty()) return false;

  std::vector<StageSamples> stream_samples(runtimes.size());
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
          StageTiming timing;
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
    result->wall_ms = elapsed_ms(wall_start, wall_end);
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

double percentile(const std::vector<double>& values, double fraction) {
  if (values.empty()) return 0.0;
  std::vector<double> sorted(values);
  std::sort(sorted.begin(), sorted.end());
  const double position = (sorted.size() - 1) * fraction;
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  if (lower == upper) return sorted[lower];
  const double weight = position - lower;
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

Statistics summarize(const std::vector<double>& values) {
  Statistics result;
  if (values.empty()) return result;
  result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  result.p50 = percentile(values, 0.50);
  result.p95 = percentile(values, 0.95);
  result.min = *std::min_element(values.begin(), values.end());
  result.max = *std::max_element(values.begin(), values.end());
  return result;
}

void print_statistics(const std::string& label, const std::vector<double>& values) {
  const Statistics stats = summarize(values);
  std::cout << std::left << std::setw(13) << label << std::right << std::fixed
            << std::setprecision(3) << " mean=" << std::setw(8) << stats.mean
            << " ms  p50=" << std::setw(8) << stats.p50
            << "  p95=" << std::setw(8) << stats.p95
            << "  min=" << std::setw(8) << stats.min
            << "  max=" << std::setw(8) << stats.max << std::endl;
}

std::string json_escape(const std::string& value) {
  std::string result;
  for (size_t i = 0; i < value.size(); ++i) {
    const char ch = value[i];
    if (ch == '\\' || ch == '"') result.push_back('\\');
    if (ch == '\n') {
      result += "\\n";
    } else {
      result.push_back(ch);
    }
  }
  return result;
}

void write_metric(std::ofstream* stream, const std::string& name,
                  const std::vector<double>& values, bool comma) {
  const Statistics stats = summarize(values);
  *stream << "    \"" << name << "\": {\"mean\": " << stats.mean
          << ", \"p50\": " << stats.p50 << ", \"p95\": " << stats.p95
          << ", \"min\": " << stats.min << ", \"max\": " << stats.max
          << "}" << (comma ? "," : "") << "\n";
}

bool write_json(const std::string& path, const Options& options,
                const StageSamples& samples, size_t detections,
                size_t completed_frames, double aggregate_wall_ms) {
  std::ofstream stream(path.c_str());
  if (!stream) {
    std::cerr << "[ERROR] Cannot write benchmark JSON: " << path << std::endl;
    return false;
  }
  stream << std::fixed << std::setprecision(6);
  stream << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"model\": \"" << json_escape(options.model_path) << "\",\n"
         << "  \"image\": \"" << json_escape(options.image_path) << "\",\n"
         << "  \"implementation\": \"native_cpp_yolo26_ltrb\",\n"
         << "  \"timing_scope\": \"in_memory_bgr_to_detections\",\n"
         << "  \"pipeline_streams\": " << options.pipeline_streams << ",\n"
         << "  \"runtime_submission_threads\": " << options.pipeline_streams << ",\n"
         << "  \"cpu_thread_policy\": \""
         << (options.opencv_threads == 0 ? "all_online" : "fixed") << "\",\n"
         << "  \"online_cpu_threads\": " << cv::getNumberOfCPUs() << ",\n"
         << "  \"opencv_threads\": " << cv::getNumThreads() << ",\n"
         << "  \"warmup_frames_per_round\": " << options.warmup << ",\n"
         << "  \"runs_per_round\": " << options.runs << ",\n"
         << "  \"frames_per_stream_per_round\": " << options.runs << ",\n"
         << "  \"rounds\": " << options.rounds << ",\n"
         << "  \"timed_frames\": " << completed_frames << ",\n"
         << "  \"aggregate_wall_ms\": " << aggregate_wall_ms << ",\n"
         << "  \"detections_per_frame\": " << detections << ",\n"
         << "  \"score_threshold\": " << options.score_threshold << ",\n"
         << "  \"nms_threshold\": " << options.nms_threshold << ",\n"
         << "  \"metrics_ms\": {\n";
  write_metric(&stream, "preprocess", samples.preprocess, true);
  write_metric(&stream, "runtime", samples.runtime, true);
  write_metric(&stream, "postprocess", samples.postprocess, true);
  write_metric(&stream, "end_to_end", samples.end_to_end, false);
  stream << "  },\n"
         << "  \"throughput_fps\": "
         << (aggregate_wall_ms > 0.0 ? completed_frames * 1000.0 / aggregate_wall_ms
                                     : 0.0)
         << "\n}\n";
  return true;
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

    std::cout << "[INFO] YOLO26 direct-LTRB C++ sample" << std::endl;
    std::cout << "[INFO] OpenCV: " << CV_VERSION
              << ", online CPUs: " << online_cpu_threads
              << ", CPU thread policy: "
              << (options.opencv_threads == 0 ? "all-online" : "fixed")
              << ", OpenCV threads: " << cv::getNumThreads() << std::endl;
    std::cout << "[INFO] Loading model: " << options.model_path << std::endl;
    std::cout << "[INFO] Pipeline streams: " << options.pipeline_streams << std::endl;

    const Clock::time_point load_start = Clock::now();
    std::vector<std::unique_ptr<Yolo26Runtime> > runtime_storage;
    std::vector<Yolo26Runtime*> runtimes;
    runtime_storage.reserve(options.pipeline_streams);
    runtimes.reserve(options.pipeline_streams);
    for (int stream = 0; stream < options.pipeline_streams; ++stream) {
      std::unique_ptr<Yolo26Runtime> runtime(new Yolo26Runtime());
      if (!runtime->initialize(options.model_path)) return 1;
      if (!runtimes.empty() &&
          (runtime->input_h() != runtimes[0]->input_h() ||
           runtime->input_w() != runtimes[0]->input_w())) {
        std::cerr << "[ERROR] Runtime contexts expose different input shapes"
                  << std::endl;
        return 1;
      }
      runtimes.push_back(runtime.get());
      runtime_storage.push_back(std::move(runtime));
    }
    std::cout << "[INFO] Runtime context load: " << std::fixed
              << std::setprecision(3)
              << elapsed_ms(load_start, Clock::now()) << " ms" << std::endl;

    const cv::Mat image = cv::imread(options.image_path);
    if (image.empty()) {
      std::cerr << "[ERROR] Cannot read image: " << options.image_path << std::endl;
      return 1;
    }
    std::cout << "[INFO] Image: " << image.cols << "x" << image.rows << std::endl;

    std::vector<Detection> detections;
    StageTiming validation_timing;
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

    StageSamples aggregate;
    double aggregate_wall_ms = 0.0;
    size_t completed_frames = 0;
    const size_t expected_detections = detections.size();
    for (int round = 0; round < options.rounds; ++round) {
      if (!run_pipeline_streams(runtimes, image, options, options.warmup,
                                expected_detections, false, nullptr)) {
        return 1;
      }

      BenchmarkRound current;
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

    if (!options.json_path.empty() &&
        !write_json(options.json_path, options, aggregate, expected_detections,
                    completed_frames, aggregate_wall_ms)) {
      return 1;
    }
    if (!options.json_path.empty()) {
      std::cout << "[INFO] Benchmark JSON: " << options.json_path << std::endl;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    print_usage(argv[0]);
    return 1;
  }
}
