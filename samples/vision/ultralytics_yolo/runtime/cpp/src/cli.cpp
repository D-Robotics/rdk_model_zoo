/*
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

// CLI implementation of the ultralytics_yolo runtime: option
// parsing (kebab-case flags matching runtime/python/cli.py), image loading,
// the per-task renderers/reporters, and the bounded end-to-end detect
// benchmark. OpenCV types are confined to this translation unit.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cli.hpp"
#include "imagenet_labels.hpp"

namespace yolo {
namespace {

typedef std::chrono::steady_clock Clock;

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

const std::vector<std::string> kKeypointNames = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"};

const std::vector<std::pair<int, int>> kSkeleton = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},           // Head
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},  // Arms
    {5, 11}, {6, 12}, {11, 12},               // Torso
    {11, 13}, {13, 15}, {12, 14}, {14, 16}    // Legs
};

const cv::Scalar kKeypointColor = cv::Scalar(0, 0, 255);   // Red
const cv::Scalar kSkeletonColor = cv::Scalar(255, 0, 0);   // Blue
const cv::Scalar kBboxColor = cv::Scalar(0, 255, 0);       // Green

int parse_positive(const std::string& value, const std::string& option) {
  const int parsed = std::stoi(value);
  if (parsed <= 0) throw std::runtime_error(option + " must be positive");
  return parsed;
}

// Benchmark-flag parsing helpers (strtol/strtof based, no exceptions): the
// exact error strings are part of the CLI contract.
bool parse_int(const std::string& value, int minimum, int* output) {
  if (value.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (errno != 0 || end == value.c_str() || *end != '\0' ||
      parsed < minimum || parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  *output = static_cast<int>(parsed);
  return true;
}

bool parse_float(const std::string& value, float* output) {
  if (value.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const float parsed = std::strtof(value.c_str(), &end);
  if (errno != 0 || end == value.c_str() || *end != '\0' ||
      !std::isfinite(parsed)) {
    return false;
  }
  *output = parsed;
  return true;
}

bool is_sha256(const std::string& value) {
  if (value.size() != 64) return false;
  for (size_t i = 0; i < value.size(); ++i) {
    if (!std::isxdigit(static_cast<unsigned char>(value[i]))) return false;
  }
  return true;
}

// Detect render: axis-aligned boxes with a class-name/score label chip.
void draw_detections(cv::Mat* image,
                     const std::vector<YoloDetect::Detection>& detections) {
  for (size_t i = 0; i < detections.size(); ++i) {
    const YoloDetect::Detection& detection = detections[i];
    const cv::Scalar color = kColors[detection.class_id % kColors.size()];
    const cv::Rect2d bbox(detection.x1, detection.y1,
                          detection.x2 - detection.x1,
                          detection.y2 - detection.y1);
    cv::rectangle(*image, bbox, color, 2);
    const std::string label =
        kCocoNames[detection.class_id] + " " +
        std::to_string(static_cast<int>(detection.score * 100.0f)) + "%";
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    const int x = static_cast<int>(detection.x1);
    const int y = std::max(static_cast<int>(detection.y1),
                           text_size.height + 8);
    cv::rectangle(*image, cv::Point(x, y - text_size.height - 8),
                  cv::Point(x + text_size.width, y), color, cv::FILLED);
    cv::putText(*image, label, cv::Point(x, y - 4), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  }
}

// Segment render: box plus label chip, drawn per detection.
void draw_segment_detection(cv::Mat& img, int class_id, float score, float x1,
                            float y1, float x2, float y2) {
  const int left = static_cast<int>(x1);
  const int top = static_cast<int>(y1);
  const int right = static_cast<int>(x2);
  const int bottom = static_cast<int>(y2);

  const cv::Scalar color = kColors[class_id % kColors.size()];
  cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom), color, 2);

  const std::string label =
      kCocoNames[class_id] + ": " + std::to_string(score).substr(0, 4);

  int baseline;
  const cv::Size label_size =
      cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

  const int label_y = std::max(top, label_size.height);
  cv::rectangle(img, cv::Point(left, label_y - label_size.height),
                cv::Point(left + label_size.width, label_y + baseline), color,
                cv::FILLED);
  cv::putText(img, label, cv::Point(left, label_y), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(0, 0, 0), 1);
}

const float kRadToDeg = 180.0f / static_cast<float>(M_PI);

cv::RotatedRect to_rect(const RotatedBox& box) {
  return cv::RotatedRect(cv::Point2f(box.cx, box.cy),
                         cv::Size2f(box.width, box.height),
                         box.angle_rad * kRadToDeg);
}

// OBB render: rotated quad plus a class/score label at the first corner.
void draw_obb_detections(cv::Mat* image,
                         const std::vector<YoloObb::Detection>& detections) {
  for (const YoloObb::Detection& detection : detections) {
    cv::Point2f points[4];
    to_rect(detection.box).points(points);
    const cv::Scalar color = kColors[detection.class_id % 15];
    for (int p = 0; p < 4; ++p)
      cv::line(*image, points[p], points[(p + 1) % 4], color, 2, cv::LINE_AA);
    const std::string label =
        "class=" + std::to_string(detection.class_id) + " " +
        std::to_string(detection.score).substr(0, 4);
    cv::putText(*image, label, points[0], cv::FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv::LINE_AA);
  }
}

// One task pipeline through the explicit owned stages. Stage boundaries
// follow the stage API: preprocess is the pure pixel conversion, runtime is
// validate+upload+forward+owned copy, postprocess is the decode.
template <class Model>
bool run_pipeline(Model* model, const typename Model::Input& image,
                  typename Model::Result* result, StageTiming* timing) {
  const Clock::time_point start = Clock::now();
  const typename Model::Prepared prepared = model->preprocess(image);
  const Clock::time_point preprocessed = Clock::now();

  typename Model::RawResult raw = model->infer(prepared);
  const Clock::time_point inferred = Clock::now();

  const typename Model::Result decoded = model->postprocess(raw);
  const Clock::time_point finished = Clock::now();

  *result = decoded;

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

// Task result counts for the benchmark determinism check: object tasks
// report NMS-kept detections, classification reports its Top-K size.
size_t result_output_count(const YoloDetect::Result& result) {
  return result.detections.size();
}
size_t result_output_count(const YoloSegment::Result& result) {
  return result.detections.size();
}
size_t result_output_count(const YoloPose::Result& result) {
  return result.detections.size();
}
size_t result_output_count(const YoloClassify::Result& result) {
  return result.topk.size();
}
size_t result_output_count(const YoloObb::Result& result) {
  return result.detections.size();
}

}  // namespace

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [model] [image] [result.jpg] [options]\n"
      << "Options:\n"
      << "  --task detect|segment|pose|classify|obb\n"
      << "                       Task runtime to run (default: detect)\n"
      << "  --head auto|dfl|ltrb\n"
      << "                       Box decode contract (default: auto-detect\n"
      << "                       from output shapes; dfl covers\n"
      << "                       YOLOv5u/v8/v9/yolo11/yolo12/yolov13)\n"
      << "  --score-thres VALUE  Score threshold (default: 0.25)\n"
      << "  --nms-thres VALUE    NMS IoU threshold (default: 0.7 detect,\n"
      << "                       0.45 segment/pose, 0.2 obb)\n"
      << "  --classes N          OBB class channels (default: 15)\n"
      << "  --angle-sign VALUE   OBB angle convention (default: 1)\n"
      << "  --angle-offset VALUE OBB angle offset in degrees (default: 0)\n"
      << "  --no-regularize      Keep OBB boxes unregularized\n"
      << "  --resize-type 0|1    0=resize, 1=letterbox (default: 1)\n"
      << "  --topk N             Classification Top-K (default: 5)\n"
      << "  --kpt-conf-thres VALUE\n"
      << "                       Pose keypoint confidence (default: 0.5)\n"
      << "  --opencv-threads N|all\n"
      << "                       OpenCV CPU threads (default: all online CPUs)\n"
      << "  --benchmark          Run bounded end-to-end benchmark\n"
      << "  --warmup N           Warmup frames per round (default: 20)\n"
      << "  --runs N             Timed frames per round (default: 200)\n"
      << "  --rounds N           Benchmark rounds (default: 3)\n"
      << "  --pipeline-streams 1|2\n"
      << "                       Complete concurrent E2E pipelines (default: 1)\n"
      << "  --json PATH          Write aggregate benchmark JSON\n"
      << "  --runtime-source-sha256 HEX\n"
      << "  --executable-sha256 HEX\n"
      << "                       Provenance recorded in the benchmark JSON\n"
      << "  --no-save            Do not draw or save the validation result\n"
      << "  --help               Show this message\n";
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
    } else if (arg == "--no-regularize") {
      options.regularize_obb = false;
    } else if (arg == "--task") {
      options.task = next_value(arg);
    } else if (arg == "--warmup") {
      if (!parse_int(next_value(arg), 0, &options.warmup))
        throw std::runtime_error("--warmup must be a non-negative integer");
    } else if (arg == "--runs") {
      if (!parse_int(next_value(arg), 1, &options.runs))
        throw std::runtime_error("--runs must be a positive integer");
    } else if (arg == "--rounds") {
      if (!parse_int(next_value(arg), 1, &options.rounds))
        throw std::runtime_error("--rounds must be a positive integer");
    } else if (arg == "--json") {
      options.json_path = next_value(arg);
      if (options.json_path.empty())
        throw std::runtime_error("--json path cannot be empty");
    } else if (arg == "--head") {
      options.head = next_value(arg);
    } else if (arg == "--score-thres") {
      options.score_threshold = std::stof(next_value(arg));
    } else if (arg == "--nms-thres") {
      options.nms_threshold = std::stof(next_value(arg));
      options.nms_given = true;
    } else if (arg == "--kpt-conf-thres") {
      options.kpt_conf_threshold = std::stof(next_value(arg));
    } else if (arg == "--topk") {
      options.topk = parse_positive(next_value(arg), arg);
    } else if (arg == "--classes") {
      if (!parse_int(next_value(arg), 1, &options.classes))
        throw std::runtime_error("--classes must be a positive integer");
    } else if (arg == "--angle-sign") {
      if (!parse_float(next_value(arg), &options.angle_sign))
        throw std::runtime_error("--angle-sign must be finite");
    } else if (arg == "--angle-offset") {
      if (!parse_float(next_value(arg), &options.angle_offset_degrees))
        throw std::runtime_error("--angle-offset must be finite degrees");
    } else if (arg == "--resize-type") {
      options.resize_type = std::stoi(next_value(arg));
    } else if (arg == "--opencv-threads") {
      const std::string value = next_value(arg);
      if (value == "all") {
        options.opencv_threads = 0;
      } else if (!parse_int(value, 1, &options.opencv_threads)) {
        throw std::runtime_error("--opencv-threads must be 'all' or positive");
      }
    } else if (arg == "--pipeline-streams") {
      if (!parse_int(next_value(arg), 1, &options.pipeline_streams) ||
          options.pipeline_streams > 2)
        throw std::runtime_error("--pipeline-streams must be 1 or 2");
    } else if (arg == "--runtime-source-sha256") {
      const std::string value = next_value(arg);
      if (!is_sha256(value))
        throw std::runtime_error(
            "--runtime-source-sha256 must be 64 hex digits");
      options.runtime_source_sha256 = value;
    } else if (arg == "--executable-sha256") {
      const std::string value = next_value(arg);
      if (!is_sha256(value))
        throw std::runtime_error(
            "--executable-sha256 must be 64 hex digits");
      options.executable_sha256 = value;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.size() > 3)
    throw std::runtime_error("too many positional arguments");

  // Per-task defaults for model, image and result paths.
  if (options.task == "detect") {
    options.model_path = "yolo26n_detect_bayese_640x640_nv12.bin";
    options.image_path = "test_data/bus.jpg";
    options.output_path = "cpp_result.jpg";
    if (!options.nms_given) options.nms_threshold = 0.7f;
  } else if (options.task == "segment") {
    options.model_path =
        "source/reference_bin_models/seg/yolo11n_seg_bayese_640x640_nv12.bin";
    options.image_path = "test_data/bus.jpg";
    options.output_path = "segment_result.jpg";
  } else if (options.task == "pose") {
    options.model_path =
        "source/reference_bin_models/pose/yolo11n_pose_bayese_640x640_nv12.bin";
    options.image_path = "test_data/bus.jpg";
    options.output_path = "pose_result.jpg";
  } else if (options.task == "classify") {
    options.model_path =
        "source/reference_bin_models/cls/yolo11n_cls_bayese_224x224_nv12.bin";
    options.image_path = "test_data/zebra_cls.jpg";
    options.output_path.clear();
  } else if (options.task == "obb") {
    options.model_path = "yolo26n_obb_640x640_nv12.bin";
    options.image_path = "test_data/dota.jpg";
    options.output_path = "obb_result.jpg";
    if (!options.nms_given) options.nms_threshold = 0.2f;
  } else {
    throw std::runtime_error(
        "--task must be detect, segment, pose, classify or obb");
  }
  if (!positional.empty()) {
    options.model_path = positional[0];
    options.model_given = true;
  }
  if (positional.size() > 1) {
    options.image_path = positional[1];
    options.image_given = true;
  }
  if (positional.size() > 2) {
    options.output_path = positional[2];
    options.output_given = true;
  }

  if (options.head != "auto" && options.head != "dfl" && options.head != "ltrb")
    throw std::runtime_error("--head must be auto, dfl or ltrb");
  if (options.resize_type != 0 && options.resize_type != 1)
    throw std::runtime_error("--resize-type must be 0 or 1");
  if (options.score_threshold < 0.0f || options.score_threshold > 1.0f)
    throw std::runtime_error("--score-thres must be in [0, 1]");
  if (options.nms_threshold < 0.0f || options.nms_threshold > 1.0f)
    throw std::runtime_error("--nms-thres must be in [0, 1]");
  if (options.kpt_conf_threshold < 0.0f || options.kpt_conf_threshold > 1.0f)
    throw std::runtime_error("--kpt-conf-thres must be in [0, 1]");
  return options;
}

struct SourceImage::State {
  cv::Mat image;
  std::vector<uint8_t> bytes;
  bool valid = false;
};

SourceImage::SourceImage(const std::string& path)
    : state_(new State) {
  state_->image = cv::imread(path);
  state_->valid = !state_->image.empty() && state_->image.type() == CV_8UC3;
  if (state_->valid) {
    const cv::Mat continuous = state_->image.isContinuous()
                                   ? state_->image
                                   : state_->image.clone();
    const uint8_t* data = continuous.ptr<uint8_t>();
    state_->bytes.assign(
        data, data + static_cast<size_t>(continuous.rows) * continuous.cols * 3);
  }
}

SourceImage::~SourceImage() = default;

bool SourceImage::valid() const { return state_->valid; }
int SourceImage::rows() const { return state_->image.rows; }
int SourceImage::cols() const { return state_->image.cols; }
const std::vector<uint8_t>& SourceImage::bgr() const { return state_->bytes; }

void report_detect(const Options& options, const YoloDetect::Prediction& run,
                   const SourceImage& source) {
  std::cout << "[INFO] Validation detections: "
            << run.result.detections.size() << std::endl;

  if (options.save_result) {
    cv::Mat rendered(source.rows(), source.cols(), CV_8UC3,
                     const_cast<uint8_t*>(source.bgr().data()));
    draw_detections(&rendered, run.result.detections);
    if (!cv::imwrite(options.output_path, rendered))
      throw std::runtime_error("Cannot write result: " + options.output_path);
    std::cout << "[INFO] Result saved: " << options.output_path << std::endl;
  }
}

void report_segment(const Options& options,
                    const YoloSegment::Prediction& run) {
  const int input_h = run.model_rows;
  const int input_w = run.model_cols;
  cv::Mat img_display(input_h, input_w, CV_8UC3,
                      const_cast<uint8_t*>(run.model_frame_bgr.data()));
  cv::Mat mask_overlay = cv::Mat::zeros(input_h, input_w, CV_8UC3);

  for (const YoloSegment::Detection& det : run.result.detections) {
    std::cout << "[INFO] Detection: " << kCocoNames[det.class_id]
              << ", score=" << std::fixed << std::setprecision(3) << det.score
              << std::endl;

    draw_segment_detection(img_display, det.class_id, det.score, det.x1,
                           det.y1, det.x2, det.y2);

    // Degenerate mask crops were skipped in postprocess together with their
    // boxes, so every surviving detection has a usable mask plane at its
    // recorded clamped origin.
    const cv::Rect roi(det.mask.x, det.mask.y, det.mask.cols, det.mask.rows);
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > input_w ||
        roi.y + roi.height > input_h) {
      throw std::runtime_error("segment mask does not fit the model frame");
    }

    const cv::Scalar color = kColors[det.class_id % kColors.size()];
    cv::Mat color_mask(det.mask.rows, det.mask.cols, CV_8UC3, color);
    cv::Mat roi_mask(det.mask.rows, det.mask.cols, CV_8UC1,
                     const_cast<uint8_t*>(det.mask.bytes.data()));
    cv::Mat masked_color;
    cv::bitwise_and(color_mask, color_mask, masked_color, roi_mask);

    cv::Mat overlay_roi = mask_overlay(roi);
    cv::addWeighted(overlay_roi, 1.0, masked_color, 0.6, 0, overlay_roi);
  }

  cv::Mat final_result;
  cv::addWeighted(img_display, 0.7, mask_overlay, 0.3, 0, final_result);

  cv::Mat concatenated;
  cv::hconcat(img_display, mask_overlay, concatenated);
  cv::hconcat(concatenated, final_result, concatenated);

  if (!cv::imwrite(options.output_path, concatenated))
    throw std::runtime_error("Failed to save segmentation image");
  std::cout << "[INFO] Result saved to: " << options.output_path << std::endl;
  std::cout << "[INFO] Output size: " << concatenated.cols << "x"
            << concatenated.rows << std::endl;
}

void report_pose(const Options& options, const YoloPose::Prediction& run,
                 const SourceImage& source) {
  // Keypoint scores are raw logits; threshold in logit space, which is
  // equivalent to the sigmoid-space confidence threshold.
  const float kpt_thres_raw =
      -std::log(1.0f / options.kpt_conf_threshold - 1.0f);

  cv::Mat result_img(source.rows(), source.cols(), CV_8UC3,
                     const_cast<uint8_t*>(source.bgr().data()));

  for (const YoloPose::Detection& det : run.result.detections) {
    const int x1 = static_cast<int>(det.x1);
    const int y1 = static_cast<int>(det.y1);
    const int x2 = static_cast<int>(det.x2);
    const int y2 = static_cast<int>(det.y2);

    cv::rectangle(result_img, cv::Point(x1, y1), cv::Point(x2, y2), kBboxColor,
                  2);

    const std::string label =
        "person: " + std::to_string(det.score).substr(0, 4);
    int baseline;
    const cv::Size label_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int label_y = std::max(y1, label_size.height);
    cv::rectangle(result_img, cv::Point(x1, label_y - label_size.height),
                  cv::Point(x1 + label_size.width, label_y + baseline),
                  kBboxColor, cv::FILLED);
    cv::putText(result_img, label, cv::Point(x1, label_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    for (const auto& connection : kSkeleton) {
      const int idx1 = connection.first;
      const int idx2 = connection.second;
      if (det.keypoints[idx1].score >= kpt_thres_raw &&
          det.keypoints[idx2].score >= kpt_thres_raw) {
        const cv::Point pt1(static_cast<int>(det.keypoints[idx1].x),
                            static_cast<int>(det.keypoints[idx1].y));
        const cv::Point pt2(static_cast<int>(det.keypoints[idx2].x),
                            static_cast<int>(det.keypoints[idx2].y));
        cv::line(result_img, pt1, pt2, kSkeletonColor, 2);
      }
    }

    for (int i = 0; i < 17; ++i) {
      if (det.keypoints[i].score >= kpt_thres_raw) {
        const int x = static_cast<int>(det.keypoints[i].x);
        const int y = static_cast<int>(det.keypoints[i].y);
        cv::circle(result_img, cv::Point(x, y), 5, kKeypointColor, -1);
        cv::circle(result_img, cv::Point(x, y), 2, cv::Scalar(0, 255, 255), -1);
        cv::putText(result_img, std::to_string(i), cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, kKeypointColor, 2);
        cv::putText(result_img, std::to_string(i), cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
      }
    }

    std::cout << "[INFO] Person detected: score=" << std::fixed
              << std::setprecision(3) << det.score << ", bbox=(" << det.x1
              << "," << det.y1 << "," << (det.x2 - det.x1) << ","
              << (det.y2 - det.y1) << ")" << std::endl;
  }

  if (!cv::imwrite(options.output_path, result_img))
    throw std::runtime_error("Failed to save pose image");
  std::cout << "[INFO] Result saved to: " << options.output_path << std::endl;
}

void report_classify(const Options& options,
                     const YoloClassify::Prediction& run) {
  std::cout << "[INFO] Classification results:" << std::endl;
  std::cout << "[INFO] Image: " << options.image_path << std::endl;
  std::cout << std::endl;

  for (size_t i = 0; i < run.result.topk.size(); i++) {
    const ClassificationScore& res = run.result.topk[i];
    std::cout << "\033[1;32m"
              << "TOP" << (i + 1) << " -> "
              << "id: " << res.id << ", "
              << "score: " << std::fixed << std::setprecision(3)
              << res.probability << ", "
              << "name: " << IMAGENET_CLASSES.at(res.id) << "\033[0m"
              << std::endl;
  }
}

void report_obb(const Options& options, const YoloObb::Prediction& run,
                const SourceImage& source) {
  std::cout << "[INFO] Validation rotated boxes: "
            << run.result.detections.size() << std::endl;

  for (const YoloObb::Detection& detection : run.result.detections) {
    std::cout << "[INFO] class=" << detection.class_id << " score="
              << std::fixed << std::setprecision(3) << detection.score
              << " center=(" << detection.box.cx << "," << detection.box.cy
              << ") size=(" << detection.box.width << ","
              << detection.box.height
              << ") angle_deg=" << detection.box.angle_rad * kRadToDeg
              << std::endl;
  }

  if (options.save_result) {
    cv::Mat rendered(source.rows(), source.cols(), CV_8UC3,
                     const_cast<uint8_t*>(source.bgr().data()));
    draw_obb_detections(&rendered, run.result.detections);
    if (!cv::imwrite(options.output_path, rendered))
      throw std::runtime_error("Cannot write result: " + options.output_path);
    std::cout << "[INFO] Result saved: " << options.output_path << std::endl;
  }
}

namespace {

// Shared shape of every task benchmark: stream 0 reuses the model main
// constructed and validated; only the remaining streams allocate their own
// independent contexts, so the process holds exactly pipeline_streams
// runtimes and no extra validation predict is run. `make_config` rebuilds
// the task Config for the CLI-owned worker streams.
template <class Model, class MakeConfig>
int run_task_benchmark(const Options& options, const SourceImage& source,
                       Model* main_model, size_t outputs_per_frame,
                       const std::string& implementation,
                       const char* output_kind, const char* timing_scope,
                       MakeConfig make_config) {
  std::vector<std::unique_ptr<Model> > model_storage;
  std::vector<Model*> models;
  models.reserve(options.pipeline_streams);
  models.push_back(main_model);
  model_storage.reserve(std::max(0, options.pipeline_streams - 1));
  for (int stream = 1; stream < options.pipeline_streams; ++stream) {
    std::unique_ptr<Model> model(new Model(make_config(options)));
    if (model->input_h() != main_model->input_h() ||
        model->input_w() != main_model->input_w()) {
      std::cerr << "[ERROR] Runtime contexts expose different inputs"
                << std::endl;
      return 1;
    }
    models.push_back(model.get());
    model_storage.push_back(std::move(model));
  }

  typename Model::Input image;
  image.bgr = source.bgr();
  image.source_rows = source.rows();
  image.source_cols = source.cols();

  std::vector<BenchmarkPipeline> pipelines;
  for (Model* model : models) {
    pipelines.push_back([model, &image](StageTiming* timing, size_t* count,
                                        std::string* error) {
      try {
        typename Model::Result result;
        if (!run_pipeline(model, image, &result, timing)) {
          if (error) *error = "pipeline failed";
          return false;
        }
        *count = result_output_count(result);
        return true;
      } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
      }
    });
  }

  BenchmarkOptions benchmark;
  benchmark.enabled = options.benchmark;
  benchmark.save_result = options.save_result;
  benchmark.warmup_frames = options.warmup;
  benchmark.runs_per_round = options.runs;
  benchmark.rounds = options.rounds;
  benchmark.pipeline_streams = options.pipeline_streams;
  benchmark.opencv_threads = options.opencv_threads;
  benchmark.resize_type = options.resize_type;
  benchmark.score_threshold = options.score_threshold;
  benchmark.nms_threshold = options.nms_threshold;
  benchmark.angle_sign = options.angle_sign;
  benchmark.angle_offset_degrees = options.angle_offset_degrees;
  benchmark.regularize_obb = options.regularize_obb;
  benchmark.runtime_source_sha256 = options.runtime_source_sha256;
  benchmark.executable_sha256 = options.executable_sha256;
  benchmark.json_path = options.json_path;

  BenchmarkMeta meta;
  meta.model_path = options.model_path;
  meta.image_path = options.image_path;
  meta.implementation = implementation;
  meta.output_kind = output_kind;
  meta.timing_scope = timing_scope;
  meta.resize_type = options.resize_type;
  meta.cpu_thread_policy = options.opencv_threads == 0 ? "all_online" : "fixed";
  meta.online_cpu_threads = cv::getNumberOfCPUs();
  meta.opencv_threads = cv::getNumThreads();
  return run_benchmark(pipelines, benchmark, outputs_per_frame, meta) ? 0 : 1;
}

}  // namespace

int run_detect_benchmark(const Options& options, const SourceImage& source,
                         YoloDetect* main_model,
                         const YoloDetect::Prediction& validation) {
  return run_task_benchmark(
      options, source, main_model, validation.result.detections.size(),
      main_model->direct_ltrb() ? "native_cpp_yolo26_ltrb"
                                : "native_cpp_yolo_dfl",
      "detections", "in_memory_bgr_to_detections",
      [](const Options& parsed) {
        YoloDetect::Config config;
        config.model_path = parsed.model_path;
        config.head = parsed.head;
        config.score_threshold = parsed.score_threshold;
        config.nms_threshold = parsed.nms_threshold;
        config.resize_type = parsed.resize_type;
        return config;
      });
}

int run_segment_benchmark(const Options& options, const SourceImage& source,
                          YoloSegment* main_model,
                          const YoloSegment::Prediction& validation) {
  return run_task_benchmark(
      options, source, main_model, validation.result.detections.size(),
      main_model->direct_ltrb() ? "native_cpp_yolo26_seg_ltrb"
                                : "native_cpp_yolo_seg_dfl",
      "instance_masks", "in_memory_bgr_to_instance_masks",
      [](const Options& parsed) {
        YoloSegment::Config config;
        config.model_path = parsed.model_path;
        config.score_threshold = parsed.score_threshold;
        config.nms_threshold = parsed.nms_threshold;
        config.mask_threshold = 0.5f;
        config.resize_type = parsed.resize_type;
        return config;
      });
}

int run_pose_benchmark(const Options& options, const SourceImage& source,
                       YoloPose* main_model,
                       const YoloPose::Prediction& validation) {
  return run_task_benchmark(
      options, source, main_model, validation.result.detections.size(),
      main_model->direct_ltrb() ? "native_cpp_yolo26_pose_ltrb"
                                : "native_cpp_yolo_pose_dfl",
      "pose_instances", "in_memory_bgr_to_pose_instances",
      [](const Options& parsed) {
        YoloPose::Config config;
        config.model_path = parsed.model_path;
        config.score_threshold = parsed.score_threshold;
        config.nms_threshold = parsed.nms_threshold;
        config.kpt_conf_threshold = parsed.kpt_conf_threshold;
        config.resize_type = parsed.resize_type;
        return config;
      });
}

int run_classify_benchmark(const Options& options, const SourceImage& source,
                           YoloClassify* main_model,
                           const YoloClassify::Prediction& validation) {
  return run_task_benchmark(
      options, source, main_model, validation.result.topk.size(),
      "native_cpp_yolo_classify", "topk_predictions",
      "in_memory_bgr_to_topk_predictions",
      [](const Options& parsed) {
        YoloClassify::Config config;
        config.model_path = parsed.model_path;
        config.topk = parsed.topk;
        config.resize_type = parsed.resize_type;
        return config;
      });
}

int run_obb_benchmark(const Options& options, const SourceImage& source,
                      YoloObb* main_model,
                      const YoloObb::Prediction& validation) {
  return run_task_benchmark(
      options, source, main_model, validation.result.detections.size(),
      "native_cpp_yolo26_obb_ltrb", "rotated_boxes",
      "in_memory_bgr_to_rotated_boxes",
      [](const Options& parsed) {
        YoloObb::Config config;
        config.model_path = parsed.model_path;
        config.classes = parsed.classes > 0 ? parsed.classes : 15;
        config.score_threshold = parsed.score_threshold;
        config.nms_threshold = parsed.nms_threshold;
        config.angle_sign = parsed.angle_sign;
        config.angle_offset_degrees = parsed.angle_offset_degrees;
        config.regularize_obb = parsed.regularize_obb;
        config.resize_type = parsed.resize_type;
        return config;
      });
}

}  // namespace yolo
