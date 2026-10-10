// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0

// YOLOE open-vocabulary segmentation model: letterbox geometry, NV12
// preparation, the E11/E26 float head decodes, prototype mask restoration,
// board/artifact preflight and the YOLOE task stages. The board DNN adapter
// (SdkRunner) is compiled only in SDK builds (YOLOE_HAS_SDK) on top of the
// ultralytics_yolo shared backend (backend.hpp/backend.cpp); host builds get
// the SDK-free core plus link-time runner fixtures.

#include "detect.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "sha256.h"

#ifdef YOLOE_HAS_SDK
#include "backend.hpp"
#endif

namespace yoloe {

int round_even(double x) {
  int floor = static_cast<int>(std::floor(x));
  double fraction = x - floor;
  return floor + (fraction > 0.5 || (fraction == 0.5 && floor % 2));
}

Geometry make_geometry(int width, int height, Protocol protocol,
                       int resize_type) {
  if (width <= 0 || height <= 0 ||
      (protocol != Protocol::E11 && protocol != Protocol::E26) ||
      (resize_type != 0 && resize_type != 1) ||
      (protocol == Protocol::E26 && resize_type != 1))
    throw std::invalid_argument("Invalid YOLOE image geometry/protocol");
  int rw = 640, rh = 640;
  if (resize_type == 1) {
    double gain = std::min(640.0 / width, 640.0 / height);
    rw = protocol == Protocol::E26 ? round_even(width * gain)
                                   : static_cast<int>(width * gain);
    rh = protocol == Protocol::E26 ? round_even(height * gain)
                                   : static_cast<int>(height * gain);
    rw = std::clamp(rw, 1, 640);
    rh = std::clamp(rh, 1, 640);
  }
  int left = (640 - rw) / 2, top = (640 - rh) / 2;
  return {width,        height,        rw,          rh,       left, top,
          640 - rw - left, 640 - rh - top, resize_type, protocol};
}

void validate_geometry(const Geometry &g) {
  auto expected = make_geometry(g.width, g.height, g.protocol, g.resize_type);
  if (g.resized_w != expected.resized_w || g.resized_h != expected.resized_h ||
      g.left != expected.left || g.top != expected.top ||
      g.right != expected.right || g.bottom != expected.bottom)
    throw std::invalid_argument("Geometry differs from preprocessing protocol");
}

std::array<float, 4> restore_box(const std::array<float, 4> &box,
                                 const Geometry &g) {
  validate_geometry(g);
  for (float v : box)
    if (!std::isfinite(v))
      throw std::invalid_argument("Nonfinite box");
  if (box[2] < box[0] || box[3] < box[1])
    throw std::invalid_argument("Reversed box");
  float sx = static_cast<float>(static_cast<double>(g.resized_w) / g.width);
  float sy = static_cast<float>(static_cast<double>(g.resized_h) / g.height);
  return {std::clamp((box[0] - g.left) / sx, 0.f, static_cast<float>(g.width)),
          std::clamp((box[1] - g.top) / sy, 0.f, static_cast<float>(g.height)),
          std::clamp((box[2] - g.left) / sx, 0.f, static_cast<float>(g.width)),
          std::clamp((box[3] - g.top) / sy, 0.f, static_cast<float>(g.height))};
}

void validate_config(const Config &cfg) {
  if (cfg.protocol != Protocol::E11 && cfg.protocol != Protocol::E26)
    throw std::invalid_argument("Unknown YOLOE protocol");
  if (!std::isfinite(cfg.score_threshold) || cfg.score_threshold <= 0 ||
      cfg.score_threshold >= 1)
    throw std::invalid_argument("Score threshold must be finite in (0,1)");
  if (cfg.protocol == Protocol::E11) {
    if (cfg.max_det != 300 || !cfg.single_label ||
        (cfg.resize_type != 0 && cfg.resize_type != 1))
      throw std::invalid_argument(
          "E11 requires single-label and no E26 Top-K override");
    if (cfg.nms_threshold && (!std::isfinite(*cfg.nms_threshold) ||
                              *cfg.nms_threshold < 0 || *cfg.nms_threshold > 1))
      throw std::invalid_argument("E11 NMS must be finite in [0,1]");
  } else if (cfg.nms_threshold || cfg.do_morph || cfg.resize_type != 1 ||
             cfg.max_det < 1 || cfg.max_det > 8400)
    throw std::invalid_argument(
        "E26 requires letterbox, no NMS/morphology and max_det in [1,8400]");
}

Nv12Input to_nv12(const cv::Mat &pixels) {
  if (pixels.type() != CV_8UC3 || pixels.rows != 640 || pixels.cols != 640)
    throw std::invalid_argument("NV12 requires prepared uint8 BGR 640x640");
  cv::Mat i420;
  cv::cvtColor(pixels, i420, cv::COLOR_BGR2YUV_I420);
  if (!i420.isContinuous() || i420.total() != 640 * 640 * 3 / 2)
    throw std::runtime_error("Unexpected OpenCV I420 storage");
  Nv12Input input;
  input.y.resize(640 * 640);
  input.uv.resize(320 * 640);
  const auto *y = i420.ptr<uint8_t>();
  yolo::i420_to_split_nv12(y, y + 640 * 640, y + 640 * 640 + 320 * 320, 640,
                           640, input.y.data(), 640, input.uv.data(), 640);
  return input;
}

PreparedBGR prepare_bgr(const cv::Mat &image, Protocol protocol,
                        int resize_type) {
  if (image.empty() || image.type() != CV_8UC3)
    throw std::invalid_argument("Expected nonempty BGR uint8 image");
  auto geometry = make_geometry(image.cols, image.rows, protocol, resize_type);
  cv::Mat resized, pixels;
  cv::resize(image, resized, {geometry.resized_w, geometry.resized_h}, 0, 0,
             resize_type == 0 ? cv::INTER_NEAREST : cv::INTER_LINEAR);
  int value = protocol == Protocol::E26 ? 114 : 127;
  cv::copyMakeBorder(resized, pixels, geometry.top, geometry.bottom,
                     geometry.left, geometry.right, cv::BORDER_CONSTANT,
                     cv::Scalar(value, value, value));
  return {pixels, geometry};
}

std::vector<RestoredMask>
restore_e26_masks(const std::vector<RawDetection> &candidates,
                  const std::vector<float> &proto, const Geometry &geometry) {
  validate_geometry(geometry);
  if (geometry.protocol != Protocol::E26 || proto.size() != 160 * 160 * 32)
    throw std::invalid_argument(
        "E26 masks require matching geometry and 160x160x32 prototype");
  for (float value : proto)
    if (!std::isfinite(value))
      throw std::invalid_argument("Nonfinite prototype");
  std::vector<RestoredMask> result;
  result.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    auto box = restore_box(candidate.box, geometry);
    for (float value : candidate.coefficients)
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite mask coefficient");
    cv::Mat raw(160, 160, CV_32FC1);
    for (int y = 0; y < 160; ++y)
      for (int x = 0; x < 160; ++x) {
        float sum = 0;
        const float *pixel = proto.data() + (y * 160 + x) * 32;
        for (int c = 0; c < 32; ++c)
          sum += pixel[c] * candidate.coefficients[c];
        if (!std::isfinite(sum))
          throw std::invalid_argument("Mask combination overflow");
        raw.at<float>(y, x) = sum;
      }
    cv::Mat canvas, binary(640, 640, CV_8UC1, cv::Scalar(0));
    cv::resize(raw, canvas, {640, 640}, 0, 0, cv::INTER_LINEAR);
    for (int y = 0; y < 640; ++y)
      for (int x = 0; x < 640; ++x)
        binary.at<unsigned char>(y, x) =
            canvas.at<float>(y, x) > 0 && x >= candidate.box[0] &&
            x < candidate.box[2] && y >= candidate.box[1] &&
            y < candidate.box[3];
    cv::Mat content = binary(cv::Rect(geometry.left, geometry.top,
                                      geometry.resized_w, geometry.resized_h));
    cv::Mat full;
    cv::resize(content, full, {geometry.width, geometry.height}, 0, 0,
               cv::INTER_NEAREST);
    auto bound = [](float value, int limit) {
      return static_cast<int>(std::clamp(static_cast<double>(value), 0.0,
                                         static_cast<double>(limit)));
    };
    int x1 = bound(box[0], geometry.width), x2 = bound(box[2], geometry.width);
    int y1 = bound(box[1], geometry.height),
        y2 = bound(box[3], geometry.height);
    cv::Mat mask(std::max(y2 - y1, 0), std::max(x2 - x1, 0), CV_8UC1);
    if (x2 > x1 && y2 > y1)
      mask = full(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    result.push_back({box, mask});
  }
  return result;
}

std::vector<RestoredMask>
restore_e11_masks(const std::vector<RawDetection> &candidates,
                  const std::vector<float> &proto, const Geometry &geometry,
                  bool do_morph) {
  validate_geometry(geometry);
  if (geometry.protocol != Protocol::E11 || proto.size() != 160 * 160 * 32)
    throw std::invalid_argument(
        "E11 masks require matching geometry and 160x160x32 prototype");
  for (float value : proto)
    if (!std::isfinite(value))
      throw std::invalid_argument("Nonfinite prototype");
  std::vector<RestoredMask> result;
  result.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    auto box = restore_box(candidate.box, geometry);
    for (float value : candidate.coefficients)
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite mask coefficient");
    // Clip to actual image content before the prototype crop, excluding
    // padding.
    int left = static_cast<int>(
        std::clamp(candidate.box[0], static_cast<float>(geometry.left),
                   static_cast<float>(640 - geometry.right)) *
        0.25f);
    int right = static_cast<int>(
        std::clamp(candidate.box[2], static_cast<float>(geometry.left),
                   static_cast<float>(640 - geometry.right)) *
        0.25f);
    int top = static_cast<int>(
        std::clamp(candidate.box[1], static_cast<float>(geometry.top),
                   static_cast<float>(640 - geometry.bottom)) *
        0.25f);
    int bottom = static_cast<int>(
        std::clamp(candidate.box[3], static_cast<float>(geometry.top),
                   static_cast<float>(640 - geometry.bottom)) *
        0.25f);
    auto bound = [](float value, int limit) {
      return static_cast<int>(std::clamp(static_cast<double>(value), 0.0,
                                         static_cast<double>(limit)));
    };
    int width = std::max(0, bound(box[2], geometry.width) -
                                bound(box[0], geometry.width));
    int height = std::max(0, bound(box[3], geometry.height) -
                                 bound(box[1], geometry.height));
    cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));
    if (width && height && right > left && bottom > top) {
      cv::Mat cropped(bottom - top, right - left, CV_8UC1);
      for (int y = top; y < bottom; ++y)
        for (int x = left; x < right; ++x) {
          const float *pixel = proto.data() + (y * 160 + x) * 32;
          float sum = 0;
          for (int c = 0; c < 32; ++c)
            sum += pixel[c] * candidate.coefficients[c];
          if (!std::isfinite(sum))
            throw std::invalid_argument("Mask combination overflow");
          cropped.at<unsigned char>(y - top, x - left) = sum > 0.5f;
        }
      cv::resize(cropped, mask, {width, height}, 0, 0, cv::INTER_LANCZOS4);
      if (do_morph)
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                         cv::Mat::ones(5, 5, CV_8UC1));
      // Lanczos may overshoot to 2; retain foreground support as binary 0/1.
      cv::threshold(mask, mask, 0, 1, cv::THRESH_BINARY);
    }
    result.push_back({box, mask});
  }
  return result;
}

namespace detail {

float candidate_iou(const RawDetection &a, const RawDetection &b) {
  float w = std::max(0.f, std::min(a.box[2], b.box[2]) -
                              std::max(a.box[0], b.box[0]));
  float h = std::max(0.f, std::min(a.box[3], b.box[3]) -
                              std::max(a.box[1], b.box[1]));
  float intersection = w * h;
  float area_a = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]);
  float area_b = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]);
  return intersection / (area_a + area_b - intersection + 1e-9f);
}

std::vector<RawDetection> nms_e11(const std::vector<RawDetection> &candidates,
                                  float threshold) {
  std::map<int, std::vector<size_t>> classes;
  for (size_t i = 0; i < candidates.size(); ++i)
    classes[candidates[i].label].push_back(i);
  std::vector<RawDetection> kept;
  for (auto &entry : classes) {
    auto &ids = entry.second;
    std::stable_sort(ids.begin(), ids.end(), [&](size_t a, size_t b) {
      return candidates[a].score > candidates[b].score;
    });
    std::vector<bool> suppressed(ids.size(), false);
    for (size_t i = 0; i < ids.size(); ++i) {
      if (suppressed[i])
        continue;
      kept.push_back(candidates[ids[i]]);
      for (size_t j = i + 1; j < ids.size(); ++j)
        if (!suppressed[j] &&
            candidate_iou(candidates[ids[i]], candidates[ids[j]]) > threshold)
          suppressed[j] = true;
    }
  }
  return kept;
}

} // namespace detail

// Ten compact float NHWC vectors, ordered cls/DFL64/coeff32 at strides 8/16/32
// then prototype. Score >= threshold, suppress IoU > NMS, per class.
std::vector<RawDetection>
decode_e11(const std::array<std::vector<float>, 10> &outputs,
           float score_threshold, float nms_threshold) {
  if (!std::isfinite(score_threshold) || score_threshold <= 0 ||
      score_threshold >= 1 || !std::isfinite(nms_threshold) ||
      nms_threshold < 0 || nms_threshold > 1)
    throw std::invalid_argument("E11 requires score in (0,1), NMS in [0,1]");
  for (int i = 0; i < 10; ++i) {
    int grid = i == 9 ? 160 : 80 >> (i / 3),
        channels = i == 9 ? 32 : (i % 3 == 0 ? 4585 : (i % 3 == 1 ? 64 : 32));
    if (outputs[i].size() != static_cast<size_t>(grid) * grid * channels)
      throw std::invalid_argument("Wrong compact YOLOE11 tensor size");
    for (float value : outputs[i])
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite YOLOE11 output");
  }
  float threshold = yolo::raw_logit_threshold(score_threshold);
  std::vector<RawDetection> candidates;
  for (int scale = 0; scale < 3; ++scale) {
    int grid = 80 >> scale, stride = 8 << scale;
    for (int anchor = 0; anchor < grid * grid; ++anchor) {
      const float *cls =
          outputs[3 * scale].data() + static_cast<size_t>(anchor) * 4585;
      int label = static_cast<int>(std::max_element(cls, cls + 4585) - cls);
      if (cls[label] < threshold)
        continue;
      RawDetection detection;
      detection.label = label;
      detection.score = yolo::sigmoid(cls[label]);
      float distances[4];
      yolo::decode_box_dfl(outputs[3 * scale + 1].data() +
                               static_cast<size_t>(anchor) * 64,
                           distances);
      float x = anchor % grid + 0.5f, y = anchor / grid + 0.5f;
      yolo::box_from_distances(x, y, distances, static_cast<float>(stride),
                               &detection.box[0], &detection.box[1],
                               &detection.box[2], &detection.box[3]);
      for (float value : detection.box)
        if (!std::isfinite(value))
          throw std::invalid_argument("Nonfinite decoded E11 box");
      std::copy_n(outputs[3 * scale + 2].data() +
                      static_cast<size_t>(anchor) * 32,
                  32, detection.coefficients.begin());
      candidates.push_back(detection);
    }
  }
  return detail::nms_e11(candidates, nms_threshold);
}

namespace detail {

constexpr int kModelWidth = 640, kClasses = 4585, kMaskChannels = 32,
              kMaskSize = 160;
constexpr std::array<int, 3> kStrides{8, 16, 32};
struct Tensor {
  const float *data;
  int h, w, channels;
  float at(int anchor, int channel) const {
    return data[static_cast<size_t>(anchor) * channels + channel];
  }
};

std::vector<RawDetection> decode(const std::vector<Tensor> &tensors,
                                 float threshold, int max_det,
                                 bool single_label) {
  if (!(threshold > 0.0f && threshold < 1.0f) || max_det < 1 ||
      max_det > 8400 || tensors.size() != 10) {
    throw std::invalid_argument("Invalid threshold, max_det, or output count");
  }
  for (int i = 0; i < 10; ++i) {
    const auto &tensor = tensors[i];
    const int hw = i == 9 ? kMaskSize : kModelWidth / kStrides[i / 3];
    const int channels =
        i == 9 ? kMaskChannels
               : (i % 3 == 0 ? kClasses : (i % 3 == 1 ? 4 : kMaskChannels));
    if (!tensor.data || tensor.h != hw || tensor.w != hw ||
        tensor.channels != channels) {
      throw std::invalid_argument("Invalid raw-v1 output shape");
    }
    const size_t count = static_cast<size_t>(hw) * hw * channels;
    for (size_t j = 0; j < count; ++j) {
      if (!std::isfinite(tensor.data[j])) {
        throw std::invalid_argument("Non-finite model output");
      }
    }
  }

  struct Candidate {
    float value;
    int scale;
    int anchor;
    int label;
    int rank{0};
  };

  std::vector<Candidate> anchors;
  anchors.reserve(8400);
  for (int scale = 0; scale < 3; ++scale) {
    const auto &cls = tensors[scale * 3];
    for (int anchor = 0; anchor < cls.h * cls.w; ++anchor) {
      int label = 0;
      for (int c = 1; c < kClasses; ++c) {
        if (cls.at(anchor, c) > cls.at(anchor, label))
          label = c;
      }
      anchors.push_back({cls.at(anchor, label), scale, anchor, label});
    }
  }

  auto order = [](const Candidate &a, const Candidate &b) {
    if (a.value != b.value)
      return a.value > b.value;
    if (a.scale != b.scale)
      return a.scale < b.scale;
    if (a.anchor != b.anchor)
      return a.anchor < b.anchor;
    return a.label < b.label;
  };
  const size_t anchor_count = std::min<size_t>(max_det, anchors.size());
  std::partial_sort(anchors.begin(), anchors.begin() + anchor_count,
                    anchors.end(), order);
  anchors.resize(anchor_count);

  if (!single_label) {
    std::vector<Candidate> classes;
    classes.reserve(anchor_count * kClasses);
    for (int rank = 0; rank < static_cast<int>(anchor_count); ++rank) {
      const auto &anchor = anchors[rank];
      for (int c = 0; c < kClasses; ++c) {
        classes.push_back({tensors[3 * anchor.scale].at(anchor.anchor, c),
                           anchor.scale, anchor.anchor, c, rank});
      }
    }
    const size_t class_count = std::min<size_t>(max_det, classes.size());
    std::partial_sort(classes.begin(), classes.begin() + class_count,
                      classes.end(),
                      [](const Candidate &a, const Candidate &b) {
                        if (a.value != b.value)
                          return a.value > b.value;
                        if (a.rank != b.rank)
                          return a.rank < b.rank;
                        return a.label < b.label;
                      });
    classes.resize(class_count);
    anchors = std::move(classes);
  }

  const float raw_threshold = std::log(threshold / (1.0f - threshold));
  std::vector<RawDetection> result;
  result.reserve(anchors.size());
  for (const auto &candidate : anchors) {
    if (candidate.value <= raw_threshold)
      continue;
    const int stride = kStrides[candidate.scale];
    const int grid = kModelWidth / stride;
    const float x = candidate.anchor % grid + 0.5f;
    const float y = candidate.anchor / grid + 0.5f;
    const auto &box = tensors[3 * candidate.scale + 1];
    RawDetection detection;
    detection.box = {(x - box.at(candidate.anchor, 0)) * stride,
                     (y - box.at(candidate.anchor, 1)) * stride,
                     (x + box.at(candidate.anchor, 2)) * stride,
                     (y + box.at(candidate.anchor, 3)) * stride};
    detection.score =
        1.0f / (1.0f + std::exp(-std::clamp(candidate.value, -80.0f, 80.0f)));
    detection.label = candidate.label;
    for (int c = 0; c < kMaskChannels; ++c) {
      detection.coefficients[c] =
          tensors[3 * candidate.scale + 2].at(candidate.anchor, c);
    }
    for (float value : detection.box)
      if (!std::isfinite(value))
        throw std::invalid_argument("Decoded box overflow");
    result.push_back(detection);
  }
  return result;
}

} // namespace detail

// Semantic order: (classes, direct LTRB, mask coefficients) for each stride,
// then prototype. Exact vector sizes are checked before any dereference.
std::vector<RawDetection>
decode_e26(const std::array<std::vector<float>, 10> &outputs, float threshold,
           int max_det, bool single_label) {
  std::vector<detail::Tensor> tensors;
  for (int i = 0; i < 10; ++i) {
    int grid = i == 9 ? 160 : (80 >> (i / 3));
    int channels = i == 9 ? 32 : (i % 3 == 0 ? 4585 : (i % 3 == 1 ? 4 : 32));
    if (outputs[i].size() != static_cast<size_t>(grid) * grid * channels)
      throw std::invalid_argument("Wrong compact YOLOE26 tensor size");
    tensors.push_back({outputs[i].data(), grid, grid, channels});
  }
  return detail::decode(tensors, threshold, max_det, single_label);
}

std::array<int, 10> bind_heads(const std::vector<yolo::OutputShape> &shapes,
                               int box_channels) {
  if (shapes.size() != 10 || (box_channels != 4 && box_channels != 64))
    throw std::invalid_argument(
        "YOLOE requires ten heads and explicit LTRB4 or DFL64 geometry.");
  std::array<int, 10> roles{};
  for (int scale = 0; scale < 3; ++scale) {
    int grid = 80 >> scale;
    for (int role = 0; role < 3; ++role) {
      int channels = role == 0 ? 4585 : (role == 1 ? box_channels : 32);
      roles[scale * 3 + role] =
          yolo::find_output_by_shape(shapes, grid, grid, channels);
    }
  }
  roles[9] = yolo::find_output_by_shape(shapes, 160, 160, 32);
  for (int index : roles)
    if (index < 0)
      throw std::invalid_argument(
          "Missing, ambiguous or incompatible YOLOE output role.");
  return roles;
}

Result decode_result(const Heads &heads, const Geometry &geometry,
                     const Config &cfg) {
  validate_config(cfg);
  validate_geometry(geometry);
  if (geometry.protocol != cfg.protocol ||
      geometry.resize_type != cfg.resize_type)
    throw std::invalid_argument(
        "Postprocessing geometry/configuration mismatch");
  auto candidates = cfg.protocol == Protocol::E11
                        ? decode_e11(heads, cfg.score_threshold,
                                     cfg.nms_threshold.value_or(0.7f))
                        : decode_e26(heads, cfg.score_threshold, cfg.max_det,
                                     cfg.single_label);
  auto masks =
      cfg.protocol == Protocol::E11
          ? restore_e11_masks(candidates, heads[9], geometry, cfg.do_morph)
          : restore_e26_masks(candidates, heads[9], geometry);
  Result result;
  result.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i)
    result.push_back({masks[i].box, candidates[i].score, candidates[i].label,
                      std::move(masks[i].mask)});
  return result;
}

namespace {
std::string digest(std::string value) {
  if (value.size() != 64 ||
      !std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
      }))
    throw std::invalid_argument("Expected a 64-digit model SHA-256");
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}
} // namespace

void verify_preflight(const SdkModel &model, const std::string &expected_sha256,
                      const std::string &labels,
                      const rdk::NativeIdentity &actual) {
  const auto expected = digest(expected_sha256);
  const auto detected = rdk::identify_target(actual);
  if (detected.empty() || detected != model.target)
    throw std::invalid_argument(
        "Local board identity is unknown or does not match target " +
        model.target);
  std::ifstream file(model.path, std::ios::binary);
  if (!file || file.peek() == std::ifstream::traits_type::eof())
    throw std::invalid_argument("Missing or empty model file");
  if (rdk::sha256_file(model.path) != expected)
    throw std::invalid_argument(
        "Model SHA-256 mismatch or unreadable model file");
  if (rdk::sha256_file(labels) != kVocabularySha256)
    throw std::invalid_argument(
        "Vocabulary SHA-256 mismatch: expected fixed 4585 ordered PF classes");
}

SdkPreflight make_preflight(std::string expected_sha256, std::string labels) {
  expected_sha256 = digest(std::move(expected_sha256));
  return [expected_sha256, labels](const SdkModel &model) {
    verify_preflight(model, expected_sha256, labels,
                     rdk::read_native_identity());
  };
}

YOLOE::YOLOE(Config config, std::unique_ptr<Runner> runner)
    : config_(config), runner_(std::move(runner)) {
  validate_config(config_);
  if (!runner_ || runner_->protocol() != config_.protocol)
    throw std::invalid_argument(
        "Provide a backend matching the YOLOE protocol");
}
Prepared YOLOE::preprocess(const cv::Mat &image) const {
  auto prepared = prepare_bgr(image, config_.protocol, config_.resize_type);
  return Prepared(to_nv12(prepared.pixels), prepared.geometry, identity_);
}
RawBatch YOLOE::infer(const Prepared &input) {
  if (input.owner_ != identity_)
    throw std::invalid_argument("Prepared input belongs to another task");
  return RawBatch(runner_->infer(input.input_), input.geometry_, identity_);
}
Result YOLOE::postprocess(const RawBatch &raw) const {
  if (raw.owner_ != identity_)
    throw std::invalid_argument("Raw outputs belong to another task");
  return decode_result(raw.outputs_, raw.geometry_, config_);
}
Result YOLOE::predict(const cv::Mat &image) {
  return postprocess(infer(preprocess(image)));
}

#ifdef YOLOE_HAS_SDK

namespace {
void checked(int rc, const char *action) {
  if (rc)
    throw std::runtime_error(std::string(action) +
                             " failed: " + std::to_string(rc));
}
Protocol model_protocol(const SdkModel &model) {
  const bool e11 = model.variant.rfind("11", 0) == 0;
  if (!supported_native_model(model))
    throw std::invalid_argument("Unsupported YOLOE target/variant pair");
#ifdef YOLO_DNN_STACK_X5
  if (model.target != "x5")
    throw std::invalid_argument(
        "Target requires UCP, but this adapter uses X5 SDK");
#else
  if (model.target == "x5")
    throw std::invalid_argument(
        "Target requires X5, but this adapter uses UCP SDK");
#endif
  return e11 ? Protocol::E11 : Protocol::E26;
}
} // namespace

struct SdkRunner::Impl {
  Protocol family;
  yolo::PackedModelOwner packed;
  hbDNNHandle_t model = nullptr;
  yolo::InputPlan plan;
  yolo::Nv12Input input;
  yolo::TaskOutputs output;
  std::array<int, 10> roles{};
};
SdkRunner::SdkRunner(SdkModel spec, SdkPreflight preflight)
    : impl_(std::make_unique<Impl>()) {
  impl_->family = model_protocol(spec);
  if (!preflight)
    throw std::invalid_argument(
        "Provide board/artifact preflight before SDK use");
  preflight(spec);
  std::ifstream file(spec.path, std::ios::binary);
  if (!file || file.peek() == std::ifstream::traits_type::eof())
    throw std::invalid_argument("Missing or empty model file");
  const char *path = spec.path.c_str();
  checked(hbDNNInitializeFromFiles(&impl_->packed.handle, &path, 1),
          "Model initialization");
  if (!impl_->packed.handle)
    throw std::runtime_error("SDK returned a null packed model");
  const char **names = nullptr;
  int count = 0;
  checked(hbDNNGetModelNameList(&names, &count, impl_->packed.handle),
          "Model names");
  if (count != 1 || !names || !names[0] || !names[0][0])
    throw std::invalid_argument("Expected one named model");
  checked(hbDNNGetModelHandle(&impl_->model, impl_->packed.handle, names[0]),
          "Model handle");
  if (!impl_->model)
    throw std::runtime_error("SDK returned a null model handle");
  std::string error;
  impl_->plan = yolo::probe_input_protocol(impl_->model, &error);
  const auto expected = spec.target == "x5" ? yolo::InputProtocol::kPackedNv12
                                            : yolo::InputProtocol::kSplitNv12;
  if (impl_->plan.protocol != expected || impl_->plan.input_h != 640 ||
      impl_->plan.input_w != 640)
    throw std::invalid_argument("Expected target-specific 640x640 NV12: " +
                                error);
  impl_->output.bind(
      impl_->model, 10, [&](const std::vector<yolo::OutputShape> &shapes) {
        impl_->roles =
            bind_heads(shapes, impl_->family == Protocol::E11 ? 64 : 4);
      });
  if (!impl_->input.allocate(impl_->model, impl_->plan))
    throw std::runtime_error("Cannot allocate YOLOE input");
  impl_->output.allocate();
}
SdkRunner::~SdkRunner() = default;
Protocol SdkRunner::protocol() const { return impl_->family; }
Heads SdkRunner::infer(const Nv12Input &input) {
  if (!impl_->input.upload_planes(impl_->plan, input.y.data(), input.y.size(),
                                  input.uv.data(), input.uv.size()))
    throw std::invalid_argument(
        "Invalid NV12 input or failed input cache clean");
  checked(yolo::infer_sync(impl_->output.tensors(), impl_->input.tensors(),
                           impl_->input.input_count(), impl_->model),
          "Inference");
  auto physical = impl_->output.read();
  Heads result;
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = std::move(physical.at(impl_->roles[i]));
  return result;
}

#endif // YOLOE_HAS_SDK

} // namespace yoloe
