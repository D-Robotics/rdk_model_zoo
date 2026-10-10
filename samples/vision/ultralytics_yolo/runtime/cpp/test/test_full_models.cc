// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
//
// Host end-to-end fixture for the five production task models: real OpenCV
// image math plus the fake_dnn_io SDK doubles drive every named model
// through preprocess -> infer -> postprocess and predict with fabricated
// deterministic non-empty outputs (one detection box, one instance mask,
// one 17-keypoint skeleton, one known Top-K, one rotated box), and the
// owned raw stage output of an earlier call must survive a later, different
// inference unchanged. Tensor contents are written by the tensor-aware
// inference double; this is not board execution evidence.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "backend.hpp"
#include "yolo.hpp"

namespace {

std::atomic<int> infer_calls(0);
// Installed by the active scenario; writes the fabricated outputs into the
// freshly allocated SDK tensors of the current inference call.
std::function<void(hbDNNTensor *, int)> fill_outputs;
std::vector<hbDNNTensorProperties> input_metadata;
std::vector<hbDNNTensorProperties> output_metadata;

#define expect(v)                                                              \
  do {                                                                         \
    if (!(v)) throw std::runtime_error(#v);                                    \
  } while (0)

bool near_value(float value, double expected, double epsilon = 1e-5) {
  return std::fabs(static_cast<double>(value) - expected) < epsilon;
}

hbDNNTensorProperties packed_input(int h, int w) {
  hbDNNTensorProperties p{};
  p.tensorType = HB_DNN_IMG_TYPE_NV12;
  p.quantiType = NONE;
  p.tensorLayout = HB_DNN_LAYOUT_NCHW;
  p.validShape = {4, {1, 3, h, w}};
  p.alignedShape = p.validShape;
  p.alignedByteSize = h * w * 3 / 2;
  return p;
}

hbDNNTensorProperties byte_plane(int tensor_type, int rows, int cols,
                                 int channels) {
  hbDNNTensorProperties p{};
  p.tensorType = tensor_type;
  p.quantiType = NONE;
  p.tensorLayout = HB_DNN_LAYOUT_NHWC;
  p.validShape = {4, {1, rows, cols, channels}};
  p.alignedShape = p.validShape;
  p.stride[3] = 1;
  p.stride[2] = channels;
  p.stride[1] = cols * channels;
  p.stride[0] = rows * cols * channels;
  p.alignedByteSize = p.stride[0];
  return p;
}

hbDNNTensorProperties float_map(int h, int w, int c) {
  hbDNNTensorProperties p{};
  p.tensorType = HB_DNN_TENSOR_TYPE_F32;
  p.quantiType = NONE;
  p.tensorLayout = HB_DNN_LAYOUT_NHWC;
  p.validShape = {4, {1, h, w, c}};
  p.alignedShape = p.validShape;
  p.stride[3] = 4;
  p.stride[2] = c * 4;
  p.stride[1] = w * c * 4;
  p.stride[0] = h * w * c * 4;
  p.alignedByteSize = p.stride[0];
  return p;
}

void install_input(int h, int w) {
#ifdef YOLO_DNN_STACK_X5
  input_metadata = {packed_input(h, w)};
#else
  input_metadata = {byte_plane(HB_DNN_TENSOR_TYPE_U8, h, w, 1),
                    byte_plane(HB_DNN_TENSOR_TYPE_U8, h / 2, w / 2, 2)};
#endif
}

void install_stride_outputs(int cls_channels, int extra_channels,
                            bool with_prototype) {
  // Per stride: class map (80 detect/segment, 1 pose), direct-LTRB box map
  // and one task extra (mask coefficients / keypoints); segment adds the
  // prototype. Detect passes extra_channels == 0 and stops after the box.
  output_metadata.clear();
  for (int stride : {8, 16, 32}) {
    const int grid = 640 / stride;
    output_metadata.push_back(float_map(grid, grid, cls_channels));
    output_metadata.push_back(float_map(grid, grid, 4));
    if (extra_channels > 0)
      output_metadata.push_back(float_map(grid, grid, extra_channels));
  }
  if (with_prototype)
    output_metadata.push_back(float_map(160, 160, 32));
}

void install_obb_outputs(int classes) {
  // Per stride: class map, direct-LTRB box map and 1-channel angle map.
  output_metadata.clear();
  for (int stride : {8, 16, 32}) {
    const int grid = 640 / stride;
    output_metadata.push_back(float_map(grid, grid, classes));
    output_metadata.push_back(float_map(grid, grid, 4));
    output_metadata.push_back(float_map(grid, grid, 1));
  }
}

float *tensor_data(hbDNNTensor *tensor) {
  return static_cast<float *>(YOLO_SYS_MEM(*tensor)->virAddr);
}

void fill_negative(hbDNNTensor *outputs, int count) {
  for (int i = 0; i < count; ++i) {
    float *data = tensor_data(outputs + i);
    const int values = outputs[i].properties.alignedByteSize / 4;
    for (int j = 0; j < values; ++j) data[j] = -100.0f;
  }
}

int output_index(hbDNNTensor *outputs, int count, int h, int w, int c) {
  for (int i = 0; i < count; ++i) {
    const hbDNNTensorShape &shape = outputs[i].properties.validShape;
    if (shape.dimensionSize[1] == h && shape.dimensionSize[2] == w &&
        shape.dimensionSize[3] == c)
      return i;
  }
  throw std::runtime_error("fixture output role not found");
}

// The single active anchor is stride-8 cell (row 40, col 40): grid centre
// (40.5, 40.5) * 8 = 324, and with LTRB distances 2 the model box spans
// (308, 308) to (340, 340) exactly.
constexpr int kGrid8 = 80;
constexpr int kAnchor = 40 * kGrid8 + 40;
constexpr double kCenter = 40.5;
constexpr double kBoxLo = (kCenter - 2.0) * 8.0;
constexpr double kBoxHi = (kCenter + 2.0) * 8.0;
const double kSigmoid4 = 1.0 / (1.0 + std::exp(-4.0));
const double kSigmoid5 = 1.0 / (1.0 + std::exp(-5.0));

// First inference of a scenario uses logit 4, every later one logit 5.
float scenario_logit(int scenario_base) {
  return infer_calls.load() - scenario_base == 1 ? 4.0f : 5.0f;
}

std::vector<uint8_t> constant_bgr(int rows, int cols) {
  const cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(60, 120, 200));
  const uint8_t *begin = image.ptr<uint8_t>();
  return std::vector<uint8_t>(begin, begin + static_cast<size_t>(rows) * cols * 3);
}

void check_box(float x1, float y1, float x2, float y2) {
  expect(near_value(x1, kBoxLo, 1e-3));
  expect(near_value(y1, kBoxLo, 1e-3));
  expect(near_value(x2, kBoxHi, 1e-3));
  expect(near_value(y2, kBoxHi, 1e-3));
}

}  // namespace

// SDK hooks at global scope: the production sources reference these symbols
// from other translation units.
int hbDNNInitializeFromFiles(void **handle, const char **, int) {
  *handle = reinterpret_cast<void *>(1);
  return 0;
}
int hbDNNGetModelNameList(const char ***names, int *count, void *) {
  static const char *model_name = "fixture_model";
  *names = &model_name;
  *count = 1;
  return 0;
}
int hbDNNGetModelHandle(void **out, void *model, const char *) {
  *out = model;
  return 0;
}
int hbDNNGetInputCount(int32_t *n, hbDNNHandle_t) {
  *n = static_cast<int32_t>(input_metadata.size());
  return 0;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                  int i) {
  if (i < 0 || i >= static_cast<int>(input_metadata.size())) return -1;
  *p = input_metadata[i];
  return 0;
}
int hbDNNGetOutputCount(int32_t *n, hbDNNHandle_t) {
  *n = static_cast<int32_t>(output_metadata.size());
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                   int i) {
  if (i < 0 || i >= static_cast<int>(output_metadata.size())) return -1;
  *p = output_metadata[i];
  return 0;
}
int test_allocate(TestMemory *m, int n) {
  if (n <= 0) return -1;
  m->virAddr = std::calloc(static_cast<size_t>(n), 1);
  m->memSize = n;
  return m->virAddr == nullptr ? -1 : 0;
}
int test_free(TestMemory *m) {
  std::free(m->virAddr);
  m->virAddr = nullptr;
  return 0;
}
int test_flush(TestMemory *, int) { return 0; }
int test_infer(void **task, hbDNNTensor *outputs, hbDNNTensor *inputs) {
  (void)inputs;
  if (!fill_outputs) return -1;
  *task = reinterpret_cast<void *>(1);
  ++infer_calls;
  fill_outputs(outputs, static_cast<int>(output_metadata.size()));
  return 0;
}
int test_wait(void *) { return 0; }
int test_release(void *) { return 0; }
int test_submit(void *, hbUCPSchedParam *) { return 0; }
int hbDNNRelease(void *) { return 0; }

namespace {

void run_detect() {
  install_input(640, 640);
  install_stride_outputs(80, 0, false);
  const int base = infer_calls.load();
  fill_outputs = [&](hbDNNTensor *outputs, int count) {
    fill_negative(outputs, count);
    const int cls = output_index(outputs, count, kGrid8, kGrid8, 80);
    const int box = output_index(outputs, count, kGrid8, kGrid8, 4);
    tensor_data(outputs + cls)[kAnchor * 80 + 3] = scenario_logit(base);
    float *ltrb = tensor_data(outputs + box) + kAnchor * 4;
    ltrb[0] = ltrb[1] = ltrb[2] = ltrb[3] = 2.0f;
  };

  yolo::YoloDetect::Config config;
  config.model_path = "detect-fixture.bin";
  yolo::YoloDetect model(config);
  expect(model.input_h() == 640 && model.input_w() == 640);

  yolo::YoloDetect::Input input{constant_bgr(640, 640), 640, 640};
  const yolo::YoloDetect::Prepared prepared = model.preprocess(input);
  expect(prepared.y_plane.size() == 640u * 640u);
  expect(prepared.uv_plane.size() == 640u * 640u / 2u);

  const yolo::YoloDetect::RawResult raw_first = model.infer(prepared);
  const yolo::YoloDetect::Result first = model.postprocess(raw_first);
  expect(first.detections.size() == 1);
  expect(first.detections[0].class_id == 3);
  expect(near_value(first.detections[0].score, kSigmoid4));
  check_box(first.detections[0].x1, first.detections[0].y1,
            first.detections[0].x2, first.detections[0].y2);

  // Owned raw stage output survives a later, different inference.
  const yolo::YoloDetect::RawResult raw_second = model.infer(prepared);
  const yolo::YoloDetect::Result first_again = model.postprocess(raw_first);
  expect(first_again.detections.size() == 1);
  expect(near_value(first_again.detections[0].score, kSigmoid4));
  expect(first_again.detections[0].class_id == 3);
  const yolo::YoloDetect::Result second = model.postprocess(raw_second);
  expect(second.detections.size() == 1);
  expect(near_value(second.detections[0].score, kSigmoid5));

  const yolo::YoloDetect::Prediction prediction = model.predict(input);
  expect(prediction.result.detections.size() == 1);
  expect(near_value(prediction.result.detections[0].score, kSigmoid5));
  expect(prediction.model_rows == 640 && prediction.model_cols == 640);
  // The detect renderer draws on the caller's source image, so predict does
  // not carry a model frame.
  expect(prediction.model_frame_bgr.empty());
}

// Nonuniform prototype: independent x and y block terms of equal amplitude
// with different periods (8 cells in y, 12 in x), giving logits {20, 0, -20}.
// After sigmoid, resize and threshold the mask is a quadrant block pattern
// with differing rows and columns and both byte values. The 48-px x period
// deliberately does not divide the 32-byte crop stride, so a flat
// contiguous copy across ROI rows produces different bytes than the
// row-addressed crop.
float proto_value(int r, int c) {
  return (((r % 8) < 4) ? 10.0f : -10.0f) +
         (((c % 12) < 6) ? 10.0f : -10.0f);
}

// Expected full-frame binary mask from the fabricated prototype, through the
// same OpenCV chain the model runs (sigmoid, INTER_LINEAR resize to the
// input size, 0.5 threshold, x255). The returned crop must equal this
// parent frame row by row at the detection box.
cv::Mat expected_binary_mask() {
  cv::Mat low(160, 160, CV_32F);
  for (int r = 0; r < 160; ++r)
    for (int c = 0; c < 160; ++c) low.at<float>(r, c) = proto_value(r, c);
  cv::Mat sigmoid_mask;
  cv::exp(-low, sigmoid_mask);
  sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);
  cv::Mat resized_mask;
  cv::resize(sigmoid_mask, resized_mask, cv::Size(640, 640), 0, 0,
             cv::INTER_LINEAR);
  cv::Mat binary_mask;
  cv::threshold(resized_mask, binary_mask, 0.5, 1.0, cv::THRESH_BINARY);
  binary_mask.convertTo(binary_mask, CV_8U, 255);
  return binary_mask;
}

void run_segment() {
  install_input(640, 640);
  install_stride_outputs(80, 32, true);
  const int base = infer_calls.load();
  fill_outputs = [&](hbDNNTensor *outputs, int count) {
    fill_negative(outputs, count);
    const int cls = output_index(outputs, count, kGrid8, kGrid8, 80);
    const int box = output_index(outputs, count, kGrid8, kGrid8, 4);
    const int mce = output_index(outputs, count, kGrid8, kGrid8, 32);
    const int proto = output_index(outputs, count, 160, 160, 32);
    tensor_data(outputs + cls)[kAnchor * 80 + 5] = scenario_logit(base);
    float *ltrb = tensor_data(outputs + box) + kAnchor * 4;
    ltrb[0] = ltrb[1] = ltrb[2] = ltrb[3] = 2.0f;
    float *coefficients = tensor_data(outputs + mce) + kAnchor * 32;
    for (int k = 0; k < 32; ++k) coefficients[k] = k == 0 ? 1.0f : 0.0f;
    float *prototype = tensor_data(outputs + proto);
    for (int r = 0; r < 160; ++r)
      for (int c = 0; c < 160; ++c)
        prototype[(r * 160 + c) * 32] = proto_value(r, c);
  };

  yolo::YoloSegment::Config config;
  config.model_path = "segment-fixture.bin";
  yolo::YoloSegment model(config);

  yolo::YoloSegment::Input input{constant_bgr(640, 640), 640, 640};
  const yolo::YoloSegment::Prepared prepared = model.preprocess(input);
  const yolo::YoloSegment::RawResult raw_first = model.infer(prepared);
  const yolo::YoloSegment::Result first = model.postprocess(raw_first);
  expect(first.detections.size() == 1);
  expect(first.detections[0].class_id == 5);
  expect(near_value(first.detections[0].score, kSigmoid4));
  check_box(first.detections[0].x1, first.detections[0].y1,
            first.detections[0].x2, first.detections[0].y2);
  // The narrow (32x32) crop sits at a nonzero offset (308, 308) inside the
  // full-frame mask; every returned row must equal the parent ROI row.
  const yolo::YoloSegment::MaskPlane &mask = first.detections[0].mask;
  expect(mask.x == 308 && mask.y == 308 && mask.rows == 32 && mask.cols == 32);
  expect(mask.bytes.size() == 32u * 32u);
  const cv::Mat parent = expected_binary_mask();
  for (int r = 0; r < 32; ++r) {
    const uint8_t *expected_row = parent.ptr<uint8_t>(308 + r) + 308;
    expect(std::equal(expected_row, expected_row + 32,
                      mask.bytes.begin() + r * 32));
  }
  // The shape is row-dependent (proto y-period 8): the ROI spans several
  // y-bands, so the returned rows cannot all be the same pattern. Row 4
  // (parent y=312) lies in a -10 band (all-zero row); row 16 (parent y=324)
  // lies in a +10 band, where the +10 column blocks light up.
  std::vector<std::vector<uint8_t>> distinct_rows;
  for (int r = 0; r < 32; ++r) {
    std::vector<uint8_t> row(mask.bytes.begin() + r * 32,
                             mask.bytes.begin() + (r + 1) * 32);
    if (std::find(distinct_rows.begin(), distinct_rows.end(), row) ==
        distinct_rows.end())
      distinct_rows.push_back(std::move(row));
  }
  expect(distinct_rows.size() >= 2);
  expect(std::all_of(mask.bytes.begin() + 4 * 32, mask.bytes.begin() + 5 * 32,
                     [](uint8_t value) { return value == 0; }));
  expect(mask.bytes[16 * 32] == 255);
  // Nonuniform result: both byte values occur, and a block-interior byte far
  // from any resize blend edge decodes analytically: proto cell (79, 79)
  // gives logit -20 -> 0.
  bool has_zero = false;
  bool has_255 = false;
  for (uint8_t value : mask.bytes) {
    has_zero = has_zero || value == 0;
    has_255 = has_255 || value == 255;
  }
  expect(has_zero && has_255);
  expect(mask.bytes[8 * 32 + 8] == 0);

  const yolo::YoloSegment::RawResult raw_second = model.infer(prepared);
  const yolo::YoloSegment::Result first_again = model.postprocess(raw_first);
  expect(first_again.detections.size() == 1);
  expect(first_again.detections[0].mask.bytes == mask.bytes);
  expect(near_value(first_again.detections[0].score, kSigmoid4));
  const yolo::YoloSegment::Result second = model.postprocess(raw_second);
  expect(second.detections.size() == 1);
  expect(near_value(second.detections[0].score, kSigmoid5));
  expect(second.detections[0].mask.bytes == mask.bytes);

  const yolo::YoloSegment::Prediction prediction = model.predict(input);
  expect(prediction.result.detections.size() == 1);
  expect(prediction.result.detections[0].mask.bytes == mask.bytes);
  expect(near_value(prediction.result.detections[0].score, kSigmoid5));
}

void run_pose() {
  install_input(640, 640);
  install_stride_outputs(1, 51, false);
  const int base = infer_calls.load();
  fill_outputs = [&](hbDNNTensor *outputs, int count) {
    fill_negative(outputs, count);
    const int cls = output_index(outputs, count, kGrid8, kGrid8, 1);
    const int box = output_index(outputs, count, kGrid8, kGrid8, 4);
    const int kpt = output_index(outputs, count, kGrid8, kGrid8, 51);
    tensor_data(outputs + cls)[kAnchor] = scenario_logit(base);
    float *ltrb = tensor_data(outputs + box) + kAnchor * 4;
    ltrb[0] = ltrb[1] = ltrb[2] = ltrb[3] = 2.0f;
    float *keypoints = tensor_data(outputs + kpt) + kAnchor * 51;
    for (int k = 0; k < 17; ++k) {
      keypoints[3 * k + 0] = 0.5f * k;        // x: (k*0.5 + 40.5) * 8
      keypoints[3 * k + 1] = 2.0f - 0.25f * k; // y: (2 - 0.25k + 40.5) * 8
      keypoints[3 * k + 2] = 4.0f;             // raw confidence logit
    }
  };

  yolo::YoloPose::Config config;
  config.model_path = "pose-fixture.bin";
  yolo::YoloPose model(config);

  yolo::YoloPose::Input input{constant_bgr(640, 640), 640, 640};
  const yolo::YoloPose::Prepared prepared = model.preprocess(input);
  const yolo::YoloPose::RawResult raw_first = model.infer(prepared);
  const yolo::YoloPose::Result first = model.postprocess(raw_first);
  expect(first.detections.size() == 1);
  expect(near_value(first.detections[0].score, kSigmoid4));
  check_box(first.detections[0].x1, first.detections[0].y1,
            first.detections[0].x2, first.detections[0].y2);
  for (int k = 0; k < 17; ++k) {
    const yolo::YoloPose::Keypoint &keypoint = first.detections[0].keypoints[k];
    expect(near_value(keypoint.x, (0.5 * k + kCenter) * 8.0, 1e-3));
    expect(near_value(keypoint.y, (2.0 - 0.25 * k + kCenter) * 8.0, 1e-3));
    expect(near_value(keypoint.score, 4.0, 1e-6));
  }

  const yolo::YoloPose::RawResult raw_second = model.infer(prepared);
  const yolo::YoloPose::Result first_again = model.postprocess(raw_first);
  expect(first_again.detections.size() == 1);
  expect(near_value(first_again.detections[0].score, kSigmoid4));
  expect(near_value(first_again.detections[0].keypoints[7].x,
                    (0.5 * 7 + kCenter) * 8.0, 1e-3));
  const yolo::YoloPose::Result second = model.postprocess(raw_second);
  expect(second.detections.size() == 1);
  expect(near_value(second.detections[0].score, kSigmoid5));

  const yolo::YoloPose::Prediction prediction = model.predict(input);
  expect(prediction.result.detections.size() == 1);
  expect(near_value(prediction.result.detections[0].score, kSigmoid5));
  expect(near_value(prediction.result.detections[0].keypoints[0].x,
                    (0.0 + kCenter) * 8.0, 1e-3));
}

void run_classify() {
  install_input(224, 224);
  output_metadata.clear();
  {
    hbDNNTensorProperties p = float_map(1, 1, 1000);
    output_metadata.push_back(p);
  }
  const int base = infer_calls.load();
  fill_outputs = [&](hbDNNTensor *outputs, int count) {
    expect(count == 1);
    float *logits = tensor_data(outputs);
    for (int i = 0; i < 1000; ++i) logits[i] = -10.0f;
    // First call lifts class 7; later calls lift class 9 instead.
    if (infer_calls.load() - base == 1) {
      logits[7] = 10.0f;
    } else {
      logits[9] = 11.0f;
    }
  };

  yolo::YoloClassify::Config config;
  config.model_path = "classify-fixture.bin";
  yolo::YoloClassify model(config);
  expect(model.input_h() == 224 && model.input_w() == 224);

  // Softmax over one 10 and 999 entries of -10: the winner is
  // 1 / (1 + 999*exp(-20)), the tied losers follow in id order. The winner
  // epsilon covers storage as a 32-bit float in ClassificationScore.
  const double winner = 1.0 / (1.0 + 999.0 * std::exp(-20.0));
  const double loser = std::exp(-20.0) / (1.0 + 999.0 * std::exp(-20.0));

  yolo::YoloClassify::Input input{constant_bgr(224, 224), 224, 224};
  const yolo::YoloClassify::Prepared prepared = model.preprocess(input);
  expect(prepared.y_plane.size() == 224u * 224u);
  const yolo::YoloClassify::RawResult raw_first = model.infer(prepared);
  const yolo::YoloClassify::Result first = model.postprocess(raw_first);
  expect(first.topk.size() == 5);
  expect(first.topk[0].id == 7);
  expect(near_value(first.topk[0].probability, winner, 1e-6));
  for (int k = 1; k < 5; ++k) {
    expect(first.topk[k].id == k - 1);
    expect(near_value(first.topk[k].probability, loser, 1e-15));
  }

  const yolo::YoloClassify::RawResult raw_second = model.infer(prepared);
  const yolo::YoloClassify::Result first_again = model.postprocess(raw_first);
  expect(first_again.topk.size() == 5);
  expect(first_again.topk[0].id == 7);
  const yolo::YoloClassify::Result second = model.postprocess(raw_second);
  expect(second.topk.size() == 5);
  expect(second.topk[0].id == 9);

  const yolo::YoloClassify::Prediction prediction = model.predict(input);
  expect(prediction.result.topk.size() == 5);
  expect(prediction.result.topk[0].id == 9);
}

void run_obb() {
  install_input(640, 640);
  install_obb_outputs(15);
  const int base = infer_calls.load();
  fill_outputs = [&](hbDNNTensor *outputs, int count) {
    fill_negative(outputs, count);
    const int cls = output_index(outputs, count, kGrid8, kGrid8, 15);
    const int box = output_index(outputs, count, kGrid8, kGrid8, 4);
    const int angle = output_index(outputs, count, kGrid8, kGrid8, 1);
    tensor_data(outputs + cls)[kAnchor * 15 + 2] = scenario_logit(base);
    float *ltrb = tensor_data(outputs + box) + kAnchor * 4;
    ltrb[0] = ltrb[1] = ltrb[2] = ltrb[3] = 2.0f;
    tensor_data(outputs + angle)[kAnchor] = 0.0f;
  };

  yolo::YoloObb::Config config;
  config.model_path = "obb-fixture.bin";
  yolo::YoloObb model(config);
  expect(model.input_h() == 640 && model.input_w() == 640);
  expect(model.classes() == 15);

  yolo::YoloObb::Input input{constant_bgr(640, 640), 640, 640};
  const yolo::YoloObb::Prepared prepared = model.preprocess(input);
  // Square source at the model size: the realized letterbox ratio is
  // exactly 1, so the restore transform stays identity.
  expect(prepared.transform.scale_x == 1.0f &&
         prepared.transform.scale_y == 1.0f);
  expect(prepared.transform.shift_x == 0 && prepared.transform.shift_y == 0);

  const yolo::YoloObb::RawResult raw_first = model.infer(prepared);
  const yolo::YoloObb::Result first = model.postprocess(raw_first);
  expect(first.detections.size() == 1);
  expect(first.detections[0].class_id == 2);
  expect(near_value(first.detections[0].score, kSigmoid4));
  // Grid centre (40.5, 40.5) with symmetric LTRB distances 2 at stride 8
  // gives a 32x32 axis-aligned box centred at (324, 324).
  const yolo::RotatedBox &rotated = first.detections[0].box;
  expect(near_value(rotated.cx, 324.0, 1e-3));
  expect(near_value(rotated.cy, 324.0, 1e-3));
  expect(near_value(rotated.width, 32.0, 1e-3));
  expect(near_value(rotated.height, 32.0, 1e-3));
  expect(near_value(rotated.angle_rad, 0.0, 1e-6));

  // Owned raw stage output survives a later, different inference.
  const yolo::YoloObb::RawResult raw_second = model.infer(prepared);
  const yolo::YoloObb::Result first_again = model.postprocess(raw_first);
  expect(first_again.detections.size() == 1);
  expect(near_value(first_again.detections[0].score, kSigmoid4));
  expect(near_value(first_again.detections[0].box.cx, 324.0, 1e-3));
  const yolo::YoloObb::Result second = model.postprocess(raw_second);
  expect(second.detections.size() == 1);
  expect(near_value(second.detections[0].score, kSigmoid5));

  const yolo::YoloObb::Prediction prediction = model.predict(input);
  expect(prediction.result.detections.size() == 1);
  expect(near_value(prediction.result.detections[0].score, kSigmoid5));
  expect(prediction.model_rows == 640 && prediction.model_cols == 640);

  // The OBB decoder validates only the values it consumes: nonfinite box
  // and angle in an unselected background cell must not fail the frame.
  const float kNaN = std::nanf("");
  fill_outputs = [&](hbDNNTensor *outputs, int count) {
    fill_negative(outputs, count);
    const int cls = output_index(outputs, count, kGrid8, kGrid8, 15);
    const int box = output_index(outputs, count, kGrid8, kGrid8, 4);
    const int angle = output_index(outputs, count, kGrid8, kGrid8, 1);
    tensor_data(outputs + cls)[kAnchor * 15 + 2] = 5.0f;
    float *ltrb = tensor_data(outputs + box) + kAnchor * 4;
    ltrb[0] = ltrb[1] = ltrb[2] = ltrb[3] = 2.0f;
    tensor_data(outputs + angle)[kAnchor] = 0.0f;
    const int background = kAnchor + 1;  // never selected: score stays -100
    float *bg_ltrb = tensor_data(outputs + box) + background * 4;
    for (int i = 0; i < 4; ++i) bg_ltrb[i] = kNaN;
    tensor_data(outputs + angle)[background] = kNaN;
  };
  const yolo::YoloObb::RawResult raw_background = model.infer(prepared);
  const yolo::YoloObb::Result background = model.postprocess(raw_background);
  expect(background.detections.size() == 1);
  expect(near_value(background.detections[0].box.cx, 324.0, 1e-3));

  // A nonfinite box at the selected cell fails the frame explicitly.
  fill_outputs = [&](hbDNNTensor *outputs, int count) {
    fill_negative(outputs, count);
    const int cls = output_index(outputs, count, kGrid8, kGrid8, 15);
    const int box = output_index(outputs, count, kGrid8, kGrid8, 4);
    tensor_data(outputs + cls)[kAnchor * 15 + 2] = 5.0f;
    float *ltrb = tensor_data(outputs + box) + kAnchor * 4;
    ltrb[0] = kNaN;
    ltrb[1] = ltrb[2] = ltrb[3] = 2.0f;
  };
  const yolo::YoloObb::RawResult raw_bad_box = model.infer(prepared);
  bool rejected_box = false;
  try {
    (void)model.postprocess(raw_bad_box);
  } catch (const std::exception &) {
    rejected_box = true;
  }
  expect(rejected_box);

  // A nonfinite angle at the selected cell fails the frame explicitly.
  fill_outputs = [&](hbDNNTensor *outputs, int count) {
    fill_negative(outputs, count);
    const int cls = output_index(outputs, count, kGrid8, kGrid8, 15);
    const int box = output_index(outputs, count, kGrid8, kGrid8, 4);
    const int angle = output_index(outputs, count, kGrid8, kGrid8, 1);
    tensor_data(outputs + cls)[kAnchor * 15 + 2] = 5.0f;
    float *ltrb = tensor_data(outputs + box) + kAnchor * 4;
    ltrb[0] = ltrb[1] = ltrb[2] = ltrb[3] = 2.0f;
    tensor_data(outputs + angle)[kAnchor] = kNaN;
  };
  const yolo::YoloObb::RawResult raw_bad_angle = model.infer(prepared);
  bool rejected_angle = false;
  try {
    (void)model.postprocess(raw_bad_angle);
  } catch (const std::exception &) {
    rejected_angle = true;
  }
  expect(rejected_angle);
}

}  // namespace

int main() {
  try {
    run_detect();
    run_segment();
    run_pose();
    run_classify();
    run_obb();
  } catch (const std::exception &error) {
    std::printf("test_full_models: FAIL %s\n", error.what());
    return 1;
  }
  std::printf("test_full_models: OK (%d inferences)\n", infer_calls.load());
  return 0;
}
