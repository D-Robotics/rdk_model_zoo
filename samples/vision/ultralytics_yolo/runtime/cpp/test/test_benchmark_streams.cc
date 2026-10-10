// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
//
// Host regression test for the benchmark resource shape: the process holds
// exactly --pipeline-streams runtime contexts (the model main constructed is
// reused as stream 0) and performs exactly one validation predict before the
// timed rounds, plus the benchmark option surface (this fixture links the
// parser in src/cli.cpp). Compiles the production task/cli/backend sources
// against the fake_dnn_io stack doubles plus real OpenCV. The detect
// scenarios leave the preallocated -100 outputs (zero detections); one
// scenario per remaining task drives that task's benchmark wrapper with
// fabricated deterministic non-empty outputs and checks context count,
// inference count, context release and the task JSON record.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "backend.hpp"
#include "cli.hpp"

namespace {

std::atomic<int> contexts_created(0);
std::atomic<int> infer_calls(0);
std::atomic<int> model_releases(0);
std::vector<hbDNNTensorProperties> input_metadata;
std::vector<hbDNNTensorProperties> output_metadata;
// Installed by the active scenario; writes fabricated outputs into the
// freshly allocated SDK tensors of the current inference call. When unset
// the -100 prefill of test_allocate stays (zero detections).
std::function<void(hbDNNTensor*, int)> fill_outputs;

#define expect(v)                                                              \
  do {                                                                         \
    if (!(v)) throw std::runtime_error(#v);                                    \
  } while (0)

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

void install_model() {
  // Direct-LTRB detect geometry: 640x640 input, three scales of
  // 80-channel class maps plus 4-channel box maps.
  install_stride_outputs(80, 0, false);
  install_input(640, 640);
}

void install_classify_model() {
  output_metadata.clear();
  output_metadata.push_back(float_map(1, 1, 1000));
  install_input(224, 224);
}

void install_obb_model() {
  // Per stride: 15-class map, direct-LTRB box map, 1-channel angle map.
  output_metadata.clear();
  for (int stride : {8, 16, 32}) {
    const int grid = 640 / stride;
    output_metadata.push_back(float_map(grid, grid, 15));
    output_metadata.push_back(float_map(grid, grid, 4));
    output_metadata.push_back(float_map(grid, grid, 1));
  }
  install_input(640, 640);
}

float* tensor_data(hbDNNTensor* tensor) {
  return static_cast<float*>(YOLO_SYS_MEM(*tensor)->virAddr);
}

int output_index(hbDNNTensor* outputs, int count, int h, int w, int c) {
  for (int i = 0; i < count; ++i) {
    const hbDNNTensorShape& shape = outputs[i].properties.validShape;
    if (shape.dimensionSize[1] == h && shape.dimensionSize[2] == w &&
        shape.dimensionSize[3] == c)
      return i;
  }
  throw std::runtime_error("fixture output role not found");
}

// The single active anchor is stride-8 cell (row 40, col 40), the same cell
// test_full_models drives: symmetric LTRB distances 2 give a 32x32 box
// centred at (324, 324) in the 640x640 model frame.
constexpr int kGrid8 = 80;
constexpr int kAnchor = 40 * kGrid8 + 40;
constexpr float kAnchorLogit = 4.0f;

// Shared body of the stride-task fills: -100 everywhere, then the anchor
// cell gets an active class logit, symmetric LTRB distances and (when the
// task carries one) an extras vector with only the first channel set.
void fill_anchor_cell(hbDNNTensor* outputs, int count, int cls_channels,
                      int extra_channels, int active_class) {
  const int cls = output_index(outputs, count, kGrid8, kGrid8, cls_channels);
  const int box = output_index(outputs, count, kGrid8, kGrid8, 4);
  const int extra = extra_channels > 0
                        ? output_index(outputs, count, kGrid8, kGrid8,
                                       extra_channels)
                        : -1;
  for (int i = 0; i < count; ++i) {
    const int values = outputs[i].properties.alignedByteSize / 4;
    for (int j = 0; j < values; ++j) tensor_data(outputs + i)[j] = -100.0f;
  }
  tensor_data(outputs + cls)[kAnchor * cls_channels + active_class] =
      kAnchorLogit;
  float* ltrb = tensor_data(outputs + box) + kAnchor * 4;
  ltrb[0] = ltrb[1] = ltrb[2] = ltrb[3] = 2.0f;
  if (extra_channels > 0) {
    float* extras = tensor_data(outputs + extra) + kAnchor * extra_channels;
    for (int k = 0; k < extra_channels; ++k) extras[k] = k == 0 ? 1.0f : 0.0f;
  }
}

void fill_segment_outputs() {
  fill_outputs = [](hbDNNTensor* outputs, int count) {
    fill_anchor_cell(outputs, count, 80, 32, 5);
    // Uniform zero prototype: sigmoid 0.5 stays under the 0.5 mask
    // threshold, so the crop is a valid deterministic all-zero plane.
    const int proto = output_index(outputs, count, 160, 160, 32);
    float* values = tensor_data(outputs + proto);
    const int cells = 160 * 160 * 32;
    for (int i = 0; i < cells; ++i) values[i] = 0.0f;
  };
}

void fill_pose_outputs() {
  fill_outputs = [](hbDNNTensor* outputs, int count) {
    fill_anchor_cell(outputs, count, 1, 51, 0);
    const int kpt = output_index(outputs, count, kGrid8, kGrid8, 51);
    float* keypoints = tensor_data(outputs + kpt) + kAnchor * 51;
    for (int k = 0; k < 17; ++k) {
      keypoints[3 * k + 0] = 0.5f * k;
      keypoints[3 * k + 1] = 2.0f - 0.25f * k;
      keypoints[3 * k + 2] = 4.0f;
    }
  };
}

void fill_classify_outputs() {
  fill_outputs = [](hbDNNTensor* outputs, int) {
    float* logits = tensor_data(outputs);
    for (int i = 0; i < 1000; ++i) logits[i] = -10.0f;
    logits[7] = 10.0f;
  };
}

void fill_obb_outputs() {
  fill_outputs = [](hbDNNTensor* outputs, int count) {
    fill_anchor_cell(outputs, count, 15, 0, 2);
    const int angle = output_index(outputs, count, kGrid8, kGrid8, 1);
    tensor_data(outputs + angle)[kAnchor] = 0.0f;
  };
}

}  // namespace

// SDK hooks at global scope: the production sources reference these symbols
// from other translation units.
int hbDNNInitializeFromFiles(void** handle, const char**, int) {
  *handle = reinterpret_cast<void*>(1);
  ++contexts_created;
  return 0;
}
int hbDNNGetModelNameList(const char*** names, int* count, void*) {
  static const char* model_name = "yolo26n_detect_bayese_640x640_nv12";
  *names = &model_name;
  *count = 1;
  return 0;
}
int hbDNNGetModelHandle(void** out, void* model, const char*) {
  *out = model;
  return 0;
}
int hbDNNGetInputCount(int32_t* n, hbDNNHandle_t) {
  *n = static_cast<int32_t>(input_metadata.size());
  return 0;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties* p, hbDNNHandle_t,
                                  int i) {
  if (i < 0 || i >= static_cast<int>(input_metadata.size())) return -1;
  *p = input_metadata[i];
  return 0;
}
int hbDNNGetOutputCount(int32_t* n, hbDNNHandle_t) {
  *n = static_cast<int32_t>(output_metadata.size());
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties* p, hbDNNHandle_t,
                                   int i) {
  *p = output_metadata.at(i);
  return 0;
}
int test_allocate(TestMemory* m, int n) {
  if (n <= 0) return -1;
  m->virAddr = std::calloc(static_cast<size_t>(n), 1);
  m->memSize = n;
  if (m->virAddr == nullptr) return -1;
  // Every output byte pattern is a finite float far below any threshold, so
  // each frame deterministically decodes to zero detections.
  float* values = static_cast<float*>(m->virAddr);
  for (int i = 0; i < n / 4; ++i) values[i] = -100.0f;
  return 0;
}
int test_free(TestMemory* m) {
  std::free(m->virAddr);
  m->virAddr = nullptr;
  return 0;
}
int test_flush(TestMemory*, int) { return 0; }
int test_infer(void** task, hbDNNTensor* outputs, hbDNNTensor* inputs) {
  (void)inputs;
  *task = reinterpret_cast<void*>(1);
  ++infer_calls;
  if (fill_outputs)
    fill_outputs(outputs, static_cast<int>(output_metadata.size()));
  return 0;
}
int test_wait(void*) { return 0; }
int test_release(void*) { return 0; }
int test_submit(void*, hbUCPSchedParam*) { return 0; }
int hbDNNRelease(void*) {
  ++model_releases;
  return 0;
}

namespace {

// Mirrors main.cpp's detect path: construct, predict, report, benchmark.
void run_scenario(const std::string& image_path, const std::string& json_path,
                  int pipeline_streams, int expected_contexts,
                  int expected_infers) {
  contexts_created = 0;
  infer_calls = 0;
  model_releases = 0;
  fill_outputs = nullptr;
  install_model();

  const std::string model_path = "yolo26n_detect_bayese_640x640_nv12.bin";
  const std::string streams = std::to_string(pipeline_streams);
  std::vector<std::string> arguments = {
      "ultralytics_yolo_cpp", model_path, image_path,
      "--benchmark",          "--warmup", "1",
      "--runs",               "2",        "--rounds",
      "1",                    "--pipeline-streams", streams,
      "--no-save",            "--json",   json_path};
  std::vector<char*> argv;
  for (const std::string& argument : arguments)
    argv.push_back(const_cast<char*>(argument.c_str()));

  const yolo::Options options =
      yolo::parse_options(static_cast<int>(argv.size()), argv.data());
  const yolo::SourceImage source(options.image_path);
  expect(source.valid());

  yolo::YoloDetect::Config config;
  config.model_path = options.model_path;
  config.head = options.head;
  config.score_threshold = options.score_threshold;
  config.nms_threshold = options.nms_threshold;
  config.resize_type = options.resize_type;
  yolo::YoloDetect model(config);

  yolo::YoloDetect::Input input;
  input.bgr = source.bgr();
  input.source_rows = source.rows();
  input.source_cols = source.cols();
  const yolo::YoloDetect::Prediction run = model.predict(input);
  yolo::report_detect(options, run, source);

  expect(yolo::run_detect_benchmark(options, source, &model, run) == 0);
  expect(contexts_created == expected_contexts);
  expect(infer_calls == expected_infers);
  std::ifstream json(json_path.c_str());
  expect(json.good());
  std::string json_text((std::istreambuf_iterator<char>(json)),
                        std::istreambuf_iterator<char>());
  expect(json_text.find("\"pipeline_streams\": " + streams) !=
         std::string::npos);
}

// Task configs mirror main.cpp's branches; the benchmark wrappers rebuild
// the worker configs themselves from the same options.
yolo::YoloSegment::Config segment_config(const yolo::Options& options) {
  yolo::YoloSegment::Config config;
  config.model_path = options.model_path;
  config.score_threshold = options.score_threshold;
  config.nms_threshold = options.nms_threshold;
  config.mask_threshold = 0.5f;
  config.resize_type = options.resize_type;
  return config;
}

yolo::YoloPose::Config pose_config(const yolo::Options& options) {
  yolo::YoloPose::Config config;
  config.model_path = options.model_path;
  config.score_threshold = options.score_threshold;
  config.nms_threshold = options.nms_threshold;
  config.kpt_conf_threshold = options.kpt_conf_threshold;
  config.resize_type = options.resize_type;
  return config;
}

yolo::YoloClassify::Config classify_config(const yolo::Options& options) {
  yolo::YoloClassify::Config config;
  config.model_path = options.model_path;
  config.topk = options.topk;
  config.resize_type = options.resize_type;
  return config;
}

yolo::YoloObb::Config obb_config(const yolo::Options& options) {
  yolo::YoloObb::Config config;
  config.model_path = options.model_path;
  config.classes = options.classes > 0 ? options.classes : 15;
  config.score_threshold = options.score_threshold;
  config.nms_threshold = options.nms_threshold;
  config.angle_sign = options.angle_sign;
  config.angle_offset_degrees = options.angle_offset_degrees;
  config.regularize_obb = options.regularize_obb;
  config.resize_type = options.resize_type;
  return config;
}

// One benchmark-wrapper scenario per task: install the fake geometry and a
// deterministic fill, run exactly one validation predict, then the task
// benchmark entry with two streams, bounded warmup 1 / runs 2 and the given
// round count. Verifies two contexts total, the exact inference count
// 1 + 2*rounds*(warmup+runs), and the task's JSON record fields. Reporting
// is not exercised here (the detect scenarios already cover report wiring);
// the caller asserts both contexts were released once the main model scope
// ends.
template <class Model, class MakeConfig, class RunBenchmark>
void run_wrapper_scenario(const std::string& task, const std::string& image_path,
                          const std::string& json_path, int rounds,
                          const std::vector<std::string>& extra_arguments,
                          const std::vector<std::string>& json_needles,
                          const std::vector<std::string>& json_absent,
                          MakeConfig make_config, RunBenchmark run_benchmark) {
  contexts_created = 0;
  infer_calls = 0;
  model_releases = 0;

  std::vector<std::string> arguments = {
      "ultralytics_yolo_cpp", "fixture-model.bin", image_path,
      "--task",               task,                 "--benchmark",
      "--warmup",             "1",                  "--runs",
      "2",                    "--rounds",           std::to_string(rounds),
      "--pipeline-streams",   "2",                  "--no-save",
      "--json",               json_path};
  arguments.insert(arguments.end(), extra_arguments.begin(),
                   extra_arguments.end());
  std::vector<char*> argv;
  for (const std::string& argument : arguments)
    argv.push_back(const_cast<char*>(argument.c_str()));

  const yolo::Options options =
      yolo::parse_options(static_cast<int>(argv.size()), argv.data());
  const yolo::SourceImage source(options.image_path);
  expect(source.valid());

  Model model(make_config(options));
  typename Model::Input input;
  input.bgr = source.bgr();
  input.source_rows = source.rows();
  input.source_cols = source.cols();
  const typename Model::Prediction run = model.predict(input);
  expect(run_benchmark(options, source, &model, run) == 0);

  expect(contexts_created == 2);
  expect(infer_calls == 1 + 2 * rounds * (1 + 2));

  std::ifstream json(json_path.c_str());
  expect(json.good());
  const std::string json_text((std::istreambuf_iterator<char>(json)),
                              std::istreambuf_iterator<char>());
  for (const std::string& needle : json_needles)
    expect(json_text.find(needle) != std::string::npos);
  for (const std::string& needle : json_absent)
    expect(json_text.find(needle) == std::string::npos);
}


// Benchmark option surface, asserted here because this fixture links the
// parser in src/cli.cpp: the flag values parse with upstream semantics
// (warmup 0 allowed, streams capped at 2, SHA256 provenance validated) and
// invalid input is rejected with the CLI contract's error strings.
void check_option_surface() {
  const std::string source_sha(64, 'a');
  const std::string executable_sha(64, 'b');
  std::vector<std::string> arguments = {
      "ultralytics_yolo_cpp", "--benchmark", "--warmup", "0",
      "--runs",               "3",           "--rounds", "1",
      "--pipeline-streams",   "2",           "--opencv-threads", "all",
      "--no-save",            "--runtime-source-sha256", source_sha,
      "--executable-sha256",  executable_sha};
  std::vector<char*> argv;
  for (const std::string& argument : arguments)
    argv.push_back(const_cast<char*>(argument.c_str()));

  const yolo::Options options =
      yolo::parse_options(static_cast<int>(argv.size()), argv.data());
  expect(options.benchmark);
  expect(options.pipeline_streams == 2);
  expect(options.warmup == 0);
  expect(options.runs == 3);
  expect(options.rounds == 1);
  expect(options.opencv_threads == 0);
  expect(!options.save_result);
  expect(options.runtime_source_sha256 == source_sha);
  expect(options.executable_sha256 == executable_sha);

  const char* invalid[] = {"ultralytics_yolo_cpp", "--runtime-source-sha256",
                           "nope"};
  bool rejected = false;
  try {
    yolo::parse_options(3, const_cast<char**>(invalid));
  } catch (const std::exception& error) {
    rejected = std::string(error.what()) ==
               "--runtime-source-sha256 must be 64 hex digits";
  }
  expect(rejected);
}

}  // namespace

int main() {
  // Per-stack, per-process scratch directory: the x5 and ucp binaries share
  // this source, and ctest -j runs them concurrently. Removing a possible
  // leftover of a recycled pid keeps the directory fresh; everything is
  // cleaned up on exit.
#ifdef YOLO_DNN_STACK_X5
  const std::string stack = "x5";
#else
  const std::string stack = "ucp";
#endif
  namespace fs = std::filesystem;
  const std::string directory =
      "/tmp/yolo_test_benchmark_streams_" + stack + "_" +
      std::to_string(static_cast<long>(::getpid()));
  std::error_code ec;
  fs::remove_all(fs::path(directory), ec);
  expect(fs::create_directories(fs::path(directory), ec));
  const std::string image_path = directory + "/bus.png";
  {
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(60, 120, 200));
    expect(cv::imwrite(image_path, image));
  }

  int status = 0;
  try {
    check_option_surface();
    // Two streams: main model (stream 0) + one worker context; one
    // validation predict + 2 streams x (1 warmup + 2 timed) inferences.
    run_scenario(image_path, directory + "/streams2.json", 2, 2, 7);
    // Every created context was released by the time the scenario returned.
    expect(model_releases == 2);
    // One stream: exactly the main model and its single validation predict.
    run_scenario(image_path, directory + "/streams1.json", 1, 1, 4);
    expect(model_releases == 1);

    // Segment wrapper with rounds > 1: 1 validation predict + 2 streams x
    // 2 rounds x (1 warmup + 2 timed) = 13 inferences, one mask detection
    // per frame.
    install_stride_outputs(80, 32, true);
    install_input(640, 640);
    fill_segment_outputs();
    run_wrapper_scenario<yolo::YoloSegment>(
        "segment", image_path, directory + "/segment.json", 2, {}, {
            "\"pipeline_streams\": 2",
            "\"output_kind\": \"instance_masks\"",
            "\"outputs_per_frame\": 1",
            "\"rounds\": 2",
        }, {}, segment_config, yolo::run_segment_benchmark);
    expect(model_releases == 2);

    // Pose wrapper: one skeleton detection per frame.
    install_stride_outputs(1, 51, false);
    install_input(640, 640);
    fill_pose_outputs();
    run_wrapper_scenario<yolo::YoloPose>(
        "pose", image_path, directory + "/pose.json", 1, {}, {
            "\"pipeline_streams\": 2",
            "\"output_kind\": \"pose_instances\"",
            "\"outputs_per_frame\": 1",
        }, {}, pose_config, yolo::run_pose_benchmark);
    expect(model_releases == 2);

    // Classify wrapper: Top-5 per frame; Top-K records no score/NMS.
    install_classify_model();
    fill_classify_outputs();
    run_wrapper_scenario<yolo::YoloClassify>(
        "classify", image_path, directory + "/classify.json", 1, {}, {
            "\"pipeline_streams\": 2",
            "\"output_kind\": \"topk_predictions\"",
            "\"outputs_per_frame\": 5",
        }, {
            "\"score_threshold\"",
            "\"nms_threshold\"",
        }, classify_config, yolo::run_classify_benchmark);
    expect(model_releases == 2);

    // OBB wrapper: one rotated box per frame; angle options and provenance
    // SHAs are recorded in the JSON.
    install_obb_model();
    fill_obb_outputs();
    const std::string source_sha(64, 'a');
    const std::string executable_sha(64, 'b');
    run_wrapper_scenario<yolo::YoloObb>(
        "obb", image_path, directory + "/obb.json", 1,
        {"--runtime-source-sha256", source_sha, "--executable-sha256",
         executable_sha},
        {
            "\"pipeline_streams\": 2",
            "\"output_kind\": \"rotated_boxes\"",
            "\"outputs_per_frame\": 1",
            "\"angle_sign\": 1.000000",
            "\"angle_offset_degrees\": 0.000000",
            "\"regularize_obb\": true",
            "\"runtime_source_sha256\": \"" + source_sha + "\"",
            "\"executable_sha256\": \"" + executable_sha + "\"",
        },
        {
            "\"detections_per_frame\"",
        },
        obb_config, yolo::run_obb_benchmark);
    expect(model_releases == 2);
  } catch (const std::exception& error) {
    std::printf("test_benchmark_streams: FAIL %s\n", error.what());
    status = 1;
  }
  fs::remove_all(fs::path(directory), ec);
  if (status == 0) std::printf("test_benchmark_streams: OK\n");
  return status;
}
