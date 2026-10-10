// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Board-side driver shared by the classify, segment, pose and OBB programs:
// image preprocessing, `MODEL IMAGE [OUTPUT] [options]` parsing, one
// validation frame and the optional bounded 1/2-stream E2E benchmark.
#ifndef YOLO_COMMON_TASK_BENCHMARK_H_
#define YOLO_COMMON_TASK_BENCHMARK_H_
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/benchmark.h"
#include "common/decode.h"
namespace yolo {

// Stretch (resize_type 0) or letterbox with gray 127 padding (resize_type 1).
// `transform` records the scale and padding used, for the inverse mapping.
inline cv::Mat preprocess_image(const cv::Mat& image, int input_h, int input_w,
                                int resize_type, ImageTransform* transform) {
  cv::Mat result;
  *transform = ImageTransform();
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

struct TaskCommand {
  std::vector<std::string> paths;  // model, image and optional output path
  BenchmarkOptions options;
  bool help = false;
};

// Leading arguments that do not start with "--" are positional paths and
// replace `defaults` in order; the remaining arguments are benchmark options.
// `options` carries the task's own defaults (for example its NMS threshold).
inline TaskCommand parse_task_command(int argc, char** argv,
                                      const std::vector<std::string>& defaults,
                                      const BenchmarkOptions& options) {
  TaskCommand command;
  command.paths = defaults;
  command.options = options;
  int index = 1;
  for (size_t slot = 0; index < argc && slot < defaults.size(); ++index, ++slot) {
    const std::string arg(argv[index]);
    if (arg == "-h" || arg == "--help") {
      command.help = true;
      return command;
    }
    if (arg.compare(0, 2, "--") == 0) break;
    command.paths[slot] = arg;
  }
  for (int i = index; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help") {
      command.help = true;
      return command;
    }
  }
  std::string error;
  if (!parse_benchmark_options(argc, argv, index, &command.options, &error))
    throw std::invalid_argument(error);
  return command;
}

inline void print_task_usage(const char* program, const std::string& positional,
                             bool object_task) {
  std::cout << "Usage: " << program << " " << positional << " [options]\n"
            << "  --benchmark              Run the bounded end-to-end benchmark\n"
            << "  --warmup N               Warmup frames per stream per round (20)\n"
            << "  --runs N                 Timed frames per stream per round (200)\n"
            << "  --rounds N               Benchmark rounds (3)\n"
            << "  --pipeline-streams 1|2   Independent concurrent pipelines (1)\n"
            << "  --opencv-threads all|N   OpenCV CPU threads (all online CPUs)\n"
            << "  --resize-type 0|1        0 stretch, 1 letterbox (task default)\n"
            << "  --json PATH              Write the aggregate benchmark JSON\n"
            << "  --runtime-source-sha256 HEX, --executable-sha256 HEX\n"
            << "                           Provenance recorded in the JSON\n"
            << "  --no-save                Skip the result image\n";
  if (object_task)
    std::cout << "  --score P, --nms P       Score and NMS IoU thresholds\n";
}

// Runtime contract: `explicit Runtime(...)` loads one model context;
// `int input_h() const`, `int input_w() const`,
// `const char* implementation() const` (the JSON implementation label); and
// `size_t run(const cv::Mat& image, int resize_type, StageTiming* timing)`
// executes one complete frame, keeps the result for reporting and returns the
// number of task outputs. `run` throws on any failure.
template <class Runtime>
int run_task(const TaskCommand& command, int default_resize_type,
             BenchmarkMeta meta,
             const std::function<std::unique_ptr<Runtime>()>& make_runtime,
             const std::function<void(Runtime&, const cv::Mat&, int)>& report) {
  const BenchmarkOptions& options = command.options;
  const int online_cpu_threads = std::max(1, cv::getNumberOfCPUs());
  cv::setUseOptimized(true);
  cv::setNumThreads(options.opencv_threads == 0 ? online_cpu_threads
                                                : options.opencv_threads);
  const int resize_type =
      options.resize_type < 0 ? default_resize_type : options.resize_type;
  std::cout << "[INFO] OpenCV: " << CV_VERSION << ", online CPUs: "
            << online_cpu_threads << ", OpenCV threads: " << cv::getNumThreads()
            << ", pipeline streams: " << options.pipeline_streams
            << ", resize type: " << resize_type << std::endl;

  std::vector<std::unique_ptr<Runtime> > runtimes;
  for (int stream = 0; stream < options.pipeline_streams; ++stream) {
    runtimes.push_back(make_runtime());
    if (runtimes.back()->input_h() != runtimes.front()->input_h() ||
        runtimes.back()->input_w() != runtimes.front()->input_w())
      throw std::runtime_error("Runtime contexts expose different inputs.");
  }
  const cv::Mat image = cv::imread(command.paths[1]);
  if (image.empty())
    throw std::runtime_error("Cannot read image: " + command.paths[1]);

  StageTiming timing;
  const size_t outputs = runtimes.front()->run(image, resize_type, &timing);
  std::cout << std::fixed << std::setprecision(3)
            << "[INFO] Validation: outputs=" << outputs
            << ", preprocess=" << timing.preprocess_ms
            << " ms, runtime=" << timing.runtime_ms
            << " ms, postprocess=" << timing.postprocess_ms
            << " ms, end_to_end=" << timing.end_to_end_ms << " ms" << std::endl;
  report(*runtimes.front(), image, resize_type);
  if (!options.enabled) return 0;

  std::vector<BenchmarkPipeline> pipelines;
  for (size_t stream = 0; stream < runtimes.size(); ++stream) {
    Runtime* runtime = runtimes[stream].get();
    pipelines.push_back([runtime, &image, resize_type](
                            StageTiming* frame, size_t* count, std::string* error) {
      try {
        *count = runtime->run(image, resize_type, frame);
        return true;
      } catch (const std::exception& exception) {
        *error = exception.what();
        return false;
      }
    });
  }
  meta.implementation = runtimes.front()->implementation();
  meta.model_path = command.paths[0];
  meta.image_path = command.paths[1];
  meta.resize_type = resize_type;
  meta.cpu_thread_policy = options.opencv_threads == 0 ? "all_online" : "fixed";
  meta.online_cpu_threads = online_cpu_threads;
  meta.opencv_threads = cv::getNumThreads();
  return run_benchmark(pipelines, options, outputs, meta) ? 0 : 1;
}

// Rejects nonfinite output values a decoder consumes from a zero-copy view.
inline void require_finite(const float* values, int count) {
  for (int i = 0; i < count; ++i)
    if (!std::isfinite(values[i]))
      throw std::runtime_error("Output contains nonfinite values.");
}

inline double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                         const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace yolo
#endif
