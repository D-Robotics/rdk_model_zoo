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

// Entry point of the ultralytics_yolo C++ runtime. The task is
// selected with --task; the model is constructed here, fed caller-owned
// pixels through Input, run with predict, and the returned Prediction is
// handed to the CLI for reporting.

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>

#include "cli.hpp"
#include "yolo.hpp"

int main(int argc, char** argv) {
  try {
    const yolo::Options options = yolo::parse_options(argc, argv);

    const int online_cpu_threads = std::max(1, cv::getNumberOfCPUs());
    const int opencv_threads =
        options.opencv_threads == 0 ? online_cpu_threads : options.opencv_threads;
    cv::setUseOptimized(true);
    cv::setNumThreads(opencv_threads);

    std::cout << "[INFO] YOLO " << options.task
              << " C++ sample (direct-LTRB + DFL heads)" << std::endl;
    std::cout << "[INFO] OpenCV: " << CV_VERSION
              << ", online CPUs: " << online_cpu_threads
              << ", CPU thread policy: "
              << (options.opencv_threads == 0 ? "all-online" : "fixed")
              << ", OpenCV threads: " << cv::getNumThreads() << std::endl;
    std::cout << "[INFO] Loading model: " << options.model_path << std::endl;
    std::cout << "[INFO] Pipeline streams: " << options.pipeline_streams
              << std::endl;

    // The CLI owns image loading; the model receives pixels, never a path.
    const yolo::SourceImage source(options.image_path);
    if (!source.valid()) {
      std::cerr << "[ERROR] Cannot read image: " << options.image_path
                << std::endl;
      return 1;
    }
    std::cout << "[INFO] Image: " << source.cols() << "x" << source.rows()
              << std::endl;

    if (options.task == "detect") {
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

      if (options.benchmark)
        return yolo::run_detect_benchmark(options, source, &model, run);
      return 0;
    }

    if (options.task == "segment") {
      yolo::YoloSegment::Config config;
      config.model_path = options.model_path;
      config.score_threshold = options.score_threshold;
      config.nms_threshold = options.nms_threshold;
      config.mask_threshold = 0.5f;
      config.resize_type = options.resize_type;
      yolo::YoloSegment model(config);

      yolo::YoloSegment::Input input;
      input.bgr = source.bgr();
      input.source_rows = source.rows();
      input.source_cols = source.cols();
      const yolo::YoloSegment::Prediction run = model.predict(input);
      yolo::report_segment(options, run);

      if (options.benchmark)
        return yolo::run_segment_benchmark(options, source, &model, run);
      return 0;
    }

    if (options.task == "pose") {
      yolo::YoloPose::Config config;
      config.model_path = options.model_path;
      config.score_threshold = options.score_threshold;
      config.nms_threshold = options.nms_threshold;
      config.kpt_conf_threshold = options.kpt_conf_threshold;
      config.resize_type = options.resize_type;
      yolo::YoloPose model(config);

      yolo::YoloPose::Input input;
      input.bgr = source.bgr();
      input.source_rows = source.rows();
      input.source_cols = source.cols();
      const yolo::YoloPose::Prediction run = model.predict(input);
      yolo::report_pose(options, run, source);

      if (options.benchmark)
        return yolo::run_pose_benchmark(options, source, &model, run);
      return 0;
    }

    if (options.task == "obb") {
      yolo::YoloObb::Config config;
      config.model_path = options.model_path;
      config.classes = options.classes > 0 ? options.classes : 15;
      config.score_threshold = options.score_threshold;
      config.nms_threshold = options.nms_threshold;
      config.angle_sign = options.angle_sign;
      config.angle_offset_degrees = options.angle_offset_degrees;
      config.regularize_obb = options.regularize_obb;
      config.resize_type = options.resize_type;
      yolo::YoloObb model(config);

      yolo::YoloObb::Input input;
      input.bgr = source.bgr();
      input.source_rows = source.rows();
      input.source_cols = source.cols();
      const yolo::YoloObb::Prediction run = model.predict(input);
      yolo::report_obb(options, run, source);

      if (options.benchmark)
        return yolo::run_obb_benchmark(options, source, &model, run);
      return 0;
    }

    yolo::YoloClassify::Config config;
    config.model_path = options.model_path;
    config.topk = options.topk;
    config.resize_type = options.resize_type;
    yolo::YoloClassify model(config);

    yolo::YoloClassify::Input input;
    input.bgr = source.bgr();
    input.source_rows = source.rows();
    input.source_cols = source.cols();
    const yolo::YoloClassify::Prediction run = model.predict(input);
    yolo::report_classify(options, run);

    if (options.benchmark)
      return yolo::run_classify_benchmark(options, source, &model, run);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    yolo::print_usage(argv[0]);
    return 1;
  }
}
