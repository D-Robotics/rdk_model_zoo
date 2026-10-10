// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// YOLOv5 native entry point: parse the command line, construct the detect
// model, load the source image through the CLI, run predict, then report the
// returned prediction (result plus evidence) through the CLI for the dump and
// the rendered output. Exit codes: 0 success (or --help), 2 rejected or
// failed run (a failure record is written when --dump-dir is set).

#include "cli.hpp"
#include "detect.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  yolov5::RuntimeOptions options;
  options.output_path = "result.jpg";
  int status = 0;
  std::string failure;
  try {
    if (yolov5::parse_options(argc, argv, &options)) return 0;

    yolov5::Yolov5::Config config;
    config.target = options.target;
    config.model_path = options.model_path;
    config.score_threshold = options.score_threshold;
    config.nms_threshold = options.nms_threshold;
    config.priority = options.priority;
    config.bpu_core = options.bpu_core;

    yolov5::Yolov5 model(config);
    // The CLI owns the file IO: it loads the image and hands the model
    // caller-owned pixels; the model never touches a path.
    yolov5::SourceImage source(options);
    yolov5::Yolov5::Input input;
    input.bgr = source.bgr();
    input.source_cols = source.cols();
    input.source_rows = source.rows();
    const yolov5::Yolov5::Prediction run = model.predict(input);
    status = yolov5::report(options, run, source, model.input_size());
  } catch (const std::exception& error) {
    failure = error.what();
    std::cerr << "yolov5_cpp: " << failure << "\n";
    status = 2;
  }
  if (status != 0 && !options.dump_dir.empty()) {
    yolov5::write_failure_record(options, failure, status);
  }
  return status;
}
