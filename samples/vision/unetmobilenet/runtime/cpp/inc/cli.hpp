// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "segment.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace unetmobilenet {
struct CliOptions {
    std::string model_path, test_img, target;
    double alpha_f = 0.75;
    std::string img_save_path = "result.jpg";
    std::string mask_save_path = "unetmobilenet_mask.png";
    std::string report_path = "unetmobilenet_cpp_report.json";
    int priority = 0;
    int bpu_core = -1;
    bool help = false;
};
// Parse kebab-case flags (source underscore aliases accepted). Throws
// std::invalid_argument on unknown flags, missing values, bad numerics or
// missing required options.
CliOptions parse_options(const std::vector<std::string>& args);
void print_help();
cv::Mat load_image(const std::string& path);
std::string json_quote(const std::string& value);
// alpha_f weights the original image, matching the S source sample.
cv::Mat render_overlay(const cv::Mat& image, const cv::Mat& labels, double alpha_f = 0.75);
// Write the overlay image, the 8-bit label mask and the JSON report, and
// print the report to stdout.
void save_results(const CliOptions& options, const cv::Mat& image,
                  const cv::Mat& labels, const ScoreSpec& scores);
}  // namespace unetmobilenet
