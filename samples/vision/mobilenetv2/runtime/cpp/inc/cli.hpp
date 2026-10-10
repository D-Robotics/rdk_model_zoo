// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cli.hpp
 * @brief MobileNetV2 CLI surface: options and small helpers.
 *
 * Parsing, defaults, image/label loading and result printing live here so the
 * model implementation stays free of presentation concerns.
 */

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "model_types.hpp"

namespace mobilenetv2 {

/** Parsed command-line options (kebab-case names match the Python runtime). */
struct CliOptions
{
    std::string model_path;             ///< HBM model path.
    std::string test_img;               ///< BGR test image path.
    std::string label_file;             ///< One class name per line.
    int top_k{5};                       ///< Number of printed classes.
    bool help{false};                   ///< Print usage and exit.
};

/// Default HBM path for the configured board target.
std::string default_model_path();

/**
 * @brief Parse argv into options.
 *
 * Accepts both "--flag value" and "--flag=value" spellings.
 * @throws std::invalid_argument on unknown flags, missing values or an
 *         invalid --top-k.
 */
CliOptions parse_options(int argc, char** argv);

/// Print the usage text.
void print_help(const char* program);

/**
 * @brief Load a BGR image.
 * @throws std::runtime_error when the file cannot be read.
 */
cv::Mat load_image(const std::string& path);

/**
 * @brief Load the linewise label file; an unreadable file yields no labels.
 */
std::vector<std::string> load_labels(const std::string& path);

/// Print the Top-K results, one per line, with labels when available.
void print_results(const std::vector<Classification>& results,
                   const std::vector<std::string>& labels);

}  // namespace mobilenetv2
