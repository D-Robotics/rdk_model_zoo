// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cli.hpp
 * @brief ResNet18 command-line options, input loading and result printing.
 *
 * The CLI owns option parsing (kebab-case flags matching the Python runtime),
 * path defaults that depend on the configured board, and result presentation.
 * No model or SDK work happens here.
 */

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "model_types.hpp"

namespace resnet {

/** Parsed command line for one program run. */
struct CliOptions
{
    std::string model_path;  ///< HBM model path (default: board model location).
    std::string test_img = "../../../test_data/zebra_cls.jpg";  ///< BGR input image.
    std::string label_file = "../../../../../../datasets/imagenet/imagenet_classes.names";  ///< One label per line.
    int top_k = 5;  ///< Number of printed classes.
    bool help = false;  ///< Print usage and exit.
};

/**
 * @brief Default HBM path for the configured board.
 *
 * S600 builds use the S600 model location; every other build uses the S100
 * location, matching the historical SoC-dependent gflags default.
 *
 * @return Absolute default model path.
 */
std::string default_model_path();

/**
 * @brief Parse argv into options.
 *
 * Accepted flags: --model-path, --test-img, --label-file, --top-k and
 * --help. Unknown flags or missing values throw std::invalid_argument.
 *
 * @param argc Argument count including the program name.
 * @param argv Argument values.
 * @return Parsed options with the documented defaults filled in.
 */
CliOptions parse_options(int argc, char** argv);

/** Print the usage text (one line per option). */
void print_help(const char* program);

/**
 * @brief Load the BGR test image.
 *
 * @param path Image path.
 * @return Loaded image; throws std::runtime_error when it cannot be read.
 */
cv::Mat load_image(const std::string& path);

/**
 * @brief Load the linewise label file for presentation.
 *
 * Matching the historical behavior, a missing or unreadable file yields an
 * empty vector and results print with raw class ids.
 *
 * @param path Label file path.
 * @return Labels indexed by class id; empty on failure.
 */
std::vector<std::string> load_labels(const std::string& path);

/**
 * @brief Print Top-K classification results.
 *
 * @param results Top-K results from the model.
 * @param labels Labels indexed by class id; may be empty.
 */
void print_results(const std::vector<Classification>& results,
                   const std::vector<std::string>& labels);

}  // namespace resnet
