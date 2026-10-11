// Copyright (c) 2025 D-Robotics Corporation
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file main.cpp
 * @brief MobileNetV2 classification sample entry.
 *
 * Parses the command line, constructs the MobileNetV2 model (construction
 * loads the HBM runtime), classifies one image through predict, and prints
 * the Top-K classes. Options, image/label loading and presentation live in
 * the CLI helpers.
 */

#include <iostream>

#include "classify.hpp"
#include "cli.hpp"

int main(int argc, char** argv)
{
    try {
        const auto options = mobilenetv2::parse_options(argc, argv);
        if (options.help) {
            mobilenetv2::print_help(argv[0]);
            return 0;
        }

        // Construction loads the model, queries tensors and allocates buffers.
        MobileNetV2 model(options.model_path, options.resize_shorter);

        // One synchronous classification of the test image.
        const auto image = mobilenetv2::load_image(options.test_img);
        const auto top_k = model.predict(image, options.top_k);

        mobilenetv2::print_results(top_k, mobilenetv2::load_labels(options.label_file));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
