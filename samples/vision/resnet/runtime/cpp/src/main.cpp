// Copyright (c) 2025, XiangshunZhao D-Robotics.
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file main.cpp
 * @brief ResNet18 classification sample entry.
 *
 * Parses the command line, constructs the Resnet18 model (construction loads
 * the HBM runtime), classifies one image through predict, and prints the
 * Top-K classes. Options, image/label loading and presentation live in the
 * CLI helpers.
 */

#include <iostream>

#include "classify.hpp"
#include "cli.hpp"

int main(int argc, char** argv)
{
    try {
        const auto options = resnet::parse_options(argc, argv);
        if (options.help) {
            resnet::print_help(argv[0]);
            return 0;
        }

        // Construction loads the model, queries tensors and allocates buffers.
        Resnet18 model(options.model_path);

        // One synchronous classification of the test image.
        const auto image = resnet::load_image(options.test_img);
        const auto top_k = model.predict(image, options.top_k);

        resnet::print_results(top_k, resnet::load_labels(options.label_file));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
