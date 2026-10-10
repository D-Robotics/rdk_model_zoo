// Copyright (c) 2025 D-Robotics Corporation
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file main.cpp
 * @brief PaddleOCR detection + recognition sample entry.
 *
 * Parses the command line, constructs the PaddleOCR pipeline (construction
 * loads both HBM runtimes), runs predict on one image, prints the recognized
 * texts and saves the rendered result. Options, image/dictionary loading and
 * presentation live in the CLI helpers.
 */

#include <iostream>

#include "cli.hpp"
#include "ocr.hpp"

int main(int argc, char** argv)
{
    try {
        const auto options = ocr::parse_options(argc, argv);
        if (options.help) {
            ocr::print_help(argv[0]);
            return 0;
        }

        // Construction loads both models, queries tensors and allocates
        // buffers (detector first, then recognizer).
        PaddleOCR model(options.det_model_path, options.rec_model_path);

        // One synchronous two-stage pass over the test image: detect, crop,
        // recognize each crop.
        const auto image = ocr::load_image(options.test_img);
        const auto dictionary = ocr::load_token_dictionary(options.vocabulary_path);
        const OcrOptions pipeline_options{options.threshold, options.ratio_prime};
        const auto result =
            model.predict(image, dictionary, pipeline_options);

        ocr::print_results(result);
        ocr::render_result(image, result.det.boxes, result.texts,
                           options.font_path, options.img_save_path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
