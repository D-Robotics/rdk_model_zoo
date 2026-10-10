// Copyright (c) 2025 D-Robotics Corporation
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cli.cpp
 * @brief PaddleOCR CLI implementation: parsing, defaults, loading, rendering.
 */

#include "cli.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ocr.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "file_io.hpp"
#include "visualize.hpp"

namespace ocr {

std::string default_det_model_path()
{
#if defined(SOC_S600)
    return "/opt/hobot/model/s600/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm";
#else
    return "/opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm";
#endif
}

std::string default_rec_model_path()
{
#if defined(SOC_S600)
    return "/opt/hobot/model/s600/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm";
#else
    return "/opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm";
#endif
}

CliOptions parse_options(int argc, char** argv)
{
    CliOptions options;
    options.det_model_path = default_det_model_path();
    options.rec_model_path = default_rec_model_path();

    // Both "--flag value" and "--flag=value" spellings are accepted; the
    // launcher forwards the equals form.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string flag = arg;
        std::string inline_value;
        bool has_inline_value = false;
        const auto eq = arg.find('=');
        if (eq != std::string::npos) {
            flag = arg.substr(0, eq);
            inline_value = arg.substr(eq + 1);
            has_inline_value = true;
        }
        auto require_value = [&]() -> std::string {
            if (has_inline_value)
                return inline_value;
            if (i + 1 >= argc)
                throw std::invalid_argument(flag + " needs a value");
            return argv[++i];
        };
        auto parse_float = [&](const std::string& flag_name) -> float {
            const std::string value = require_value();
            size_t parsed = 0;
            const float parsed_value = std::stof(value, &parsed);
            if (parsed != value.size())
                throw std::invalid_argument(flag_name + " must be a number");
            return parsed_value;
        };

        if (flag == "--help" || flag == "-h") {
            options.help = true;
        } else if (flag == "--det-model-path") {
            options.det_model_path = require_value();
        } else if (flag == "--rec-model-path") {
            options.rec_model_path = require_value();
        } else if (flag == "--test-img") {
            options.test_img = require_value();
        } else if (flag == "--vocabulary-path") {
            options.vocabulary_path = require_value();
        } else if (flag == "--threshold") {
            options.threshold = parse_float("--threshold");
        } else if (flag == "--ratio-prime") {
            options.ratio_prime = parse_float("--ratio-prime");
        } else if (flag == "--img-save-path") {
            options.img_save_path = require_value();
        } else if (flag == "--font-path") {
            options.font_path = require_value();
        } else {
            throw std::invalid_argument("Unknown option: " + flag);
        }
    }
    return options;
}

void print_help(const char* program)
{
    std::cout << "Usage: " << program
              << " [--det-model-path HBM] [--rec-model-path HBM]"
                 " [--test-img IMAGE] [--vocabulary-path FILE]"
                 " [--threshold F] [--ratio-prime F]"
                 " [--img-save-path FILE] [--font-path FILE]\n"
              << "  --det-model-path   detector HBM path (default follows the board:\n"
                 "                     /opt/hobot/model/<board>/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm)\n"
              << "  --rec-model-path   recognizer HBM path (default follows the board:\n"
                 "                     /opt/hobot/model/<board>/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm)\n"
              << "  --test-img         BGR test image (default: ../../../test_data/gt_2322.jpg)\n"
              << "  --vocabulary-path  character dictionary, one token per line\n"
              << "  --threshold        detection-map binarization threshold (default: 0.5)\n"
              << "  --ratio-prime      contour dilation ratio (default: 2.7)\n"
              << "  --img-save-path    where to save the result image (default: result.jpg)\n"
              << "  --font-path        TrueType font used to render recognized text\n";
}

cv::Mat load_image(const std::string& path)
{
    cv::Mat image = load_bgr_image(path);
    if (image.empty())
        throw std::runtime_error("Failed to load image: " + path);
    return image;
}

std::vector<std::string> load_token_dictionary(const std::string& path)
{
    // Read the vocabulary verbatim (one line == one token). See the note in
    // cli.hpp: the generic linewise label loader mangles '{'/'}'/',' tokens.
    std::vector<std::string> lines;
    std::ifstream dict_ifs(path);
    if (!dict_ifs.is_open())
        throw std::runtime_error("Failed to open dictionary: " + path);
    std::string line;
    while (std::getline(dict_ifs, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        lines.push_back(line);
    }
    return lines;
}

void print_results(const OcrResult& result)
{
    // Successful crops keep their original crop index; skipped crops are
    // reported with their index and cause on stderr.
    for (std::size_t i = 0; i < result.texts.size(); ++i)
        std::cout << "[" << result.text_crop_indices[i] << "] Prediction: "
                  << result.texts[i] << std::endl;
    for (const auto& error : result.crop_errors)
        std::cerr << "[" << error.crop_index << "] recognition failed: "
                  << error.message << std::endl;
}

void render_result(const cv::Mat& image,
                   const std::vector<std::vector<cv::Point>>& boxes,
                   const std::vector<std::string>& texts,
                   const std::string& font_path,
                   const std::string& save_path)
{
    // Draw polygon boxes on a copy of the original image.
    const auto img_boxes = draw_polygon_boxes(image, boxes);

    // White canvas with the recognized text rendered near the boxes. When
    // crops were skipped, texts pair with boxes compactly (text j at
    // boxes[j]) — not by original crop index.
    cv::Mat white_canvas(img_boxes.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    const auto img_with_text =
        draw_text(white_canvas, texts, boxes, font_path, 35,
                  cv::Scalar(0, 0, 255),  // red (BGR)
                  2);

    // Side-by-side: left = detected boxes, right = recognized text.
    cv::Mat combined;
    cv::hconcat(img_boxes, img_with_text, combined);

    if (!cv::imwrite(save_path, combined))
        throw std::runtime_error("Failed to save result image: " + save_path);
    std::cout << "[Saved] Result saved to: " << save_path << std::endl;
}

}  // namespace ocr
