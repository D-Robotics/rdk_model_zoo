// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cli.cpp
 * @brief MobileNetV2 CLI implementation: parsing, defaults, loading, printing.
 */

#include "cli.hpp"

#include <iostream>
#include <stdexcept>

#include "file_io.hpp"
#include "visualize.hpp"

namespace mobilenetv2 {

std::string default_model_path()
{
#if defined(SOC_S600)
    return "../../model/s600/mobilenetv2_100_nashp_224x224_nv12.hbm";
#elif defined(SOC_S100P)
    return "../../model/s100p/mobilenetv2_100_nashm_224x224_nv12.hbm";
#else
    return "../../model/s100/mobilenetv2_100_nashe_224x224_nv12.hbm";
#endif
}

CliOptions parse_options(int argc, char** argv)
{
    CliOptions options;
    options.model_path = default_model_path();

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

        if (flag == "--help" || flag == "-h") {
            options.help = true;
        } else if (flag == "--model-path") {
            options.model_path = require_value();
        } else if (flag == "--test-img") {
            options.test_img = require_value();
        } else if (flag == "--label-file") {
            options.label_file = require_value();
        } else if (flag == "--top-k") {
            const std::string value = require_value();
            size_t parsed = 0;
            options.top_k = std::stoi(value, &parsed);
            if (parsed != value.size() || options.top_k < 1)
                throw std::invalid_argument("--top-k must be a positive integer");
        } else if (flag == "--resize-shorter") {
            const std::string value = require_value();
            size_t parsed = 0;
            options.resize_shorter = std::stoi(value, &parsed);
            if (parsed != value.size() || options.resize_shorter < 1)
                throw std::invalid_argument("--resize-shorter must be a positive integer");
        } else {
            throw std::invalid_argument("Unknown option: " + flag);
        }
    }
    return options;
}

void print_help(const char* program)
{
    std::cout << "Usage: " << program
              << " [--model-path HBM] [--test-img IMAGE] [--label-file FILE]"
                 " [--top-k N] [--resize-shorter N]\n"
              << "  --model-path   HBM model path (default follows the board: "
                 "../../model/<board>/mobilenetv2_100_<march>_224x224_nv12.hbm)\n"
              << "  --test-img     BGR test image (default: "
                 "../../../test_data/zebra_cls.jpg)\n"
              << "  --label-file   one class name per line (default: "
                 "../../../test_data/imagenet1000_labels.txt)\n"
              << "  --top-k        number of printed classes (default: 5)\n"
              << "  --resize-shorter  shorter edge before the 224 center crop, "
                 "int(224 / crop_pct) (default: 256)\n";
}

cv::Mat load_image(const std::string& path)
{
    cv::Mat image = load_bgr_image(path);
    if (image.empty())
        throw std::runtime_error("Failed to load image: " + path);
    return image;
}

std::vector<std::string> load_labels(const std::string& path)
{
    return load_linewise_labels(path);
}

void print_results(const std::vector<Classification>& results,
                   const std::vector<std::string>& labels)
{
    print_topk_results(results, labels);
}

}  // namespace mobilenetv2
