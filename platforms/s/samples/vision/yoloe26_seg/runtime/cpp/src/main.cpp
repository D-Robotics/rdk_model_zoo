// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file main.cpp
 * @brief Command-line image demo for YOLOE-26 prompt-free segmentation.
 */

#include "yoloe26seg.hpp"

#include <gflags/gflags.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef YOLOE26_SOURCE_DIR
#define YOLOE26_SOURCE_DIR "."
#endif

namespace {

const std::string kDefaultImagePath =
    std::string(YOLOE26_SOURCE_DIR) + "/../../test_data/office_desk.jpg";
const std::string kDefaultLabelPath =
    std::string(YOLOE26_SOURCE_DIR) + "/../../test_data/coco_extended.names";

DEFINE_string(model_path, "",
              "Path to the canonical board-matching HBM; empty selects the downloaded n model.");
DEFINE_string(model_size, "n", "Released model size: n, s, m, l, or x.");
DEFINE_string(test_img, kDefaultImagePath, "Path to the BGR test image.");
DEFINE_string(label_file, kDefaultLabelPath,
              "Path to the 4585-class label file, one class per line.");
DEFINE_string(output_path, "result.jpg", "Path for the rendered result image.");
DEFINE_double(score_thres, 0.25,
              "Confidence threshold in the open interval (0, 1).");
DEFINE_int32(max_det, 300, "Maximum number of candidates to retain (1..8400).");
DEFINE_bool(multi_label, false,
            "Keep multiple classes per selected anchor instead of one class.");

std::vector<std::string> load_labels(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot read labels: " + path);
    std::vector<std::string> labels;
    std::string label;
    while (std::getline(stream, label)) {
        if (!label.empty() && label.back() == '\r') label.pop_back();
        labels.push_back(std::move(label));
    }
    if (labels.size() != 4585) {
        throw std::runtime_error("Expected 4585 labels, got " +
                                 std::to_string(labels.size()));
    }
    return labels;
}

void render_result(cv::Mat& image,
                   const std::vector<std::string>& labels,
                   const InstanceSegResult& result) {
    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& detection = result.detections[i];
        if (detection.class_id < 0 ||
            detection.class_id >= static_cast<int>(labels.size())) {
            continue;
        }
        const cv::Scalar color((detection.class_id * 47) % 200 + 40,
                               (detection.class_id * 83) % 200 + 40,
                               (detection.class_id * 131) % 200 + 40);
        const int x1 = std::clamp(static_cast<int>(detection.bbox[0]), 0, image.cols);
        const int y1 = std::clamp(static_cast<int>(detection.bbox[1]), 0, image.rows);
        const int x2 = std::clamp(static_cast<int>(detection.bbox[2]), 0, image.cols);
        const int y2 = std::clamp(static_cast<int>(detection.bbox[3]), 0, image.rows);
        if (i < result.masks.size() && !result.masks[i].empty() && x2 > x1 &&
            y2 > y1) {
            cv::Mat mask;
            if (result.masks[i].size() == cv::Size(x2 - x1, y2 - y1)) {
                mask = result.masks[i];
            } else {
                cv::resize(result.masks[i], mask, cv::Size(x2 - x1, y2 - y1),
                           0, 0, cv::INTER_NEAREST);
            }
            cv::Mat roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            for (int y = 0; y < roi.rows; ++y) {
                for (int x = 0; x < roi.cols; ++x) {
                    if (!mask.at<unsigned char>(y, x)) continue;
                    auto& pixel = roi.at<cv::Vec3b>(y, x);
                    for (int channel = 0; channel < 3; ++channel) {
                        pixel[channel] = static_cast<unsigned char>(
                            pixel[channel] * 0.6f + color[channel] * 0.4f);
                    }
                }
            }
        }
        if (x2 > x1 && y2 > y1) {
            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
            cv::putText(image,
                        labels[detection.class_id] + " " +
                            cv::format("%.2f", detection.score),
                        cv::Point(x1, std::max(y1 - 5, 12)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
        }
    }
}

}  // namespace

/**
 * @brief Run one image through the staged C++ inference pipeline.
 *
 * @param[in] argc Number of command-line arguments.
 * @param[in] argv Command-line argument array.
 * @return 0 on success, non-zero when parsing, initialization, inference, or
 *         output rendering fails.
 */
int main(int argc, char** argv) {
    gflags::SetUsageMessage("yoloe26seg --model_path=... --test_img=... \
--label_file=... --output_path=result.jpg");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (argc != 1) {
        std::cerr << "Unexpected positional argument; use named snake_case flags.\n";
        return 2;
    }

    try {
        const std::string model_path = FLAGS_model_path.empty()
                                           ? yoloe26::YoloE26Seg::default_model_path(
                                                 FLAGS_model_size)
                                           : FLAGS_model_path;
        yoloe26::YoloE26SegConfig config;
        config.model_path = model_path;
        config.model_size = FLAGS_model_size;
        config.score_threshold = static_cast<float>(FLAGS_score_thres);
        config.max_det = FLAGS_max_det;
        config.single_label = !FLAGS_multi_label;

        yoloe26::YoloE26Seg model(config);
        const int init_code = model.init();
        if (init_code != 0) {
            std::cerr << "YoloE26Seg init failed: " << init_code << '\n';
            return init_code;
        }

        const cv::Mat image = cv::imread(FLAGS_test_img, cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Cannot read image: " + FLAGS_test_img);
        const auto labels = load_labels(FLAGS_label_file);
        const InstanceSegResult result = model.predict(image);

        cv::Mat rendered = image.clone();
        // Result masks are bbox-local CV_8UC1 matrices, so render them inside
        // their clipped integer box before drawing the float-coordinate box.
        render_result(rendered, labels, result);
        if (!cv::imwrite(FLAGS_output_path, rendered)) {
            throw std::runtime_error("Cannot write output image: " + FLAGS_output_path);
        }

        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& detection = result.detections[i];
            std::cout << detection.class_id << ' ' << detection.score << ' '
                      << detection.bbox[0] << ' ' << detection.bbox[1] << ' '
                      << detection.bbox[2] << ' ' << detection.bbox[3] << '\n';
        }
        std::cout << "[Saved] Result saved to: " << FLAGS_output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
