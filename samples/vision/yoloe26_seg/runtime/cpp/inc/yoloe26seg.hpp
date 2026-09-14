// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace yoloe26 {
struct Detection {
    std::array<float, 4> box;
    float score;
    int label;
    std::array<float, 32> coefficients;
};

struct Result {
    std::vector<Detection> detections;
    std::vector<cv::Mat> masks;
};

// One model instance per inference thread. Owns UCP tensors and model resources.
class YoloE26Seg {
public:
    explicit YoloE26Seg(const std::string& model_path);
    ~YoloE26Seg();
    YoloE26Seg(const YoloE26Seg&) = delete;
    YoloE26Seg& operator=(const YoloE26Seg&) = delete;
    Result predict(const cv::Mat& image, float threshold = .25f,
                   int max_det = 300, bool single_label = true);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace yoloe26
