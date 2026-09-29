// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tensor_contract.hpp"
#include <functional>
#include <opencv2/core.hpp>

namespace unetmobilenet {
struct ImageContext { int original_height; int original_width; };
struct PreparedInput { cv::Mat y; cv::Mat uv; ImageContext context; };
using RawRunner=std::function<RawScores(const cv::Mat&,const cv::Mat&)>;
class UnetMobileNetTask {
public:
    explicit UnetMobileNetTask(RawRunner runner);
    // Nonempty CV_8UC3 BGR -> INTER_AREA stretch 2048x1024 -> owned NV12
    // Y CV_8UC1 [1024,2048], UV CV_8UC2 [512,1024], per-call context.
    PreparedInput pre_process(const cv::Mat& image) const;
    // Return raw padded scores and metadata without decoding or rendering.
    RawScores forward(const PreparedInput& prepared) const;
    // Affine/raw argmax, direct nearest resize -> original-size CV_32S IDs.
    cv::Mat post_process(const RawScores& raw,const ImageContext& context) const;
    // Exact composition of the three stages; returns class IDs, not overlay.
    cv::Mat predict(const cv::Mat& image) const;
private:
    RawRunner runner_;
};
}  // namespace unetmobilenet
