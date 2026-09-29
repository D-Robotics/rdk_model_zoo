// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <opencv2/core.hpp>
namespace unetmobilenet {
// alpha_f weights the original image, matching the S source sample.
cv::Mat render_overlay(const cv::Mat& image,const cv::Mat& labels,double alpha_f=0.75);
}
