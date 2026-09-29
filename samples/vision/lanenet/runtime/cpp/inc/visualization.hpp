// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tensor_contract.hpp"
#include <opencv2/core.hpp>
namespace lanenet {
cv::Mat embedding_image(const LaneResult &result);
cv::Mat binary_image(const LaneResult &result);
} // namespace lanenet
