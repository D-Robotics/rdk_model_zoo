// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
#pragma once
#include "hobot/dnn/hb_dnn.h"
#include <cstdint>
#include <vector>
namespace gemma4 {
// Exact semantic matrix, optional leading singleton axes, nonoverlapping byte
// strides and in-allocation addresses. Capacity is the originally allocated
// size.
void ValidateVisionTensor(const hbDNNTensorProperties &properties, bool input,
                          int64_t capacity = 0);
void WriteVisionInput(hbDNNTensor &tensor, const std::vector<float> &patches,
                      int64_t capacity);
std::vector<float> ReadVisionOutput(const hbDNNTensor &tensor,
                                    int64_t capacity);
} // namespace gemma4
