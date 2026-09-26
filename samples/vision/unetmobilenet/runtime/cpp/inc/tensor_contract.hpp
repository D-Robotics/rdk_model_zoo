// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace unetmobilenet {
enum class ScoreType { Int32, Float32 };
struct ScoreSpec {
    std::array<std::int64_t, 4> shape{};
    std::array<std::int64_t, 4> stride{};  // bytes, not elements
    std::size_t storage_bytes{};
    ScoreType type{ScoreType::Int32};
    bool scaled{false};
    int axis{3};
    std::vector<float> scales;
    std::vector<std::int32_t> zero_points;
};
struct RawScores {
    ScoreSpec spec;
    std::vector<unsigned char> bytes;  // owned, includes SDK padding
};
// Validates positive NHWC [1,H,W,19], non-overlapping byte strides/capacity,
// and explicit per-tensor/per-axis affine metadata before any memory read.
void validate_scores(const ScoreSpec& spec);
// Return original-resolution int32 labels; nearest resize is direct, not via
// input geometry. Argmax ties choose lowest ID. Does not mutate the raw bytes.
std::vector<std::int32_t> decode_scores(const RawScores& raw, int height, int width);
}  // namespace unetmobilenet
