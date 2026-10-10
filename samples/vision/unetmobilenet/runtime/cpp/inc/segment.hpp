// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace unetmobilenet {

// ===========================================================================
// Score tensor contract (public; no SDK types)
// ===========================================================================
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

// ===========================================================================
// Board identity (exact S aliases; no unknown-target fallback)
// ===========================================================================
// Normalize whitespace/case; map (soc, board) to "s100"/"s100p"/"s600" or "".
std::string match_native_target(std::string soc, std::string board);
// Throw unless the requested target matches the build and the local board.
void require_native_target(const std::string& requested, const std::string& built);

// ===========================================================================
// Owned stage data
// ===========================================================================
struct ImageContext { int original_height; int original_width; };
struct PreparedInput { cv::Mat y; cv::Mat uv; ImageContext context; };

// ===========================================================================
// Named model: construction loads the runtime and validates the published
// split-NV12/score contract; the stage methods own their data per call.
// ===========================================================================
class UnetMobileNet {
public:
    // Host test seam for the on-board identity check; never a CLI option.
    using ExecutionGate = std::function<void(const std::string&, const std::string&)>;
    explicit UnetMobileNet(const std::string& model_path, const std::string& target,
                           int priority = 0, int bpu_core = -1,
                           ExecutionGate gate = {});
    ~UnetMobileNet();
    UnetMobileNet(const UnetMobileNet&) = delete;
    UnetMobileNet& operator=(const UnetMobileNet&) = delete;

    // Nonempty CV_8UC3 BGR -> INTER_AREA stretch 2048x1024 -> owned NV12
    // Y CV_8UC1 [1024,2048], UV CV_8UC2 [512,1024], per-call context.
    PreparedInput preprocess(const cv::Mat& image) const;
    // Upload the planes, run one BPU task, own the raw padded scores.
    RawScores infer(const PreparedInput& prepared);
    // Affine/raw argmax, direct nearest resize -> original-size CV_32S IDs.
    cv::Mat postprocess(const RawScores& raw, const ImageContext& context) const;
    // Exact composition of the three stages; returns class IDs, not overlay.
    cv::Mat predict(const cv::Mat& image);
    // The bound output contract (shape, dtype, affine metadata).
    const ScoreSpec& score_spec() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace unetmobilenet
