// Copyright (c) 2025 D-Robotics Corporation
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file classify.hpp
 * @brief MobileNetV2 classification model interface.
 *
 * MobileNetV2 owns the loaded HBM runtime. Each stage exchanges owned
 * per-call data: preprocess resizes the shorter edge of a BGR image and
 * center-crops it into owned NV12 planes, infer uploads one prepared input
 * into the reusable SDK buffers and returns the owned raw logits copied out of
 * the output tensor, and postprocess applies softmax and selects the Top-K
 * classes. predict is the explicit composition of the three
 * stages. SDK handle and tensor types stay inside the implementation; nothing
 * in this header requires the DNN SDK.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "model_types.hpp"

/** Owned NV12 planes for one call: Y (h×w) and interleaved UV (h/2×w/2×2). */
struct MobileNetV2Prepared
{
    std::vector<uint8_t> y;   ///< Y plane bytes, tightly packed rows.
    std::vector<uint8_t> uv;  ///< Interleaved UV plane bytes (U,V per sample).
};

/**
 * Owned raw model output for one call: F32 logits as produced by the BPU.
 */
struct MobileNetV2Raw
{
    std::vector<float> logits;  ///< Flat logits, class index by position.
};

/**
 * @brief MobileNetV2 inference model.
 *
 * Construction loads the packed HBM model, queries tensor properties and
 * allocates the input/output tensor buffers; any failure throws
 * std::runtime_error (the message carries the SDK error description) and
 * releases everything acquired so far. One instance is one runtime context;
 * the internal SDK buffers are reused across calls while every stage returns
 * owned data, so results stay valid across subsequent predicts. Instances
 * are neither copyable nor thread-safe.
 */
class MobileNetV2
{
public:
    /**
     * @brief Load the model and prepare its tensors.
     *
     * @param model_path Path to the quantized *.hbm model file.
     * @param resize_shorter Shorter edge before the center crop,
     *        int(input / crop_pct): 256 for both published variants.
     * @throws std::runtime_error when any DNN/UCP step or tensor allocation
     *         fails, or the model exposes no usable input/output tensor.
     */
    explicit MobileNetV2(const std::string& model_path, int resize_shorter = 256);

    /// Release tensor memory and model handles acquired at construction.
    ~MobileNetV2();

    MobileNetV2(const MobileNetV2&) = delete;
    MobileNetV2& operator=(const MobileNetV2&) = delete;

    /**
     * @brief Center-crop a BGR image into owned NV12 planes.
     *
     * Antialiased bicubic shorter-edge resize to resize_shorter (a bit-exact
     * port of Pillow's, shared with utils/tools/mobilenet/cpp/geometry.hpp),
     * center crop to the model input, BGR → YUV(I420), then repack U/V into
     * the interleaved NV12 UV plane: the geometry the published models were
     * calibrated and evaluated with, identical to the Python runtime.
     *
     * @param image BGR uint8 image; not modified.
     * @return Owned prepared input for infer.
     * @throws std::invalid_argument for an empty image, odd or non-square
     *         model input dimensions, or resize_shorter below the input size.
     */
    MobileNetV2Prepared preprocess(const cv::Mat& image) const;

    /**
     * @brief Upload one prepared input and run a synchronous BPU inference.
     *
     * Copies the owned NV12 planes into the reusable input tensor buffers
     * honoring their byte strides, cleans the input caches, submits the task
     * with default ANY-core scheduling at priority 0, waits for completion,
     * invalidates the output caches and copies the F32 output out into an
     * owned vector. The task handle is released on every exit path.
     *
     * @param input Prepared input from preprocess.
     * @return Owned raw logits for postprocess.
     * @throws std::runtime_error when any SDK step fails; the message carries
     *         the SDK error description.
     */
    MobileNetV2Raw infer(const MobileNetV2Prepared& input);

    /**
     * @brief Select the Top-K classes from the raw output.
     *
     * A numerically stable softmax turns the logits into probabilities; an
     * over-large top_k is clamped to the class count (matching the shared
     * Top-K helper).
     *
     * @param raw   Raw output from infer.
     * @param top_k Number of classes to keep.
     * @return Top-K results sorted by descending probability.
     * @throws std::invalid_argument when the raw output is empty.
     */
    std::vector<Classification> postprocess(const MobileNetV2Raw& raw,
                                            int top_k) const;

    /**
     * @brief Classify one image: preprocess → infer → postprocess.
     *
     * @param image BGR uint8 image.
     * @param top_k Number of classes to keep.
     * @return Top-K results owned by the caller.
     */
    std::vector<Classification> predict(const cv::Mat& image,
                                        int top_k);

    /// Model input width in pixels.
    int input_width() const;

    /// Model input height in pixels.
    int input_height() const;

private:
    struct Impl;                  ///< SDK state, defined in classify.cpp.
    std::unique_ptr<Impl> impl_;  ///< Never null after construction.
};
