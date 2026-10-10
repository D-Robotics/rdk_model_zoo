// Copyright (c) 2025 D-Robotics Corporation
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file classify.hpp
 * @brief MobileNetV2 classification model interface.
 *
 * MobileNetV2 owns the loaded HBM runtime. Each stage exchanges owned
 * per-call data: preprocess letterboxes a BGR image into owned NV12 planes,
 * infer uploads one prepared input into the reusable SDK buffers and returns
 * the owned raw output copied out of the output tensor, and postprocess
 * selects the Top-K classes. The model's output node already contains
 * post-softmax probabilities, so postprocess reads them directly and applies
 * no further normalization. predict is the explicit composition of the three
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
 * Owned raw model output for one call: F32 probabilities as produced by the
 * BPU. The model's "prob" output is already a probability distribution.
 */
struct MobileNetV2Raw
{
    std::vector<float> probabilities;  ///< Flat probabilities, class index
                                       ///< by position.
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
     * @throws std::runtime_error when any DNN/UCP step or tensor allocation
     *         fails, or the model exposes no usable input/output tensor.
     */
    explicit MobileNetV2(const std::string& model_path);

    /// Release tensor memory and model handles acquired at construction.
    ~MobileNetV2();

    MobileNetV2(const MobileNetV2&) = delete;
    MobileNetV2& operator=(const MobileNetV2&) = delete;

    /**
     * @brief Letterbox-resize a BGR image into owned NV12 planes.
     *
     * Uniform-scale letterbox with 127-gray padding, BGR → YUV(I420), then
     * repack U/V into the interleaved NV12 UV plane. The geometry is computed
     * per call; nothing is cached on the instance.
     *
     * @param image BGR uint8 image; not modified.
     * @return Owned prepared input for infer.
     * @throws std::invalid_argument for an empty image or odd model input
     *         dimensions.
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
     * @return Owned raw probabilities for postprocess.
     * @throws std::runtime_error when any SDK step fails; the message carries
     *         the SDK error description.
     */
    MobileNetV2Raw infer(const MobileNetV2Prepared& input);

    /**
     * @brief Select the Top-K classes from the raw output.
     *
     * The output node already contains probabilities, so they are read
     * directly without any normalization; an over-large top_k is clamped to
     * the class count (matching the shared Top-K helper).
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
