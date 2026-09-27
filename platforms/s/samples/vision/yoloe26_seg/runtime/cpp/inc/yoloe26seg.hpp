// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file yoloe26seg.hpp
 * @brief Public C++ pipeline for the YOLOE-26 prompt-free segmenter.
 *
 * The interface follows the Model Zoo C++ convention: constructing the model
 * object is cheap, init() owns all runtime resources, and pre_process(),
 * infer(), and post_process() can be used separately when an application
 * needs to schedule or batch those stages itself.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "model_types.hpp"

namespace yoloe26 {

/**
 * @brief Runtime and decoding options for YOLOE-26 segmentation.
 *
 * An empty model_path asks init() to infer the board march and use the
 * repository's downloaded model for model_size. The default configuration is
 * therefore suitable for the n model after run.sh or download_model.sh has
 * populated the model directory.
 */
struct YoloE26SegConfig {
    std::string model_path{};       ///< HBM path; empty selects the board-aware default.
    std::string model_size{"n"};  ///< Released model size: n, s, m, l, or x.
    float score_threshold{0.25f};  ///< Sigmoid confidence threshold in (0, 1).
    int max_det{300};              ///< Maximum number of retained candidates.
    bool single_label{true};       ///< Keep the best class at each anchor.
};

/**
 * @brief Prepare a BGR image in the model input tensors.
 *
 * @param[in,out] input_tensors Two NV12 tensors (Y and interleaved UV).
 * @param[in] image Source BGR CV_8UC3 image.
 * @param[in] input_w Model input width.
 * @param[in] input_h Model input height.
 * @param[in] image_format Only "BGR" is supported.
 * @return 0 on success, otherwise a negative error code.
 */
int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                    const cv::Mat& image,
                    int input_w,
                    int input_h,
                    const std::string& image_format = "BGR");

/**
 * @brief Run synchronous BPU inference on prepared tensors.
 *
 * When sched_param is null, hbDNNInferV2(nullptr, ...) is intentionally used:
 * this is the supported synchronous single-core path. A non-null parameter
 * enables the optional UCP submit/wait path.
 *
 * @param[in,out] output_tensors Output tensors to fill.
 * @param[in,out] input_tensors Prepared input tensors.
 * @param[in] dnn_handle Loaded model handle.
 * @param[in] sched_param Optional UCP scheduling parameters.
 * @return 0 on success, otherwise the underlying runtime error code.
 */
int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param = nullptr);

/**
 * @brief Decode raw model outputs into source-image detections and masks.
 *
 * Each output mask is CV_8UC1 with values 0 or 1 and is local to the clipped,
 * integer-truncated source-image bounding box. Empty or degenerate instances
 * retain an empty mask so vector indices remain aligned with detections.
 *
 * @param[in] output_tensors Ten raw NHWC output tensors in protocol order.
 * @param[in] config Candidate selection options.
 * @param[in] source_width Original image width.
 * @param[in] source_height Original image height.
 * @param[in] input_width Model canvas width used for preprocessing.
 * @param[in] input_height Model canvas height used for preprocessing.
 * @return Index-aligned common Model Zoo instance segmentation result.
 */
InstanceSegResult post_process(const std::vector<hbDNNTensor>& output_tensors,
                               const YoloE26SegConfig& config,
                               int source_width,
                               int source_height,
                               int input_width = 640,
                               int input_height = 640);

/**
 * @brief High-level owner for one YOLOE-26 inference thread.
 *
 * Construction only stores configuration. Call init() explicitly before
 * predict(); all model and tensor resources are released by the destructor.
 */
class YoloE26Seg {
public:
    /**
     * @brief Construct an uninitialized model.
     * @param[in] config Runtime options copied without loading the HBM.
     */
    explicit YoloE26Seg(YoloE26SegConfig config = YoloE26SegConfig{});

    /** @brief Release model, tensor, and packed-HBM resources. */
    ~YoloE26Seg();

    YoloE26Seg(const YoloE26Seg&) = delete;
    YoloE26Seg& operator=(const YoloE26Seg&) = delete;
    YoloE26Seg(YoloE26Seg&&) = delete;
    YoloE26Seg& operator=(YoloE26Seg&&) = delete;

    /**
     * @brief Initialize the model and allocate all tensor buffers.
     *
     * @param[in] model_path Optional HBM path. Null or empty uses the path in
     *                       the configuration, then the board-aware default.
     * @return 0 on success; non-zero on any validation, DNN, or allocation
     *         failure. Errors are contained and do not escape this function.
     */
    int32_t init(const char* model_path = nullptr) noexcept;

    /**
     * @brief Run preprocessing, synchronous inference, and postprocessing with the configured options.
     *
     * @param[in] image Source BGR CV_8UC3 image.
     * @return Common Model Zoo instance segmentation result.
     * @throws std::runtime_error if the model is not initialized or a staged
     *         operation fails; std::invalid_argument for invalid options.
     */
    InstanceSegResult predict(const cv::Mat& image);

    /**
     * @brief Report whether init() completed successfully.
     * @return true after successful initialization, otherwise false.
     */
    bool initialized() const noexcept { return inited_; }

    /**
     * @brief Return the board-aware default HBM path for a released size.
     *
     * @param[in] model_size Released model size: n, s, m, l, or x.
     * @return Absolute path under this sample's downloaded model directory.
     * @throws std::invalid_argument for an unsupported size or board.
     *
     * The result is independent of the current working directory.
     */
    static std::string default_model_path(const std::string& model_size = "n");

    /** @brief Loaded DNN model handle used by the staged infer() function. */
    hbDNNHandle_t dnn_handle{nullptr};
    /** @brief Prepared input tensors (Y and UV). */
    std::vector<hbDNNTensor> input_tensors;
    /** @brief Allocated output tensors in raw protocol order. */
    std::vector<hbDNNTensor> output_tensors;
    /** @brief Model input height in pixels. */
    int input_h{0};
    /** @brief Model input width in pixels. */
    int input_w{0};

private:
    /** @brief Release all owned runtime resources; safe after partial init. */
    void release() noexcept;

    YoloE26SegConfig config_;
    hbDNNPackedHandle_t packed_dnn_handle_{nullptr};
    int input_count_{0};
    int output_count_{0};
    bool inited_{false};
};

}  // namespace yoloe26
