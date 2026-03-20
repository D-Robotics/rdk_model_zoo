/*
 * Copyright (c) 2025 D-Robotics Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file paddle_ocr.hpp
 * @brief Define high-level inference wrappers and pipeline interfaces for the
 *        PaddleOCR text detection and recognition models.
 *
 * This file provides structured C++ interfaces encapsulating the complete
 * two-stage OCR inference workflow on D-Robotics S100 platforms:
 *
 *  - @ref PaddleOCRDet  — DB-algorithm text detection (NV12 input, int16 output)
 *  - @ref PaddleOCRRec  — CRNN text recognition (F32 NCHW input, CTC F32 output)
 *
 * Free functions for preprocessing, inference, and postprocessing are declared
 * alongside the class wrappers so that they can be used independently or
 * composed into custom pipelines.
 *
 * @note This sample only supports RDK S100 platform.
 *       RDK S600 is NOT supported; the models were compiled for S100 BPU.
 *
 * @see paddle_ocr.cpp
 */

#pragma once

#include "file_io.hpp"
#include "runtime.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"
#include "visualize.hpp"
#include <polyclipping/clipper.hpp>


// ---------------------------------------------------------------------------
// Configuration structs
// ---------------------------------------------------------------------------

/**
 * @brief Configuration parameters for the PaddleOCR text detection pipeline.
 *
 * @note This model only supports RDK S100 platform.
 */
struct PaddleOCRDetConfig
{
    float ratio_prime{2.7f};   ///< Contour dilation ratio (D' = area * ratio_prime / perimeter)
    float threshold{0.5f};     ///< Binarization threshold for the int16 prediction map
};

/**
 * @brief Configuration parameters for the PaddleOCR text recognition pipeline.
 *
 * No model-specific hyperparameters are required beyond the model path.
 *
 * @note This model only supports RDK S100 platform.
 */
struct PaddleOCRRecConfig
{
    // No additional hyperparameters required for the recognition model.
};


// ---------------------------------------------------------------------------
// TextDetResult
// ---------------------------------------------------------------------------

/**
 * @brief Intermediate result of the PaddleOCR text detection stage.
 *
 * Contains perspective-rectified text region crops (ready for the recognition
 * stage) and the corresponding 4-point polygon boxes, index-aligned.
 */
struct TextDetResult
{
    std::vector<cv::Mat>                crops;  ///< Perspective-rectified crop images, aligned with boxes
    std::vector<std::vector<cv::Point>> boxes;  ///< 4-point polygon boxes (pixel coordinates), aligned with crops
};


// ---------------------------------------------------------------------------
// PaddleOCRDet class
// ---------------------------------------------------------------------------

/**
 * @class PaddleOCRDet
 * @brief Wrapper class for PaddleOCR text detection using D-Robotics DNN / UCP APIs.
 *
 * This class encapsulates model loading, NV12 input tensor preparation, BPU
 * inference, and binary-map postprocessing for the DB-based text detector.
 *
 * Typical usage:
 * @code
 *   PaddleOCRDet det;
 *   det.init("/opt/hobot/model/s100/basic/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm");
 *   pre_process_det(det.input_tensors, img, det.input_w, det.input_h);
 *   infer(det.output_tensors, det.input_tensors, det.dnn_handle);
 *   TextDetResult result = post_process_det(det.output_tensors, img, threshold, ratio_prime);
 * @endcode
 *
 * @note This model only supports RDK S100 platform. Not thread-safe.
 */
class PaddleOCRDet
{
public:
    hbDNNHandle_t dnn_handle{nullptr};               ///< Handle of the loaded detection model
    std::vector<hbDNNTensor> input_tensors;          ///< Input tensor descriptors and memory
    std::vector<hbDNNTensor> output_tensors;         ///< Output tensor descriptors and memory
    int input_h{0};                                  ///< Model input height (pixels)
    int input_w{0};                                  ///< Model input width (pixels)

    /**
     * @brief Construct a PaddleOCRDet instance in an uninitialized state.
     *
     * No resources are allocated here. Call init() to load the model and
     * prepare tensor buffers.
     */
    PaddleOCRDet();

    /**
     * @brief Initialize model resources from a *.hbm model file.
     *
     * Performs:
     * - Loading the packed model from disk
     * - Selecting the first model handle in the pack
     * - Querying input/output tensor counts and properties
     * - Caching model input resolution (H, W) from NV12 HW layout
     *   (dimensionSize[1] = H, dimensionSize[2] = W)
     * - Allocating input/output tensor memory buffers
     *
     * @param[in] model_path Path to the quantized *.hbm detection model file (S100 only).
     * @retval 0        Success.
     * @retval non-zero DNN or UCP API error.
     */
    int32_t init(const char* model_path);

    /**
     * @brief Destructor.
     *
     * Releases all allocated tensor memory and DNN model resources.
     * Safe to call even if init() was never called or failed partially.
     */
    ~PaddleOCRDet();

private:
    int model_count_{0};                              ///< Number of models in the packed DNN handle
    hbDNNPackedHandle_t packed_dnn_handle_{nullptr};  ///< Packed DNN handle
    int32_t input_count_{0};                          ///< Number of input tensors
    int32_t output_count_{0};                         ///< Number of output tensors
    bool inited_{false};                              ///< Whether init() has been called successfully
};


// ---------------------------------------------------------------------------
// PaddleOCRRec class
// ---------------------------------------------------------------------------

/**
 * @class PaddleOCRRec
 * @brief Wrapper class for PaddleOCR text recognition using D-Robotics DNN / UCP APIs.
 *
 * This class encapsulates model loading, float32 NCHW input tensor preparation,
 * BPU inference, and CTC greedy decoding for the CRNN-based text recognizer.
 *
 * Typical usage:
 * @code
 *   PaddleOCRRec rec;
 *   rec.init("/opt/hobot/model/s100/basic/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm");
 *   pre_process_rec(rec.input_tensors, crop, rec.input_w, rec.input_h);
 *   infer(rec.output_tensors, rec.input_tensors, rec.dnn_handle);
 *   auto text = post_process_rec(rec.output_tensors, id2token, rec.seq_len, rec.num_classes);
 * @endcode
 *
 * @note This model only supports RDK S100 platform. Not thread-safe.
 */
class PaddleOCRRec
{
public:
    hbDNNHandle_t dnn_handle{nullptr};               ///< Handle of the loaded recognition model
    std::vector<hbDNNTensor> input_tensors;          ///< Input tensor descriptors and memory
    std::vector<hbDNNTensor> output_tensors;         ///< Output tensor descriptors and memory
    int input_h{0};                                  ///< Model input height (pixels)
    int input_w{0};                                  ///< Model input width (pixels)
    int seq_len{0};                                  ///< Output sequence length T
    int num_classes{0};                              ///< Output vocabulary size V (including blank)

    /**
     * @brief Construct a PaddleOCRRec instance in an uninitialized state.
     *
     * No resources are allocated here. Call init() to load the model and
     * prepare tensor buffers.
     */
    PaddleOCRRec();

    /**
     * @brief Initialize model resources from a *.hbm model file.
     *
     * Performs:
     * - Loading the packed model from disk
     * - Selecting the first model handle in the pack
     * - Querying input/output tensor counts and properties
     * - Caching model input resolution (H, W) from NCHW layout
     *   (dimensionSize[2] = H, dimensionSize[3] = W)
     * - Caching output sequence length (dimensionSize[1]) and vocabulary
     *   size (dimensionSize[2])
     * - Allocating input/output tensor memory buffers
     *
     * @param[in] model_path Path to the quantized *.hbm recognition model file (S100 only).
     * @retval 0        Success.
     * @retval non-zero DNN or UCP API error.
     */
    int32_t init(const char* model_path);

    /**
     * @brief Destructor.
     *
     * Releases all allocated tensor memory and DNN model resources.
     * Safe to call even if init() was never called or failed partially.
     */
    ~PaddleOCRRec();

private:
    int model_count_{0};                              ///< Number of models in the packed DNN handle
    hbDNNPackedHandle_t packed_dnn_handle_{nullptr};  ///< Packed DNN handle
    int32_t input_count_{0};                          ///< Number of input tensors
    int32_t output_count_{0};                         ///< Number of output tensors
    bool inited_{false};                              ///< Whether init() has been called successfully
};


// ---------------------------------------------------------------------------
// Free function declarations
// ---------------------------------------------------------------------------

/**
 * @brief Preprocess a BGR image into NV12 input tensors for the detection model.
 *
 * Resizes the image to ``(input_h, input_w)`` using INTER_AREA and converts
 * it to NV12 format, writing Y and UV planes into the two detection model
 * input tensors.
 *
 * @param[in,out] input_tensors  Detection model input tensors (two: Y and UV).
 * @param[in]     img            Input BGR image (OpenCV Mat).
 * @param[in]     input_w        Model input width in pixels.
 * @param[in]     input_h        Model input height in pixels.
 * @retval 0        Success.
 * @retval non-zero Preprocessing failure.
 */
int32_t pre_process_det(std::vector<hbDNNTensor>& input_tensors,
                        cv::Mat& img,
                        int input_w,
                        int input_h);

/**
 * @brief Execute synchronous BPU inference.
 *
 * Creates an inference task, submits it to the UCP scheduler with default
 * scheduling parameters (or the provided @p sched_param), waits for
 * completion, invalidates output caches for CPU access, and releases the
 * task handle.
 *
 * @param[in,out] output_tensors Output tensors filled by the runtime.
 * @param[in]     input_tensors  Prepared input tensors.
 * @param[in]     dnn_handle     DNN model handle used for inference.
 * @param[in]     sched_param    Optional UCP scheduling parameters (nullptr = default).
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              const hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param = nullptr);

/**
 * @brief Postprocess detection output tensors into cropped regions and polygon boxes.
 *
 * Steps:
 * 1. Threshold the int16 prediction map at ``static_cast<int16_t>(threshold)``
 *    and resize to the original image dimensions (binary mask).
 * 2. Find external contours on the binary mask.
 * 3. Dilate polygons using ClipperLib offset with @p ratio_prime.
 * 4. Convert dilated polygons to minimum-area bounding boxes (min_area = 100).
 * 5. Perspective-crop and rectify each text region from @p image.
 *
 * @param[in] output_tensors  Detection model output tensors.
 * @param[in] image           Original BGR image used for cropping.
 * @param[in] threshold       Float binarization threshold (cast to int16 internally).
 * @param[in] ratio_prime     Contour dilation scale factor.
 * @return TextDetResult containing crops and boxes, index-aligned.
 */
TextDetResult post_process_det(std::vector<hbDNNTensor>& output_tensors,
                               cv::Mat& image,
                               float threshold,
                               float ratio_prime);

/**
 * @brief Preprocess a cropped BGR word image into float32 NCHW input tensors.
 *
 * Steps:
 * 1. BGR -> RGB
 * 2. Resize to ``(input_w, input_h)`` using INTER_AREA
 * 3. Convert to float32 in [0, 1]
 * 4. Per-channel ImageNet normalization (mean 0.485/0.456/0.406, std 0.229/0.224/0.225)
 * 5. Write CHW layout into recognition model input tensor
 *
 * @param[in,out] input_tensors  Recognition model input tensors (one: float32 NCHW).
 * @param[in]     img            Cropped text region in BGR format.
 * @param[in]     input_w        Model input width in pixels.
 * @param[in]     input_h        Model input height in pixels.
 * @retval 0        Success.
 * @retval non-zero Preprocessing failure.
 */
int32_t pre_process_rec(std::vector<hbDNNTensor>& input_tensors,
                        cv::Mat& img,
                        int input_w,
                        int input_h);

/**
 * @brief Decode recognition CTC logits into a UTF-8 text string.
 *
 * Performs stride-aware greedy CTC decoding over the output tensor:
 * argmax per timestep, collapse consecutive repeats, skip blank (id 0).
 *
 * @param[in] output_tensors  Recognition model output tensors ([1, T, V] float32).
 * @param[in] id2token        Token dictionary; id2token[i] maps class id i to a string.
 * @param[in] seq_len         Sequence length T.
 * @param[in] num_classes     Vocabulary size V (including blank at index 0).
 * @return std::string Decoded text string.
 */
std::string post_process_rec(std::vector<hbDNNTensor>& output_tensors,
                              const std::vector<std::string>& id2token,
                              int seq_len,
                              int num_classes);
