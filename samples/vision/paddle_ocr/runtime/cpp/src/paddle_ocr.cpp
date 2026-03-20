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
 * @file paddle_ocr.cpp
 * @brief Implement PaddleOCR text detection and recognition inference pipeline.
 *
 * This file provides the complete implementation of:
 *  - PaddleOCRDet  — model init, destructor
 *  - PaddleOCRRec  — model init, destructor
 *  - pre_process_det  — BGR -> NV12 tensor preparation
 *  - infer            — synchronous BPU inference
 *  - post_process_det — int16 mask -> contours -> dilated boxes -> crops
 *  - pre_process_rec  — BGR -> RGB -> float32 CHW -> tensor
 *  - post_process_rec — CTC greedy decode
 *
 * Local helper functions (process_and_resize_pred, dilate_contours,
 * get_bounding_boxes, crop_and_rotate_image, ctc_greedy_decode_from_tensor)
 * are defined at file scope and are not exposed in the header.
 *
 * @see paddle_ocr.hpp
 */

#include "paddle_ocr.hpp"
#include "postprocess.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>


// ===========================================================================
// Local helper functions
// ===========================================================================

/**
 * @brief Threshold an int16 prediction map and resize to the original image size.
 *
 * Values greater than @p threshold are set to 255, all others to 0.
 * The result is resized to (img_h, img_w) using bilinear interpolation.
 *
 * @param[in] tensor     hbDNNTensor holding int16 prediction data.
 * @param[in] threshold  Binarization threshold in int16 domain.
 * @param[in] img_w      Target width to resize to.
 * @param[in] img_h      Target height to resize to.
 * @return cv::Mat       Binary mask (CV_8UC1) at (img_h, img_w).
 */
static cv::Mat process_and_resize_pred(const hbDNNTensor& tensor,
                                       int16_t threshold,
                                       int img_w, int img_h)
{
    const hbDNNTensorShape& shape = tensor.properties.validShape;
    int H = shape.dimensionSize[shape.numDimensions - 2];
    int W = shape.dimensionSize[shape.numDimensions - 1];

    const int16_t* data = reinterpret_cast<const int16_t*>(tensor.sysMem.virAddr);

    cv::Mat preds_bin(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            preds_bin.at<uint8_t>(y, x) = (data[idx] > threshold) ? 255 : 0;
        }
    }

    cv::Mat preds_resized;
    cv::resize(preds_bin, preds_resized, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

    return preds_resized;
}

/**
 * @brief Dilate (offset) contours by a data-driven scale using ClipperOffset.
 *
 * For each polygon, computes offset distance D' = area * ratio_prime / perimeter,
 * then performs polygon offsetting via ClipperLib with round joins.
 * Results that are empty or produce multiple polygons are skipped.
 *
 * @param[in] contours     Input polygons (each as a vector of cv::Point).
 * @param[in] ratio_prime  Scale factor for offset distance computation.
 * @return std::vector<std::vector<cv::Point>>  Dilated polygons.
 */
static std::vector<std::vector<cv::Point>> dilate_contours(
    const std::vector<std::vector<cv::Point>>& contours,
    float ratio_prime)
{
    std::vector<std::vector<cv::Point>> dilated_polys;

    auto safe_cast_coord = [](long long v) -> int {
        if (v > std::numeric_limits<int>::max()) return std::numeric_limits<int>::max();
        if (v < std::numeric_limits<int>::min()) return std::numeric_limits<int>::min();
        return static_cast<int>(v);
    };

    for (size_t idx = 0; idx < contours.size(); ++idx) {
        const auto& poly = contours[idx];

        double arc_length = cv::arcLength(poly, true);
        if (arc_length == 0) {
            std::cerr << "[Skip] Contour " << idx << " has zero arc length\n";
            continue;
        }

        double area = cv::contourArea(poly);
        double D_prime = area * ratio_prime / arc_length;

        ClipperLib::Path path;
        path.reserve(poly.size());
        for (const auto& pt : poly) {
            path.push_back(ClipperLib::IntPoint(pt.x, pt.y));
        }

        ClipperLib::ClipperOffset pco;
        pco.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

        ClipperLib::Paths solution;
        pco.Execute(solution, D_prime);

        if (solution.size() != 1) {
            std::cerr << "[Skip] Contour " << idx
                      << " offset result size != 1 (" << solution.size() << ")\n";
            continue;
        }

        const auto& sol_path = solution[0];
        if (sol_path.empty()) {
            std::cerr << "[Skip] Contour " << idx << " offset result empty\n";
            continue;
        }

        std::vector<cv::Point> cv_poly;
        cv_poly.reserve(sol_path.size());
        for (const auto& ipt : sol_path) {
            cv_poly.emplace_back(safe_cast_coord(ipt.X), safe_cast_coord(ipt.Y));
        }

        dilated_polys.push_back(std::move(cv_poly));
    }

    return dilated_polys;
}

/**
 * @brief Greedy CTC decode directly from a logits tensor with stride-aware access.
 *
 * Scans each timestep row, finds argmax over classes, collapses consecutive
 * repeats, and ignores the CTC blank (id 0). Concatenates dictionary tokens.
 *
 * @param[in] data         Pointer to float32 logits buffer.
 * @param[in] stride       Byte stride array for the tensor (uses stride[1] as row step).
 * @param[in] seq_len      Number of sequence timesteps.
 * @param[in] num_classes  Number of classes (vocabulary size including blank).
 * @param[in] id2token     Token dictionary; id2token[i] maps class id to string.
 * @return std::string     Decoded text string.
 */
static std::string ctc_greedy_decode_from_tensor(const float* data,
                                                  const int64_t* stride,
                                                  int seq_len,
                                                  int num_classes,
                                                  const std::vector<std::string>& id2token)
{
    std::string result;
    int prev_idx = -1;

    for (int t = 0; t < seq_len; ++t) {
        const float* row_ptr = reinterpret_cast<const float*>(
            reinterpret_cast<const uint8_t*>(data) + t * stride[1]
        );

        int idx = static_cast<int>(
            std::max_element(row_ptr, row_ptr + num_classes) - row_ptr);

        if (idx != 0 && idx != prev_idx) {
            result += id2token[idx];
        }
        prev_idx = idx;
    }
    return result;
}


// ===========================================================================
// PaddleOCRDet implementation
// ===========================================================================

PaddleOCRDet::PaddleOCRDet()
{
    // Default constructor; resources allocated in init()
}

int32_t PaddleOCRDet::init(const char* model_path)
{
    const char** model_name_list = nullptr;

    HBDNN_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_path, 1),
        "hbDNNInitializeFromFiles failed");
    HBDNN_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count_, packed_dnn_handle_),
        "hbDNNGetModelNameList failed");
    HBDNN_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle_, model_name_list[0]),
        "hbDNNGetModelHandle failed");

    HBDNN_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count_, dnn_handle),
        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count_, dnn_handle),
        "hbDNNGetOutputCount failed");

    input_tensors.resize(input_count_);
    output_tensors.resize(output_count_);

    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");
    }
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
    }

    // NV12 HW layout: dimensionSize[1] = H, dimensionSize[2] = W
    input_h = input_tensors[0].properties.validShape.dimensionSize[1];
    input_w = input_tensors[0].properties.validShape.dimensionSize[2];

    prepare_input_tensor(input_tensors);
    prepare_output_tensor(output_tensors);

    inited_ = true;
    return 0;
}

PaddleOCRDet::~PaddleOCRDet()
{
    if (!inited_) return;

    for (int i = 0; i < input_count_; i++) {
        hbUCPFree(&(input_tensors[i].sysMem));
    }
    for (int i = 0; i < output_count_; i++) {
        hbUCPFree(&(output_tensors[i].sysMem));
    }
    hbDNNRelease(packed_dnn_handle_);
}


// ===========================================================================
// PaddleOCRRec implementation
// ===========================================================================

PaddleOCRRec::PaddleOCRRec()
{
    // Default constructor; resources allocated in init()
}

int32_t PaddleOCRRec::init(const char* model_path)
{
    const char** model_name_list = nullptr;

    HBDNN_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_path, 1),
        "hbDNNInitializeFromFiles failed");
    HBDNN_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count_, packed_dnn_handle_),
        "hbDNNGetModelNameList failed");
    HBDNN_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle_, model_name_list[0]),
        "hbDNNGetModelHandle failed");

    HBDNN_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count_, dnn_handle),
        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count_, dnn_handle),
        "hbDNNGetOutputCount failed");

    input_tensors.resize(input_count_);
    output_tensors.resize(output_count_);

    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");
    }
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
    }

    // NCHW layout: dimensionSize[2] = H, dimensionSize[3] = W
    input_h = input_tensors[0].properties.validShape.dimensionSize[2];
    input_w = input_tensors[0].properties.validShape.dimensionSize[3];

    // Output [N, T, V]: seq_len at dim[1], num_classes at dim[2]
    seq_len     = output_tensors[0].properties.validShape.dimensionSize[1];
    num_classes = output_tensors[0].properties.validShape.dimensionSize[2];

    prepare_input_tensor(input_tensors);
    prepare_output_tensor(output_tensors);

    inited_ = true;
    return 0;
}

PaddleOCRRec::~PaddleOCRRec()
{
    if (!inited_) return;

    for (int i = 0; i < input_count_; i++) {
        hbUCPFree(&(input_tensors[i].sysMem));
    }
    for (int i = 0; i < output_count_; i++) {
        hbUCPFree(&(output_tensors[i].sysMem));
    }
    hbDNNRelease(packed_dnn_handle_);
}


// ===========================================================================
// Free function implementations
// ===========================================================================

int32_t pre_process_det(std::vector<hbDNNTensor>& input_tensors,
                        cv::Mat& img,
                        int input_w,
                        int input_h)
{
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_AREA);

    bgr_to_nv12_tensor(resized, input_tensors, input_h, input_w);

    return 0;
}

int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              const hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param)
{
    hbUCPTaskHandle_t task_handle{nullptr};

    HBDNN_CHECK_SUCCESS(
        hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle),
        "hbDNNInferV2 failed");

    hbUCPSchedParam default_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&default_param);
    default_param.backend = HB_UCP_BPU_CORE_ANY;

    hbUCPSchedParam* param = (sched_param != nullptr) ? sched_param : &default_param;

    HBUCP_CHECK_SUCCESS(
        hbUCPSubmitTask(task_handle, param),
        "hbUCPSubmitTask failed");

    HBUCP_CHECK_SUCCESS(
        hbUCPWaitTaskDone(task_handle, 0),
        "hbUCPWaitTaskDone failed");

    int output_count = static_cast<int>(output_tensors.size());
    for (int i = 0; i < output_count; i++) {
        hbUCPMemFlush(&output_tensors[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    HBUCP_CHECK_SUCCESS(
        hbUCPReleaseTask(task_handle),
        "hbUCPReleaseTask failed");

    return 0;
}

TextDetResult post_process_det(std::vector<hbDNNTensor>& output_tensors,
                               cv::Mat& image,
                               float threshold,
                               float ratio_prime)
{
    // 1) Threshold int16 prediction map and resize to original image size
    auto preds = process_and_resize_pred(output_tensors[0],
                                         static_cast<int16_t>(threshold),
                                         image.cols, image.rows);

    // 2) Find external contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(preds, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 3) Dilate polygons
    auto dilated_polys = dilate_contours(contours, ratio_prime);

    // 4) Convert to min-area bounding boxes
    auto boxes_list = get_bounding_boxes(dilated_polys, 100.f);

    // 5) Crop and rectify each detected box
    TextDetResult result;
    result.boxes = std::move(boxes_list);
    result.crops.reserve(result.boxes.size());
    for (const auto& box : result.boxes) {
        result.crops.push_back(crop_and_rotate_image(image, box));
    }

    return result;
}

int32_t pre_process_rec(std::vector<hbDNNTensor>& input_tensors,
                        cv::Mat& img,
                        int input_w,
                        int input_h)
{
    // 1. BGR -> RGB
    cv::Mat rgb_mat;
    cv::cvtColor(img, rgb_mat, cv::COLOR_BGR2RGB);

    // 2. Resize to model input
    cv::Mat resized;
    cv::resize(rgb_mat, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_AREA);

    // 3. To float32 in [0,1]
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

    // 4. Per-channel ImageNet normalization
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);  // R, G, B

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_val[3]  = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean[c]) / std_val[c];
    }

    // 5. Write CHW float32 into input tensor
    write_chw32_to_tensor(channels, input_tensors);

    return 0;
}

std::string post_process_rec(std::vector<hbDNNTensor>& output_tensors,
                              const std::vector<std::string>& id2token,
                              int seq_len,
                              int num_classes)
{
    const float*   data   = reinterpret_cast<const float*>(output_tensors[0].sysMem.virAddr);
    const int64_t* stride = output_tensors[0].properties.stride;

    return ctc_greedy_decode_from_tensor(data, stride, seq_len, num_classes, id2token);
}
