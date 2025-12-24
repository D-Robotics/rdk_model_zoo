/*
 * Copyright (c) 2025, XiangshunZhao D-Robotics.
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

#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core/mat.hpp>
#include "hobot/dnn/hb_dnn.h"

/**
 * @brief Allocate and configure input tensors (handles dynamic stride alignment).
 * @param[in,out] input_tensor  Vector of input tensors to prepare; this function fixes dynamic strides
 *                              (if any) and allocates cached sysMem for each tensor.
 * @return int 0 on success, negative value on failure.
 */
int prepare_input_tensor(std::vector<hbDNNTensor>& input_tensor);

/**
 * @brief Allocate output tensors according to their aligned byte size.
 * @param[in,out] output_tensor  Vector of output tensors; this function allocates cached sysMem
 *                               for each tensor based on properties.alignedByteSize.
 * @return int 0 on success, negative value on failure.
 */
int prepare_output_tensor(std::vector<hbDNNTensor>& output_tensor);

/**
 * @brief Convert a BGR image to NV12 planes and upload into model input tensors.
 * @param[in]     mat           Source image in BGR color space (CV_8UC3).
 * @param[in,out] input_tensor  Target input tensors; expects two tensors:
 *                              [0] Y plane, [1] interleaved UV plane. Function writes pixel data
 *                              with correct byte strides and flushes caches.
 * @param[in]     input_h       Target height (must be even for NV12).
 * @param[in]     input_w       Target width  (must be even for NV12).
 * @return int32_t 0 on success, -1 on invalid size or copy failure.
 */
int32_t bgr_to_nv12_tensor(cv::Mat& mat,
                           std::vector<hbDNNTensor>& input_tensor,
                           int input_h,
                           int input_w);

/**
 * @brief Write CHW float32 image data into a single tensor buffer honoring byte strides.
 * @param[in]     channels  Vector of channel Mats (size C), each CV_32F with shape H×W; represents CHW data.
 * @param[in]     tensor    Target tensor vector; only tensor[0] is used for the write. The function respects
 *                          tensor[0].properties.stride and flushes caches after the copy.
 * @return int 0 on success, negative value on failure.
 */
int write_chw32_to_tensor(const std::vector<cv::Mat>& channels,
                          const std::vector<hbDNNTensor>& tensor);

/**
 * @brief Resize with aspect ratio (letterbox) and pad into a pre-sized destination image.
 * @param[in]  src             Source image.
 * @param[out] dst             Destination image with preallocated target size (cols/rows define output size).
 * @param[in]  padding_color   Padding value applied to all channels (default: 127).
 *
 * @note Keeps aspect ratio by uniform scaling and symmetric padding to fit exactly into dst.
 */
void letterbox_resize(const cv::Mat& src,
                      cv::Mat& dst,
                      int padding_color = 127);
