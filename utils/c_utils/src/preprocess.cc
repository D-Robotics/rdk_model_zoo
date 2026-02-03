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

/**
 * @file preprocess.cc
 * @brief Provide preprocessing utilities for preparing model inputs and tensor buffers.
 *
 * This file implements common preprocessing helpers used before inference, including
 * tensor buffer preparation and image-to-tensor data preparation. The utilities are
 * designed to be reusable across different model samples and to match device/runtime
 * requirements (e.g., alignment, stride handling, and input format expectations).
 */

#include <cmath>
#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <opencv2/imgproc.hpp>

#include "runtime.hpp"
#include "preprocess.hpp"

// BPU requires w stride alignment, S600 is aligned to 64, and the rest are aligned to 32.
#define ALIGN(value, alignment) (((value) + ((alignment)-1)) & ~((alignment)-1))
#ifdef SOC_S600
#define ALIGN_64(value) ALIGN(value, 64)
#define BPU_ALIGN(value) ALIGN_64(value)
#else
#define ALIGN_32(value) ALIGN(value, 32)
#define BPU_ALIGN(value) ALIGN_32(value)
#endif

/**
 * @brief Allocate and configure input tensors (handles dynamic stride).
 * @param[in,out] input_tensor  Vector of input tensors to be prepared (allocates sysMem, fixes stride).
 * @return int 0 on success, negative on failure.
 *
 * Notes:
 * - For input with dynamic stride (-1), we compute the actual byte stride and align to BPU requirement.
 * - Memory size is computed as stride[0] * N (bytes).
 */
int prepare_input_tensor(std::vector<hbDNNTensor>& input_tensor)
{
    hbDNNTensor* input = input_tensor.data();
    for (int i = 0; i < static_cast<int>(input_tensor.size()); i++) {
        // Resolve dynamic stride (-1) from last dim to first, then align to BPU boundary.
        auto dim_len = input[i].properties.validShape.numDimensions;
        for (int32_t dim_i = dim_len - 1; dim_i >= 0; --dim_i) {
            if (input[i].properties.stride[dim_i] == -1) {
                auto cur_stride =
                    input[i].properties.stride[dim_i + 1] *
                    input[i].properties.validShape.dimensionSize[dim_i + 1];
                input[i].properties.stride[dim_i] = BPU_ALIGN(cur_stride);  // align byte stride
            }
        }

        // Total bytes = outermost stride * N
        int input_memSize =
            static_cast<int>(input[i].properties.stride[0] *
            input[i].properties.validShape.dimensionSize[0]);

        HBDNN_CHECK_SUCCESS(hbUCPMallocCached(&input[i].sysMem, input_memSize, 0),
                            "hbUCPMallocCached failed");
    }
    return 0;
}

/**
 * @brief Allocate output tensors according to alignedByteSize.
 * @param[in,out] output_tensor  Vector of output tensors to allocate (sysMem).
 * @return int 0 on success, negative on failure.
 */
int prepare_output_tensor(std::vector<hbDNNTensor>& output_tensor)
{
    hbDNNTensor* output = output_tensor.data();
    for (int i = 0; i < static_cast<int>(output_tensor.size()); i++) {
        int output_memSize = output[i].properties.alignedByteSize;           // aligned bytes from runtime
        HBDNN_CHECK_SUCCESS(hbUCPMallocCached(&output[i].sysMem, output_memSize, 0),
                            "hbUCPMallocCached failed");
    }
    return 0;
}

/**
 * @brief Convert BGR image to planar NV12 tensors (Y + interleaved UV) and upload.
 * @param[in]     mat           BGR input image (H×W×3, uint8).
 * @param[in,out] input_tensor  Input tensor vector; expects 2 tensors: [0]=Y plane, [1]=UV plane.
 * @param[in]     input_h       Model input height (must be even).
 * @param[in]     input_w       Model input width  (must be even).
 * @return int32_t 0 on success, -1 on invalid size or failure.
 *
 * Details:
 * - Converts BGR → I420 (Y + U + V planar), then packs U/V into NV12 UV plane with per-row stride.
 * - Flushes caches to DDR before inference.
 */
int32_t bgr_to_nv12_tensor(cv::Mat& mat, std::vector<hbDNNTensor>& input_tensor, int input_h, int input_w)
{
  if (input_h % 2 || input_w % 2) {
    std::cout << "input img height and width must be even!" << std::endl;
    return -1;
  }

  // BGR -> I420 (Y + U + V planar)
  cv::Mat yuv_mat;
  cv::cvtColor(mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
  uint8_t* yuv_data = yuv_mat.ptr<uint8_t>();
  uint8_t* y_data_src = yuv_data;

  // ---- Copy Y plane with row stride padding ----
  uint8_t* y_data_dst = reinterpret_cast<uint8_t*>(input_tensor[0].sysMem.virAddr);
  for (int32_t h = 0; h < input_h; ++h) {
    memcpy(y_data_dst, y_data_src, input_w);                                // copy one row of Y
    y_data_src += input_w;
    y_data_dst += input_tensor[0].properties.stride[1];                     // jump by byte stride
  }

  // ---- Pack U+V into NV12 UV plane with row stride padding ----
  int32_t uv_height = input_tensor[1].properties.validShape.dimensionSize[1];
  int32_t uv_width  = input_tensor[1].properties.validShape.dimensionSize[2];
  uint8_t* uv_data_dst = reinterpret_cast<uint8_t*>(input_tensor[1].sysMem.virAddr);

  // I420 layout: Y( H*W ), then U( H/2 * W/2 ), then V( H/2 * W/2 )
  uint8_t* u_data_src = yuv_data + input_h * input_w;
  uint8_t* v_data_src = u_data_src + uv_height * uv_width;

  for (int32_t h = 0; h < uv_height; ++h) {
    auto* cur_data = uv_data_dst;
    for (int32_t w = 0; w < uv_width; ++w) {
      *cur_data++ = *u_data_src++;                                          // U
      *cur_data++ = *v_data_src++;                                          // V
    }
    uv_data_dst += input_tensor[1].properties.stride[1];                    // next UV row (bytes)
  }

  // Ensure data visible to BPU
  hbUCPMemFlush(&input_tensor[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);
  hbUCPMemFlush(&input_tensor[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);
  return 0;
}

/**
 * @brief Write CHW float32 data into a single NHWC/CHW tensor buffer with byte stride.
 * @param[in]     channels  Vector of C channel Mats, each CV_32F with size H×W (CHW layout as vector).
 * @param[in]     tensor    Target tensor vector; uses tensor[0] sysMem and stride for writes.
 * @return int 0 on success.
 *
 * Notes:
 * - Uses tensor[0].properties.stride to step rows correctly.
 * - Expects tensor[0] to hold a contiguous CHW float32 layout.
 */
int write_chw32_to_tensor(const std::vector<cv::Mat>& channels,
                          const std::vector<hbDNNTensor>& tensor)
{
    // Read shape for sanity/logging
    int C = tensor[0].properties.validShape.dimensionSize[1];
    int H = tensor[0].properties.validShape.dimensionSize[2];
    int W = tensor[0].properties.validShape.dimensionSize[3];


    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(tensor[0].sysMem.virAddr);
    const int64_t* stride = tensor[0].properties.stride;  // bytes: [N, C, H, W]

    // Copy row-by-row for each channel
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            float* dst_row = reinterpret_cast<float*>(
                base_ptr + c * stride[1] + h * stride[2]              // CHW strides (bytes)
            );
            const float* src_row = channels[c].ptr<float>(h);
            memcpy(dst_row, src_row, static_cast<size_t>(W) * sizeof(float));
        }
    }

    // Make visible to device
    hbUCPMemFlush(&tensor[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    return 0;
}

/**
 * @brief Resize with aspect ratio (letterbox) and pad to the target size.
 * @param[in]  src            Source image (any type supported by cv::resize).
 * @param[out] dst            Destination image with final target size (pre-created with target size).
 * @param[in]  padding_color  Padding value for all channels (e.g., 114 for YOLO).
 *
 * @details
 * - Keeps aspect ratio: scales by min(dst_w/src_w, dst_h/src_h), then symmetric pads.
 * - dst must be created with desired (dst_w, dst_h) before calling.
 */
void letterbox_resize(const cv::Mat& src, cv::Mat& dst, int padding_color)
{
    int src_w = src.cols;
    int src_h = src.rows;
    int dst_w = dst.cols;
    int dst_h = dst.rows;

    // Compute uniform scale
    float scale = std::min(dst_w / static_cast<float>(src_w),
                           dst_h / static_cast<float>(src_h));
    int new_w = static_cast<int>(std::round(src_w * scale));
    int new_h = static_cast<int>(std::round(src_h * scale));

    // Resize with aspect ratio
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));                    // keep AR

    // Compute symmetric padding
    int pad_top    = (dst_h - new_h) / 2;
    int pad_bottom = dst_h - new_h - pad_top;
    int pad_left   = (dst_w - new_w) / 2;
    int pad_right  = dst_w - new_w - pad_left;

    // Apply padding into dst
    cv::copyMakeBorder(resized, dst,
                       pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(padding_color, padding_color, padding_color));
}
