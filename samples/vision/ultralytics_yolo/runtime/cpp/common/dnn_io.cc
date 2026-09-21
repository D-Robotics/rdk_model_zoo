/*
 * Copyright (c) 2026, D-Robotics.
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

#include "dnn_io.h"

#include <cstring>
#include <iostream>

#include "nv12_geometry.h"

namespace yolo {

namespace {

bool check_ret(int ret, const char* action) {
  if (ret == 0) return true;
  std::cerr << "[ERROR] " << action << " failed, error code: " << ret
            << std::endl;
  return false;
}

}  // namespace

InputPlan probe_input_protocol(hbDNNHandle_t model, std::string* error) {
  InputPlan plan;
  int32_t input_count = 0;
  if (!check_ret(hbDNNGetInputCount(&input_count, model),
                 "hbDNNGetInputCount")) {
    if (error) *error = "hbDNNGetInputCount failed";
    return plan;
  }

  if (input_count == 1) {
    hbDNNTensorProperties properties;
    std::memset(&properties, 0, sizeof(properties));
    if (!check_ret(hbDNNGetInputTensorProperties(&properties, model, 0),
                   "hbDNNGetInputTensorProperties")) {
      if (error) *error = "hbDNNGetInputTensorProperties failed";
      return plan;
    }
    if (properties.tensorType != HB_DNN_IMG_TYPE_NV12 ||
        properties.tensorLayout != HB_DNN_LAYOUT_NCHW ||
        properties.validShape.numDimensions != 4) {
      if (error) *error = "single input is not an NCHW NV12 tensor";
      return plan;
    }
    plan.protocol = InputProtocol::kPackedNv12;
    plan.input_h = properties.validShape.dimensionSize[2];
    plan.input_w = properties.validShape.dimensionSize[3];
    if ((plan.input_h & 1) != 0 || (plan.input_w & 1) != 0) {
      if (error) *error = "NV12 input shape must be even";
      plan.protocol = InputProtocol::kUnknown;
      return plan;
    }
    return plan;
  }

  if (input_count == 2) {
    hbDNNTensorProperties y_properties;
    hbDNNTensorProperties uv_properties;
    std::memset(&y_properties, 0, sizeof(y_properties));
    std::memset(&uv_properties, 0, sizeof(uv_properties));
    if (!check_ret(hbDNNGetInputTensorProperties(&y_properties, model, 0),
                   "hbDNNGetInputTensorProperties") ||
        !check_ret(hbDNNGetInputTensorProperties(&uv_properties, model, 1),
                   "hbDNNGetInputTensorProperties")) {
      if (error) *error = "hbDNNGetInputTensorProperties failed";
      return plan;
    }
    // UINT8 planes surface as HB_DNN_TENSOR_TYPE_S8 at the hbDNN API level.
    const hbDNNTensorShape& y_shape = y_properties.validShape;
    const hbDNNTensorShape& uv_shape = uv_properties.validShape;
    if (y_properties.tensorType != HB_DNN_TENSOR_TYPE_S8 ||
        uv_properties.tensorType != HB_DNN_TENSOR_TYPE_S8 ||
        y_shape.numDimensions != 4 || uv_shape.numDimensions != 4 ||
        y_shape.dimensionSize[0] != 1 || uv_shape.dimensionSize[0] != 1) {
      if (error) *error = "two inputs are not rank-4 UINT8 tensors";
      return plan;
    }
    const int h = y_shape.dimensionSize[1];
    const int w = y_shape.dimensionSize[2];
    if (y_shape.dimensionSize[3] != 1 || (h & 1) != 0 || (w & 1) != 0 ||
        uv_shape.dimensionSize[1] != h / 2 ||
        uv_shape.dimensionSize[2] != w / 2 ||
        uv_shape.dimensionSize[3] != 2) {
      if (error) *error = "split NV12 input shapes do not match the y/uv contract";
      return plan;
    }
    plan.protocol = InputProtocol::kSplitNv12;
    plan.input_h = h;
    plan.input_w = w;
    plan.y_stride = y_properties.stride[1];
    plan.uv_stride = uv_properties.stride[1];
    return plan;
  }

  if (error) *error = "unsupported input count: " + std::to_string(input_count);
  return plan;
}

bool Nv12Input::allocate(hbDNNHandle_t model, const InputPlan& plan) {
  release();
  if (plan.protocol == InputProtocol::kUnknown) {
    std::cerr << "[ERROR] Cannot allocate an unknown input protocol"
              << std::endl;
    return false;
  }

  input_count_ = plan.protocol == InputProtocol::kPackedNv12 ? 1 : 2;
  for (int i = 0; i < input_count_; ++i) {
    std::memset(&tensors_[i], 0, sizeof(tensors_[i]));
    if (!check_ret(hbDNNGetInputTensorProperties(&tensors_[i].properties,
                                                 model, i),
                   "hbDNNGetInputTensorProperties")) {
      return false;
    }
    if (!check_ret(hbSysAllocCachedMem(&tensors_[i].sysMem[0],
                                       tensors_[i].properties.alignedByteSize),
                   "hbSysAllocCachedMem(input)")) {
      return false;
    }
    allocated_[i] = true;
    std::memset(tensors_[i].sysMem[0].virAddr, 0,
                tensors_[i].properties.alignedByteSize);
  }
  return true;
}

bool Nv12Input::upload(const InputPlan& plan, const uint8_t* i420) {
  const int h = plan.input_h;
  const int w = plan.input_w;
  const int y_size = h * w;
  const uint8_t* y = i420;
  const uint8_t* u = i420 + y_size;
  const uint8_t* v = u + y_size / 4;

  if (plan.protocol == InputProtocol::kPackedNv12) {
    i420_to_packed_nv12(
        y, u, v, h, w,
        reinterpret_cast<uint8_t*>(tensors_[0].sysMem[0].virAddr));
    return check_ret(
        hbSysFlushMem(&tensors_[0].sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
        "hbSysFlushMem(input clean)");
  }
  if (plan.protocol == InputProtocol::kSplitNv12) {
    i420_to_split_nv12(
        y, u, v, h, w,
        reinterpret_cast<uint8_t*>(tensors_[0].sysMem[0].virAddr),
        plan.y_stride,
        reinterpret_cast<uint8_t*>(tensors_[1].sysMem[0].virAddr),
        plan.uv_stride);
    if (!check_ret(
            hbSysFlushMem(&tensors_[0].sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
            "hbSysFlushMem(y clean)")) {
      return false;
    }
    return check_ret(
        hbSysFlushMem(&tensors_[1].sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
        "hbSysFlushMem(uv clean)");
  }
  std::cerr << "[ERROR] Cannot upload an unknown input protocol" << std::endl;
  return false;
}

void Nv12Input::release() {
  for (int i = 0; i < 2; ++i) {
    if (allocated_[i]) {
      const int ret = hbSysFreeMem(&tensors_[i].sysMem[0]);
      if (ret != 0) {
        std::cerr << "[WARN] hbSysFreeMem(input) returned " << ret << std::endl;
      }
      allocated_[i] = false;
    }
  }
  input_count_ = 0;
}

}  // namespace yolo
