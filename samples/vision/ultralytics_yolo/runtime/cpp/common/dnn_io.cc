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
#include "nv12_geometry.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace yolo {
namespace {
bool check_ret(int ret, const char *action) {
  if (ret == 0)
    return true;
  std::cerr << "[ERROR] " << action << " failed, error code: " << ret
            << std::endl;
  return false;
}
bool same_plan(const InputPlan &a, const InputPlan &b) {
  return a.protocol == b.protocol && a.input_h == b.input_h &&
         a.input_w == b.input_w && a.y_stride == b.y_stride &&
         a.uv_stride == b.uv_stride;
}
bool geometry(int h, int w) {
  // All SDK allocations are signed-int byte counts, including NV12's UV plane.
  return h > 0 && w > 0 && !(h & 1) && !(w & 1) &&
         static_cast<int64_t>(h) * w <= std::numeric_limits<int>::max() / 3 * 2;
}
bool split_properties(hbDNNTensorProperties &p, int h, int w, int channels) {
  if (p.quantiType != NONE || p.validShape.numDimensions != 4 ||
      p.validShape.dimensionSize[0] != 1 ||
      p.validShape.dimensionSize[1] != h ||
      p.validShape.dimensionSize[2] != w ||
      p.validShape.dimensionSize[3] != channels)
    return false;
  bool type_ok = p.tensorType == HB_DNN_TENSOR_TYPE_S8;
#if !defined(YOLO_DNN_STACK_X5)
  type_ok = type_ok || p.tensorType == HB_DNN_TENSOR_TYPE_U8;
#endif
  if (!type_ok)
    return false;
  const int64_t row_bytes = static_cast<int64_t>(w) * channels;
  const int64_t aligned_row = (row_bytes + 63) / 64 * 64;
  if (aligned_row > std::numeric_limits<int>::max())
    return false;
  if (p.stride[3] == -1)
    p.stride[3] = 1;
  if (p.stride[2] == -1)
    p.stride[2] = channels;
  if (p.stride[1] == -1)
    p.stride[1] = aligned_row;
  if (p.stride[3] != 1 || p.stride[2] != channels || p.stride[1] < row_bytes)
    return false;
  const int64_t plane_bytes = static_cast<int64_t>(p.stride[1]) * h;
  if (plane_bytes <= 0 || plane_bytes > std::numeric_limits<int>::max())
    return false;
  if (p.stride[0] == -1)
    p.stride[0] = plane_bytes;
  if (p.stride[0] < plane_bytes)
    return false;
  if (p.alignedByteSize == -1)
    p.alignedByteSize = p.stride[0];
  return p.alignedByteSize >= p.stride[0];
}
// One path resolves both probe and allocation metadata, before any allocation.
bool inspect(hbDNNHandle_t model, InputPlan &plan,
             hbDNNTensorProperties (&properties)[2], std::string *error) {
  auto fail = [&](const char *text) {
    if (error)
      *error = text;
    return false;
  };
  int32_t count = 0;
  if (hbDNNGetInputCount(&count, model) != 0 || (count != 1 && count != 2))
    return fail("Expected one packed or two split NV12 inputs");
  for (int i = 0; i < count; ++i)
    if (hbDNNGetInputTensorProperties(&properties[i], model, i) != 0)
      return fail("Cannot query NV12 input descriptor");
  if (count == 1) {
#if defined(YOLO_DNN_STACK_X5)
    const auto &p = properties[0];
    if (p.tensorType != HB_DNN_IMG_TYPE_NV12 || p.quantiType != NONE ||
        p.validShape.numDimensions != 4 || p.alignedShape.numDimensions != 4 ||
        p.validShape.dimensionSize[0] != 1 ||
        (p.tensorLayout != HB_DNN_LAYOUT_NCHW &&
         p.tensorLayout != HB_DNN_LAYOUT_NHWC))
      return fail("Expected a batch-one packed NV12 input");
    const int channel = p.tensorLayout == HB_DNN_LAYOUT_NCHW ? 1 : 3;
    const int height = p.tensorLayout == HB_DNN_LAYOUT_NCHW ? 2 : 1;
    const int width = p.tensorLayout == HB_DNN_LAYOUT_NCHW ? 3 : 2;
    if (p.validShape.dimensionSize[channel] != 3)
      return fail("Packed NV12 descriptor must declare three RGB channels");
    for (int i = 0; i < 4; ++i)
      if (p.validShape.dimensionSize[i] != p.alignedShape.dimensionSize[i])
        return fail("Padded packed NV12 storage is unsupported");
    const int h = p.validShape.dimensionSize[height],
              w = p.validShape.dimensionSize[width];
    if (!geometry(h, w) ||
        p.alignedByteSize < static_cast<int64_t>(h) * w * 3 / 2)
      return fail("Invalid packed NV12 geometry or allocation");
    plan.protocol = InputProtocol::kPackedNv12;
    plan.input_h = h;
    plan.input_w = w;
    return true;
#else
    return fail("Packed NV12 requires the X5 stack");
#endif
  }
  const auto &shape = properties[0].validShape;
  if (shape.numDimensions != 4)
    return fail("Expected rank-four split NV12");
  const int h = shape.dimensionSize[1], w = shape.dimensionSize[2];
  if (!geometry(h, w) || !split_properties(properties[0], h, w, 1) ||
      !split_properties(properties[1], h / 2, w / 2, 2))
    return fail("Invalid split NV12 shape/type/stride/allocation");
  plan.protocol = InputProtocol::kSplitNv12;
  plan.input_h = h;
  plan.input_w = w;
  plan.y_stride = properties[0].stride[1];
  plan.uv_stride = properties[1].stride[1];
  return true;
}
} // namespace

int infer_sync(hbDNNTensor *outputs, hbDNNTensor *inputs, int input_count,
               hbDNNHandle_t model) {
  if (!outputs || !inputs || !model || (input_count != 1 && input_count != 2))
    return -1;
#if defined(YOLO_DNN_STACK_X5)
  hbDNNTaskHandle_t task = nullptr;
  hbDNNInferCtrlParam control;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
  hbDNNTensor *output_ptr = outputs;
  int result = hbDNNInfer(&task, &output_ptr, inputs, model, &control);
  if (result == 0)
    result = task ? hbDNNWaitTaskDone(task, 0) : -1;
  const int released = task ? hbDNNReleaseTask(task) : 0;
#else
  hbUCPTaskHandle_t task = nullptr;
  int result = hbDNNInferV2(&task, outputs, inputs, model);
  if (result == 0 && !task)
    result = -1;
  if (result == 0) {
    hbUCPSchedParam sched;
    HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
    sched.backend = HB_UCP_BPU_CORE_ANY;
    result = hbUCPSubmitTask(task, &sched);
    if (result == 0)
      result = hbUCPWaitTaskDone(task, 0);
  }
  const int released = task ? hbUCPReleaseTask(task) : 0;
#endif
  return result != 0 ? result : released;
}
InputPlan probe_input_protocol(hbDNNHandle_t model, std::string *error) {
  InputPlan plan;
  hbDNNTensorProperties properties[2]{};
  if (!inspect(model, plan, properties, error))
    return InputPlan{};
  if (error)
    error->clear();
  return plan;
}
bool Nv12Input::allocate(hbDNNHandle_t model, const InputPlan &plan) {
  release();
  InputPlan actual;
  hbDNNTensorProperties properties[2]{};
  std::string error;
  if (!inspect(model, actual, properties, &error) || !same_plan(plan, actual)) {
    std::cerr << "[ERROR] NV12 allocation contract: " << error << std::endl;
    return false;
  }
  input_count_ = actual.protocol == InputProtocol::kPackedNv12 ? 1 : 2;
  for (int i = 0; i < input_count_; ++i) {
    std::memset(&tensors_[i], 0, sizeof(tensors_[i]));
    tensors_[i].properties = properties[i];
    auto *memory = YOLO_SYS_MEM(tensors_[i]);
    int rc = YOLO_SYS_ALLOC_CACHED(memory, properties[i].alignedByteSize);
    allocated_[i] = memory->virAddr != nullptr;
    if (!check_ret(rc, "input allocation") || !memory->virAddr) {
      release();
      return false;
    }
    std::memset(memory->virAddr, 0,
                static_cast<size_t>(properties[i].alignedByteSize));
  }
  plan_ = actual;
  ready_ = true;
  return true;
}
bool Nv12Input::upload_planes(const InputPlan &plan, const uint8_t *y,
                              size_t y_bytes, const uint8_t *uv,
                              size_t uv_bytes) {
  if (!ready_ || !same_plan(plan, plan_) || !y || !uv)
    return false;
  const size_t count = static_cast<size_t>(plan_.input_h) * plan_.input_w;
  if (y_bytes != count || uv_bytes != count / 2)
    return false;
  if (plan_.protocol == InputProtocol::kPackedNv12) {
    auto *destination =
        static_cast<uint8_t *>(YOLO_SYS_MEM(tensors_[0])->virAddr);
    std::memcpy(destination, y, count);
    std::memcpy(destination + count, uv, count / 2);
  } else {
    for (int i = 0; i < 2; ++i) {
      const int rows = i == 0 ? plan_.input_h : plan_.input_h / 2;
      const int stride = i == 0 ? plan_.y_stride : plan_.uv_stride;
      const uint8_t *source = i == 0 ? y : uv;
      auto *destination =
          static_cast<uint8_t *>(YOLO_SYS_MEM(tensors_[i])->virAddr);
      for (int row = 0; row < rows; ++row)
        std::memcpy(destination + static_cast<size_t>(row) * stride,
                    source + static_cast<size_t>(row) * plan_.input_w,
                    plan_.input_w);
    }
  }
  for (int i = 0; i < input_count_; ++i)
    if (!check_ret(
            YOLO_SYS_FLUSH(YOLO_SYS_MEM(tensors_[i]), HB_SYS_MEM_CACHE_CLEAN),
            "input cache clean"))
      return false;
  return true;
}
bool Nv12Input::upload(const InputPlan &plan, const uint8_t *i420) {
  if (!ready_ || !same_plan(plan, plan_) || !i420)
    return false;
  const size_t count = static_cast<size_t>(plan_.input_h) * plan_.input_w;
  std::vector<uint8_t> uv(count / 2);
  for (size_t i = 0; i < count / 4; ++i) {
    uv[2 * i] = i420[count + i];
    uv[2 * i + 1] = i420[count + count / 4 + i];
  }
  return upload_planes(plan, i420, count, uv.data(), uv.size());
}
void Nv12Input::release() {
  ready_ = false;
  for (int i = 0; i < 2; ++i)
    if (allocated_[i]) {
      check_ret(YOLO_SYS_FREE(YOLO_SYS_MEM(tensors_[i])), "input release");
      allocated_[i] = false;
    }
  input_count_ = 0;
  plan_ = InputPlan{};
}
} // namespace yolo
