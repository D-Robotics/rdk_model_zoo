// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#ifndef YOLO_COMMON_TASK_OUTPUT_BINDING_H_
#define YOLO_COMMON_TASK_OUTPUT_BINDING_H_
#include <limits>

#include "common/dnn_io.h"
#include "common/task_outputs.h"
namespace yolo {
inline FloatOutputPlan bind_float_nhwc(const hbDNNTensorProperties& p) {
  if (p.validShape.numDimensions != 4 || p.alignedByteSize <= 0)
    throw std::invalid_argument(
        "Task output requires static rank-four geometry and allocation.");
  std::vector<int> shape(p.validShape.dimensionSize,
                         p.validShape.dimensionSize + 4);
  std::vector<size_t> strides(4);
#if defined(YOLO_DNN_STACK_X5)
  if (p.tensorLayout != HB_DNN_LAYOUT_NHWC || p.alignedShape.numDimensions != 4)
    throw std::invalid_argument("Task output requires NHWC aligned geometry.");
  size_t step = sizeof(float);
  for (int i = 3; i >= 0; --i) {
    int extent = p.alignedShape.dimensionSize[i];
    if (extent <= 0 || extent < shape[i] ||
        step > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent))
      throw std::invalid_argument("Invalid task output aligned shape.");
    strides[i] = step;
    step *= extent;
  }
  if (step > static_cast<size_t>(p.alignedByteSize))
    throw std::invalid_argument("Aligned shape exceeds allocation.");
#else
  for (int i = 0; i < 4; ++i) {
    if (p.stride[i] <= 0)
      throw std::invalid_argument(
          "Task output needs positive physical strides.");
    strides[i] = static_cast<size_t>(p.stride[i]);
  }
#endif
  return nhwc_float_plan(shape, strides, static_cast<size_t>(p.alignedByteSize),
                         p.tensorType == HB_DNN_TENSOR_TYPE_F32,
                         p.quantiType == NONE);
}
// Owns every acquired allocation. Bind before allocate; results are copied
// after successful inference/cache invalidation so decoders see compact finite
// NHWC.
class TaskOutputs {
 public:
  TaskOutputs() = default;
  TaskOutputs(const TaskOutputs&) = delete;
  TaskOutputs& operator=(const TaskOutputs&) = delete;
  ~TaskOutputs() {
    for (size_t i = 0; i < tensors_.size(); ++i)
      if (allocated_[i]) YOLO_SYS_FREE(YOLO_SYS_MEM(tensors_[i]));
  }
  void bind(hbDNNHandle_t model, int h, int w, bool segment) {
    if (!tensors_.empty())
      throw std::invalid_argument("Task outputs already bound.");
    int32_t count = 0;
    check(hbDNNGetOutputCount(&count, model), "Cannot query output count.");
    if (count != (segment ? 10 : 9))
      throw std::invalid_argument("Unexpected task output count.");
    allocated_.assign(count, false);
    tensors_.resize(count);
    std::vector<OutputShape> shapes;
    for (int i = 0; i < count; ++i) {
      std::memset(&tensors_[i], 0, sizeof(hbDNNTensor));
      check(hbDNNGetOutputTensorProperties(&tensors_[i].properties, model, i),
            "Cannot query output descriptor.");
      plans_.push_back(bind_float_nhwc(tensors_[i].properties));
      shapes.push_back(plans_.back().shape);
    }
    heads = bind_task_heads(shapes, h, w, segment);
    bound_ = true;
  }
  void allocate() {
    if (!bound_ || plans_.size() != tensors_.size() || plans_.empty())
      throw std::invalid_argument("Bind task outputs before allocation.");
    for (size_t i = 0; i < tensors_.size(); ++i) {
      if (allocated_[i])
        throw std::invalid_argument("Task output already allocated.");
      check(YOLO_SYS_ALLOC_CACHED(YOLO_SYS_MEM(tensors_[i]),
                                  tensors_[i].properties.alignedByteSize),
            "Cannot allocate task output.");
      allocated_[i] = true;
    }
  }
  hbDNNTensor* tensors() { return tensors_.data(); }
  std::vector<std::vector<float>> read() {
    if (!bound_) throw std::invalid_argument("Task outputs are not bound.");
    std::vector<std::vector<float>> result;
    for (size_t i = 0; i < tensors_.size(); ++i) {
      if (!allocated_[i])
        throw std::invalid_argument("Task output not allocated.");
      check(YOLO_SYS_FLUSH(YOLO_SYS_MEM(tensors_[i]),
                           HB_SYS_MEM_CACHE_INVALIDATE),
            "Cannot invalidate task output cache.");
      result.push_back(copy_float_output(
          YOLO_SYS_MEM(tensors_[i])->virAddr,
          static_cast<size_t>(tensors_[i].properties.alignedByteSize),
          plans_[i]));
    }
    return result;
  }
  TaskHeadPlan heads;

 private:
  static void check(int rc, const char* message) {
    if (rc) throw std::runtime_error(message);
  }
  bool bound_ = false;
  std::vector<hbDNNTensor> tensors_;
  std::vector<bool> allocated_;
  std::vector<FloatOutputPlan> plans_;
};
}  // namespace yolo
#endif
