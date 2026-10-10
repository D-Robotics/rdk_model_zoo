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

// The genuine shared DNN resource backend of the ultralytics_yolo C++ runtime.
// It owns the board SDK surface used by every task model, and is also consumed
// by the native ASR/Paraformer runtimes:
//   * the X5/S stack probe and the YOLO_SYS_* platform shims,
//   * synchronous inference (infer_sync / infer_tensors_sync),
//   * the NV12 input protocol probe and the RAII Nv12Input uploader,
//   * the model and output-tensor RAII owners (PackedModelOwner /
//     OutputTensorOwner),
//   * the stride-validated FLOAT32 output binding and reader (bind_float_nhwc
//     / TaskOutputs) and the classification binding (bind_classification).
// The pure decode/plan vocabulary it builds on lives in yolo.hpp; this header
// adds exactly the SDK-touching layer, with public signatures held stable for
// out-of-sample consumers.

#ifndef YOLO_RUNTIME_CPP_INC_BACKEND_HPP_
#define YOLO_RUNTIME_CPP_INC_BACKEND_HPP_

#include "yolo.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>

// Stack selection relies on the packaged header layout: X5 images expose the
// hbSys layer as "dnn/hb_sys.h", while S-series images expose the UCP layer
// as "hb_ucp_sys.h" and ship no dnn/hb_sys.h. If a future S-series hobot-dnn
// package ever ships a compat "dnn/hb_sys.h", this probe would silently pick
// the X5 branch on S and fail to link; guard that combination explicitly.
#if defined(__has_include)
#if __has_include("dnn/hb_sys.h") && __has_include("hb_ucp_sys.h")
#error "both hbSys and UCP layers are visible; stack selection is ambiguous"
#endif
#if __has_include("dnn/hb_sys.h")
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#define YOLO_DNN_STACK_X5 1
#elif __has_include("hb_ucp_sys.h")
#include "dnn/hb_dnn.h"
#include "hb_ucp.h"
#include "hb_ucp_sys.h"
#define YOLO_DNN_STACK_UCP 1
#endif
#endif

#ifndef YOLO_DNN_STACK_X5
#ifndef YOLO_DNN_STACK_UCP
#error "unsupported DNN stack: neither dnn/hb_sys.h nor hb_ucp_sys.h was found"
#endif
#endif

// The packed-handle typedef swaps word order between stacks.
#if defined(YOLO_DNN_STACK_X5)
typedef hbPackedDNNHandle_t yolo_packed_handle_t;
#else
typedef hbDNNPackedHandle_t yolo_packed_handle_t;
#endif

namespace yolo {

// ---------------------------------------------------------------------------
// Platform shims. X5 keeps sysMem as an array managed with hbSys* calls and
// runs hbDNNInfer with an infer-control parameter; S-series keeps a single
// hbUCPSysMem managed with hbUCP* calls and offers the simpler hbDNNInferV2.
// ---------------------------------------------------------------------------

#if defined(YOLO_DNN_STACK_X5)

#define YOLO_SYS_MEM(tensor) (&(tensor).sysMem[0])
#define YOLO_SYS_ALLOC_CACHED(mem, size) hbSysAllocCachedMem((mem), (size))
#define YOLO_SYS_FLUSH(mem, flag) hbSysFlushMem((mem), (flag))
#define YOLO_SYS_FREE(mem) hbSysFreeMem((mem))

#else // YOLO_DNN_STACK_UCP

#define YOLO_SYS_MEM(tensor) (&(tensor).sysMem)
#define YOLO_SYS_ALLOC_CACHED(mem, size) hbUCPMallocCached((mem), (size), 0)
#define YOLO_SYS_FLUSH(mem, flag) hbUCPMemFlush((mem), (flag))
#define YOLO_SYS_FREE(mem) hbUCPFree((mem))

#endif

// Synchronous inference across both stacks: submits the input tensor array,
// waits for completion and releases the task. Returns 0 on success.
int infer_sync(hbDNNTensor *outputs, hbDNNTensor *inputs, int input_count,
               hbDNNHandle_t model);

// Raw multi-input model transport. The caller validates the full tensor array
// against SDK metadata and owns all buffers until this synchronous call returns.
// infer_sync above retains the one/two-input image protocol restriction.
int infer_tensors_sync(hbDNNTensor *outputs, hbDNNTensor *inputs, int input_count,
                       hbDNNHandle_t model);

// Physical output strides in floats for an NHWC FLOAT32 tensor. X5 exposes
// alignedShape extents; S-series exposes per-dimension byte strides.
inline int tensor_row_step_floats(const hbDNNTensorProperties &properties) {
#if defined(YOLO_DNN_STACK_X5)
  const hbDNNTensorShape &aligned = properties.alignedShape;
  return aligned.numDimensions == 4
             ? aligned.dimensionSize[2] * aligned.dimensionSize[3]
             : 0;
#else
  return properties.stride[1] > 0
             ? static_cast<int>(properties.stride[1] / sizeof(float))
             : 0;
#endif
}

inline int tensor_cell_step_floats(const hbDNNTensorProperties &properties) {
#if defined(YOLO_DNN_STACK_X5)
  const hbDNNTensorShape &aligned = properties.alignedShape;
  return aligned.numDimensions == 4 ? aligned.dimensionSize[3] : 0;
#else
  return properties.stride[2] > 0
             ? static_cast<int>(properties.stride[2] / sizeof(float))
             : 0;
#endif
}

// Allocation size in bytes for a tensor whose alignedByteSize is not reported
// (dynamic outputs): derived from the valid shape and the element size.
// Returns 0 when the layout cannot be sized, which callers treat as failure.
inline int64_t output_alloc_bytes(const hbDNNTensorProperties &properties) {
  if (properties.alignedByteSize > 0)
    return properties.alignedByteSize;
  int64_t element = 0;
  if (properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    element = 4;
  } else if (properties.tensorType == HB_DNN_TENSOR_TYPE_S16 ||
             properties.tensorType == HB_DNN_TENSOR_TYPE_F16) {
    element = 2;
  } else if (properties.tensorType == HB_DNN_TENSOR_TYPE_S8) {
    element = 1;
#if !defined(YOLO_DNN_STACK_X5)
  } else if (properties.tensorType == HB_DNN_TENSOR_TYPE_U8) {
    element = 1;
#endif
  }
  if (element == 0 || properties.validShape.numDimensions <= 0)
    return 0;
  int64_t count = 1;
  for (int i = 0; i < properties.validShape.numDimensions; ++i) {
    if (properties.validShape.dimensionSize[i] <= 0)
      return 0;
    count *= properties.validShape.dimensionSize[i];
  }
  return count * element;
}

enum class InputProtocol { kUnknown, kPackedNv12, kSplitNv12 };

// Everything the preprocessing side needs to upload one frame.
struct InputPlan {
  InputProtocol protocol = InputProtocol::kUnknown;
  int input_w = 0;
  int input_h = 0;
  // Byte strides for the split protocol (properties.stride[1]); unused for
  // the packed protocol.
  int y_stride = 0;
  int uv_stride = 0;
};

// Inspects the model inputs and returns the detected protocol. `error`
// receives a human-readable reason when kUnknown is returned.
InputPlan probe_input_protocol(hbDNNHandle_t model, std::string *error);

// RAII owner of the input tensors matching an InputPlan. Owns the sysMem
// allocations and exposes a protocol-agnostic upload of I420 planes (as
// produced by cv::cvtColor(..., COLOR_BGR2YUV_I420)).
class Nv12Input {
public:
  Nv12Input() = default;
  ~Nv12Input() { release(); }

  Nv12Input(const Nv12Input &) = delete;
  Nv12Input &operator=(const Nv12Input &) = delete;

  // Queries the per-input tensor properties from `model` and allocates the
  // backing sysMem buffers.
  bool allocate(hbDNNHandle_t model, const InputPlan &plan);

  // Uploads one frame. `i420` points at an h*w*3/2 byte I420 buffer for
  // either protocol (only the planes are consumed for the split protocol).
  bool upload(const InputPlan &plan, const uint8_t *i420);

  // Upload already interleaved compact planes, validating exact source lengths
  // and the allocation-bound plan before touching device memory.
  bool upload_planes(const InputPlan &plan, const uint8_t *y, size_t y_bytes,
                     const uint8_t *uv, size_t uv_bytes);

  // Tensor array to hand to infer_sync (length = input_count()).
  hbDNNTensor *tensors() { return tensors_; }
  int input_count() const { return input_count_; }

private:
  void release();

  hbDNNTensor tensors_[2];
  bool allocated_[2] = {false, false};
  int input_count_ = 0;
  InputPlan plan_;
  bool ready_ = false;
};

// Declare the model owner before tensor owners so tensors are released first.
class PackedModelOwner {
 public:
  PackedModelOwner() = default;
  ~PackedModelOwner() { if (handle) hbDNNRelease(handle); }
  PackedModelOwner(const PackedModelOwner&)=delete;
  PackedModelOwner& operator=(const PackedModelOwner&)=delete;
  yolo_packed_handle_t handle=nullptr;
};
class OutputTensorOwner {
 public:
  OutputTensorOwner() { std::memset(&tensor,0,sizeof(tensor)); }
  ~OutputTensorOwner() { if (allocated_) YOLO_SYS_FREE(YOLO_SYS_MEM(tensor)); }
  OutputTensorOwner(const OutputTensorOwner&)=delete;
  OutputTensorOwner& operator=(const OutputTensorOwner&)=delete;
  int allocate(const hbDNNTensorProperties& properties) {
    if (allocated_ || properties.alignedByteSize<=0) return -1;
    tensor.properties=properties;
    const int rc=YOLO_SYS_ALLOC_CACHED(YOLO_SYS_MEM(tensor),properties.alignedByteSize);
    // Some failing SDK calls still return an acquired allocation. Own it so
    // constructor unwinding releases it; a successful null allocation is invalid.
    allocated_=YOLO_SYS_MEM(tensor)->virAddr != nullptr;
    return rc != 0 ? rc : (allocated_ ? 0 : -1);
  }
  hbDNNTensor tensor;
 private:
  bool allocated_=false;
};

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
    bind(model, segment ? 10 : 9, [&](const std::vector<OutputShape>& shapes) {
      heads = bind_task_heads(shapes, h, w, segment);
    });
  }
  // Task-specific semantic validation precedes every allocation; physical
  // FLOAT32/stride/capacity ownership stays shared across model families.
  void bind(hbDNNHandle_t model, int expected_count,
            const std::function<void(const std::vector<OutputShape>&)>& validate) {
    if (!validate || expected_count <= 0 || expected_count > 64)
      throw std::invalid_argument("Provide bounded output count and semantic validator.");
    if (!tensors_.empty())
      throw std::invalid_argument("Task outputs already bound.");
    int32_t count = 0;
    check(hbDNNGetOutputCount(&count, model), "Cannot query output count.");
    if (count != expected_count)
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
    validate(shapes);
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
  // Internal-only zero-copy alternative to read(): invalidates every output
  // cache and returns strided views over the padded physical buffers. Values
  // are not prescanned, so a decoder must reject nonfinite values it
  // consumes. The views borrow the SDK buffers and stay valid until the next
  // inference; they never leave a model call, and RawResult stays an owned
  // copy. read() is unchanged.
  std::vector<TensorView> views() {
    if (!bound_) throw std::invalid_argument("Task outputs are not bound.");
    std::vector<TensorView> result;
    for (size_t i = 0; i < tensors_.size(); ++i) {
      if (!allocated_[i])
        throw std::invalid_argument("Task output not allocated.");
      check(YOLO_SYS_FLUSH(YOLO_SYS_MEM(tensors_[i]),
                           HB_SYS_MEM_CACHE_INVALIDATE),
            "Cannot invalidate task output cache.");
      TensorView view;
      view.data = static_cast<const float*>(YOLO_SYS_MEM(tensors_[i])->virAddr);
      view.h = plans_[i].shape.h;
      view.w = plans_[i].shape.w;
      view.channels = plans_[i].shape.c;
      view.row_step =
          static_cast<int>(plans_[i].row_bytes / sizeof(float));
      view.cell_step =
          static_cast<int>(plans_[i].cell_bytes / sizeof(float));
      result.push_back(view);
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

inline ClassificationPlan bind_classification(const hbDNNTensorProperties& p) {
  const int rank=p.validShape.numDimensions;
  if (rank<1 || rank>4 || p.alignedByteSize<=0)
    throw std::invalid_argument("Classification requires static shape and physical allocation size.");
  std::vector<int> shape(p.validShape.dimensionSize,p.validShape.dimensionSize+rank);
  std::vector<size_t> strides(rank);
#if defined(YOLO_DNN_STACK_X5)
  if (p.alignedShape.numDimensions!=rank)
    throw std::invalid_argument("Classification aligned shape rank mismatch.");
  size_t step=sizeof(float);
  for (int i=rank-1;i>=0;--i) {
    int extent=p.alignedShape.dimensionSize[i];
    if (extent<=0 || extent<shape[i] || step>std::numeric_limits<size_t>::max()/static_cast<size_t>(extent))
      throw std::invalid_argument("Invalid classification aligned shape.");
    strides[i]=step;step*=extent;
  }
  if (step>static_cast<size_t>(p.alignedByteSize))
    throw std::invalid_argument("Classification aligned shape exceeds allocation.");
#else
  for (int i=0;i<rank;++i) {
    if (p.stride[i]<=0) throw std::invalid_argument("Classification requires positive physical strides.");
    strides[i]=static_cast<size_t>(p.stride[i]);
  }
#endif
  return classification_plan(shape,strides,static_cast<size_t>(p.alignedByteSize),
      p.tensorType==HB_DNN_TENSOR_TYPE_F32,p.quantiType==NONE);
}

} // namespace yolo

#endif // YOLO_RUNTIME_CPP_INC_BACKEND_HPP_
