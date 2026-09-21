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

// Board-side input plumbing shared by the C++ task samples, portable across
// the X5 (hb_mapper / hbSys* / hbDNNInfer) and S-series (hb_compile /
// hbUCP* / hbDNNInferV2) DNN stacks. Two input protocols are supported and
// probed from the model at load time:
//   * packed NV12 (X5 .bin): one HB_DNN_IMG_TYPE_NV12 NCHW tensor.
//   * split NV12 (S100/S100P/S600 .hbm): two UINT8 tensors, images_y
//     [1,H,W,1] followed by images_uv [1,H/2,W/2,2], each row-aligned.

#ifndef RUNTIME_CPP_COMMON_DNN_IO_H_
#define RUNTIME_CPP_COMMON_DNN_IO_H_

#include <cstdint>
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

#else  // YOLO_DNN_STACK_UCP

#define YOLO_SYS_MEM(tensor) (&(tensor).sysMem)
#define YOLO_SYS_ALLOC_CACHED(mem, size) hbUCPMallocCached((mem), (size), 0)
#define YOLO_SYS_FLUSH(mem, flag) hbUCPMemFlush((mem), (flag))
#define YOLO_SYS_FREE(mem) hbUCPFree((mem))

#endif


// Synchronous inference across both stacks: submits the input tensor array,
// waits for completion and releases the task. Returns 0 on success.
int infer_sync(hbDNNTensor* outputs, hbDNNTensor* inputs, int input_count,
               hbDNNHandle_t model);

// Physical output strides in floats for an NHWC FLOAT32 tensor. X5 exposes
// alignedShape extents; S-series exposes per-dimension byte strides.
inline int tensor_row_step_floats(const hbDNNTensorProperties& properties) {
#if defined(YOLO_DNN_STACK_X5)
  const hbDNNTensorShape& aligned = properties.alignedShape;
  return aligned.numDimensions == 4 ? aligned.dimensionSize[2] * aligned.dimensionSize[3] : 0;
#else
  return properties.stride[1] > 0 ? static_cast<int>(properties.stride[1] / sizeof(float)) : 0;
#endif
}

inline int tensor_cell_step_floats(const hbDNNTensorProperties& properties) {
#if defined(YOLO_DNN_STACK_X5)
  const hbDNNTensorShape& aligned = properties.alignedShape;
  return aligned.numDimensions == 4 ? aligned.dimensionSize[3] : 0;
#else
  return properties.stride[2] > 0 ? static_cast<int>(properties.stride[2] / sizeof(float)) : 0;
#endif
}

// Allocation size in bytes for a tensor whose alignedByteSize is not reported
// (dynamic outputs): derived from the valid shape and the element size.
// Returns 0 when the layout cannot be sized, which callers treat as failure.
inline int64_t output_alloc_bytes(const hbDNNTensorProperties& properties) {
  if (properties.alignedByteSize > 0) return properties.alignedByteSize;
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
  if (element == 0 || properties.validShape.numDimensions <= 0) return 0;
  int64_t count = 1;
  for (int i = 0; i < properties.validShape.numDimensions; ++i) {
    if (properties.validShape.dimensionSize[i] <= 0) return 0;
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
InputPlan probe_input_protocol(hbDNNHandle_t model, std::string* error);

// RAII owner of the input tensors matching an InputPlan. Owns the sysMem
// allocations and exposes a protocol-agnostic upload of I420 planes (as
// produced by cv::cvtColor(..., COLOR_BGR2YUV_I420)).
class Nv12Input {
 public:
  Nv12Input() = default;
  ~Nv12Input() { release(); }

  Nv12Input(const Nv12Input&) = delete;
  Nv12Input& operator=(const Nv12Input&) = delete;

  // Queries the per-input tensor properties from `model` and allocates the
  // backing sysMem buffers.
  bool allocate(hbDNNHandle_t model, const InputPlan& plan);

  // Uploads one frame. `i420` points at an h*w*3/2 byte I420 buffer for
  // either protocol (only the planes are consumed for the split protocol).
  bool upload(const InputPlan& plan, const uint8_t* i420);

  // Tensor array to hand to infer_sync (length = input_count()).
  hbDNNTensor* tensors() { return tensors_; }
  int input_count() const { return input_count_; }

 private:
  void release();

  hbDNNTensor tensors_[2];
  bool allocated_[2] = {false, false};
  int input_count_ = 0;
};

}  // namespace yolo

#endif  // RUNTIME_CPP_COMMON_DNN_IO_H_
