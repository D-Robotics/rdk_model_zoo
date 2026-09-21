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

// Board-side input plumbing shared by the C++ task samples. This layer probes
// the model's input protocol at load time instead of hard-coding a platform:
//   * packed NV12 (X5 .bin): one HB_DNN_IMG_TYPE_NV12 NCHW tensor.
//   * split NV12 (S100/S100P/S600 .hbm): two UINT8 tensors, images_y
//     [1,H,W,1] followed by images_uv [1,H/2,W/2,2], each row-aligned.

#ifndef RUNTIME_CPP_COMMON_DNN_IO_H_
#define RUNTIME_CPP_COMMON_DNN_IO_H_

#include <cstdint>
#include <string>

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

namespace yolo {

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
  ~Nv12Input() { release(); }

  Nv12Input(const Nv12Input&) = delete;
  Nv12Input& operator=(const Nv12Input&) = delete;

  // Queries the per-input tensor properties from `model` and allocates the
  // backing sysMem buffers.
  bool allocate(hbDNNHandle_t model, const InputPlan& plan);

  // Uploads one frame. `i420` points at an h*w*3/2 byte I420 buffer for
  // either protocol (only the planes are consumed for the split protocol).
  bool upload(const InputPlan& plan, const uint8_t* i420);

  // Tensor array to hand to hbDNNInfer (length = input_count()).
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
