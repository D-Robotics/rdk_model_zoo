// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#ifndef YOLO_COMMON_DNN_RESOURCES_H_
#define YOLO_COMMON_DNN_RESOURCES_H_
#include "common/dnn_io.h"
#include <cstring>
namespace yolo {
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
}  // namespace yolo
#endif
