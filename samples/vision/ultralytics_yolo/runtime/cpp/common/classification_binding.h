// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#ifndef YOLO_COMMON_CLASSIFICATION_BINDING_H_
#define YOLO_COMMON_CLASSIFICATION_BINDING_H_
#include "common/classification.h"
#include "common/dnn_io.h"
#include <limits>
namespace yolo {
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
}  // namespace yolo
#endif
