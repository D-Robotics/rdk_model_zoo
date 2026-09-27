// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#ifndef YOLO_COMMON_CLASSIFICATION_H_
#define YOLO_COMMON_CLASSIFICATION_H_
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace yolo {
struct ClassificationPlan {
  size_t class_stride;
  size_t required_bytes;
};
struct ClassificationScore {
  int id;
  float probability;
};

// One 1000-class vector, optionally surrounded by singleton dimensions.
// Physical padding is permitted; the class stride must be supplied by metadata.
inline ClassificationPlan classification_plan(const std::vector<int>& shape,
    const std::vector<size_t>& strides, size_t bytes, bool float32, bool unquantized) {
  if (!float32 || !unquantized || shape.empty() || shape.size()>4 || strides.size()!=shape.size())
    throw std::invalid_argument("Classification requires one unquantized FLOAT32 vector.");
  int axis=-1;
  for (size_t i=0;i<shape.size();++i) {
    if (shape[i]!=1) {
      if (shape[i]!=1000 || axis!=-1 || (shape.size()>1 && i==0))
        throw std::invalid_argument("Classification requires a single 1000-class axis and batch one.");
      axis=static_cast<int>(i);
    }
  }
  if (axis<0 || strides[axis]<sizeof(float) || strides[axis]%sizeof(float)!=0 ||
      bytes<sizeof(float) || strides[axis]>(bytes-sizeof(float))/999)
    throw std::invalid_argument("Classification class stride exceeds the output buffer or is invalid.");
  return {strides[axis],999*strides[axis]+sizeof(float)};
}

inline std::vector<ClassificationScore> classification_topk(const void* data,
    size_t bytes, const ClassificationPlan& plan, int topk) {
  if (!data || topk<1 || topk>1000 || plan.class_stride<sizeof(float) ||
      plan.class_stride%sizeof(float)!=0 || bytes<sizeof(float) ||
      plan.class_stride>(bytes-sizeof(float))/999 ||
      plan.required_bytes!=999*plan.class_stride+sizeof(float))
    throw std::invalid_argument("Invalid classification buffer, plan or Top-K.");
  std::vector<double> logits(1000);
  for (int i=0;i<1000;++i) {
    float value;
    std::memcpy(&value,static_cast<const unsigned char*>(data)+i*plan.class_stride,sizeof(value));
    if (!std::isfinite(value)) throw std::invalid_argument("Classification logits must be finite.");
    logits[i]=value;
  }
  const double maximum=*std::max_element(logits.begin(),logits.end());
  double total=0;
  for (double& value:logits) { value=std::exp(value-maximum);total+=value; }
  std::vector<ClassificationScore> result;
  for (int i=0;i<1000;++i) result.push_back({i,static_cast<float>(logits[i]/total)});
  std::partial_sort(result.begin(),result.begin()+topk,result.end(),
    [](const ClassificationScore& a,const ClassificationScore& b) {
      return a.probability!=b.probability ? a.probability>b.probability : a.id<b.id;
    });
  result.resize(topk);
  return result;
}
}  // namespace yolo
#endif
