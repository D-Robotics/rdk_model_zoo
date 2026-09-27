// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "common/classification_binding.h"
#include "common/dnn_resources.h"
#include "common/imagenet_labels.h"
#include <stdexcept>
int allocations=0,frees=0,releases=0,allocation_error=0;
#define expect(value) do { if (!(value)) throw std::runtime_error(#value); } while (0)
template<class F> void rejects(F fn) {
  bool rejected=false;
  try {fn();} catch (const std::invalid_argument&) {rejected=true;}
  expect(rejected);
}
int main() {
  expect(IMAGENET_CLASSES.size()==1000);
  hbDNNTensorProperties p{};
  p.tensorType=HB_DNN_TENSOR_TYPE_F32;p.quantiType=NONE;p.alignedByteSize=16000;
  p.validShape={4,{1,1000,1,1}};p.alignedShape={4,{1,1000,2,2}};
  p.stride[0]=16000;p.stride[1]=16;p.stride[2]=8;p.stride[3]=4;
  expect(yolo::bind_classification(p).class_stride==16);
  auto bad=p;bad.tensorType=99;rejects([&]{yolo::bind_classification(bad);});
  bad=p;bad.quantiType=1;rejects([&]{yolo::bind_classification(bad);});
  bad=p;bad.alignedByteSize=100;rejects([&]{yolo::bind_classification(bad);});
  bad=p;bad.validShape.numDimensions=0;rejects([&]{yolo::bind_classification(bad);});
#if defined(YOLO_DNN_STACK_X5)
  bad=p;bad.alignedShape.numDimensions=3;rejects([&]{yolo::bind_classification(bad);});
  bad=p;bad.alignedShape.dimensionSize[1]=999;rejects([&]{yolo::bind_classification(bad);});
#else
  bad=p;bad.stride[1]=-4;rejects([&]{yolo::bind_classification(bad);});
  bad=p;bad.stride[1]=6;rejects([&]{yolo::bind_classification(bad);});
#endif
  // Both successful and exceptional scopes release each acquired resource once.
  try {
    yolo::PackedModelOwner model;model.handle=reinterpret_cast<void*>(1);
    yolo::OutputTensorOwner output;
    expect(output.allocate(p)==0);
    expect(output.allocate(p)!=0);
    throw std::runtime_error("simulate inference failure");
  } catch (const std::runtime_error&) {}
  expect(allocations==1 && frees==1 && releases==1);
  allocation_error=-4;
  { yolo::OutputTensorOwner output;expect(output.allocate(p)==-4); }
  expect(allocations==2 && frees==1);
}
