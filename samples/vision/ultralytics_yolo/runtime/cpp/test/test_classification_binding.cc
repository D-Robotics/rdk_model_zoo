// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Host unit tests for the classification descriptor binding and the RAII
// resource owners, compiled against the production inc/backend.hpp with the
// fake_dnn_io stack doubles selecting the stack via the genuine header probe.
#include "backend.hpp"
#include "imagenet_labels.hpp"
#include <stdexcept>
int allocations=0,frees=0,releases=0,allocation_error=0;
#define expect(value) do { if (!(value)) throw std::runtime_error(#value); } while (0)
template<class F> void rejects(F fn) {
  bool rejected=false;
  try {fn();} catch (const std::invalid_argument&) {rejected=true;}
  expect(rejected);
}
// SDK hooks behind the fake stack headers; the allocation semantics cover
// success, each failure mode, and success-with-null-address.
int test_allocate(TestMemory* memory,int) {
  ++allocations;
  memory->virAddr=(allocation_error==0 || allocation_error==-5) ? reinterpret_cast<void*>(1) : nullptr;
  return allocation_error==1 ? 0 : allocation_error;
}
int test_free(TestMemory*) { ++frees;return 0; }
int hbDNNRelease(void*) { ++releases;return 0; }
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
  allocation_error=-5; // failure with an acquired allocation
  { yolo::OutputTensorOwner output; expect(output.allocate(p)==-5); }
  expect(allocations==3 && frees==2);
  allocation_error=1; // SDK claims success but returns no address
  { yolo::OutputTensorOwner output; expect(output.allocate(p)!=0); }
  expect(allocations==4 && frees==2);
}
