// Host-only descriptor/allocator double. Not SDK or ABI compatibility evidence.
#pragma once
#include <cstddef>
#include <cstdint>
using hbDNNHandle_t = void*;
constexpr int HB_DNN_TENSOR_TYPE_F32 = 1, NONE = 0, HB_DNN_LAYOUT_NHWC = 1,
              HB_SYS_MEM_CACHE_INVALIDATE = 1;
struct hbDNNTensorShape {
  int numDimensions;
  int dimensionSize[4];
};
struct hbDNNTensorProperties {
  int tensorType, quantiType, tensorLayout, alignedByteSize;
  hbDNNTensorShape validShape, alignedShape;
  long long stride[4];
};
struct TestMemory {
  void* virAddr;
};
struct hbDNNTensor {
  hbDNNTensorProperties properties;
  TestMemory sysMem;
};
int hbDNNGetOutputCount(int32_t*, hbDNNHandle_t);
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties*, hbDNNHandle_t, int);
int fake_allocate(TestMemory*, int);
int fake_free(TestMemory*);
int fake_flush(TestMemory*, int);
#define YOLO_SYS_MEM(t) (&(t).sysMem)
#define YOLO_SYS_ALLOC_CACHED(m, n) fake_allocate(m, n)
#define YOLO_SYS_FREE(m) fake_free(m)
#define YOLO_SYS_FLUSH(m, f) fake_flush(m, f)
