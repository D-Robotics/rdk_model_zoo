// Host descriptor/resource test double, NOT a board SDK compatibility header.
#pragma once
#include <cstddef>
using yolo_packed_handle_t=void*;
constexpr int HB_DNN_TENSOR_TYPE_F32=1, NONE=0;
struct hbDNNTensorShape { int numDimensions; int dimensionSize[4]; };
struct hbDNNTensorProperties {
  int tensorType,quantiType,alignedByteSize;
  hbDNNTensorShape validShape,alignedShape;
  long long stride[4];
};
struct TestMemory { void* virAddr; };
struct hbDNNTensor { hbDNNTensorProperties properties; TestMemory sysMem; };
extern int allocations,frees,releases,allocation_error;
inline int hbDNNRelease(void*) { ++releases;return 0; }
inline int fake_allocate(TestMemory*,int) { ++allocations;return allocation_error; }
inline int fake_free(TestMemory*) { ++frees;return 0; }
#define YOLO_SYS_MEM(t) (&(t).sysMem)
#define YOLO_SYS_ALLOC_CACHED(mem,bytes) fake_allocate(mem,bytes)
#define YOLO_SYS_FREE(mem) fake_free(mem)
