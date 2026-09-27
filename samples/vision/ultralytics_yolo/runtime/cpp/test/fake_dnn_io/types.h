// Host-only SDK API double, never ABI or device execution evidence.
#pragma once
#include <cstdint>
struct TestMemory {
  void *virAddr;
  int memSize;
};
using hbDNNHandle_t = void *;
using hbPackedDNNHandle_t = void *;
using hbDNNPackedHandle_t = void *;
using hbDNNTaskHandle_t = void *;
using hbUCPTaskHandle_t = void *;
constexpr int HB_DNN_TENSOR_TYPE_F32 = 1, HB_DNN_TENSOR_TYPE_F16 = 2,
              HB_DNN_TENSOR_TYPE_S16 = 3, HB_DNN_TENSOR_TYPE_S8 = 4,
              HB_DNN_TENSOR_TYPE_U8 = 5, HB_DNN_IMG_TYPE_NV12 = 6,
              HB_DNN_LAYOUT_NCHW = 0, HB_DNN_LAYOUT_NHWC = 1, NONE = 0,
              HB_SYS_MEM_CACHE_CLEAN = 1, HB_SYS_MEM_CACHE_INVALIDATE = 2;
struct hbDNNTensorShape {
  int numDimensions;
  int dimensionSize[4];
};
struct hbDNNTensorProperties {
  int tensorType, quantiType, tensorLayout, alignedByteSize;
  hbDNNTensorShape validShape, alignedShape;
  int stride[4];
};
struct hbDNNTensor {
  hbDNNTensorProperties properties;
#ifdef TEST_DNN_X5
  TestMemory sysMem[4];
#else
  TestMemory sysMem;
#endif
};
struct hbDNNInferCtrlParam {
  int placeholder;
};
#define HB_DNN_INITIALIZE_INFER_CTRL_PARAM(p) (*(p) = {})
struct hbUCPSchedParam {
  unsigned long long backend;
};
constexpr unsigned long long HB_UCP_BPU_CORE_ANY = 0xffff;
#define HB_UCP_INITIALIZE_SCHED_PARAM(p) (*(p) = {})
int hbDNNGetInputCount(int32_t *, hbDNNHandle_t);
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t, int);
int test_allocate(TestMemory *, int);
int test_free(TestMemory *);
int test_flush(TestMemory *, int);
int test_create(void **);
int test_wait(void *);
int test_release(void *);
int test_submit(void *, hbUCPSchedParam *);
inline int hbSysAllocCachedMem(TestMemory *m, int n) {
  return test_allocate(m, n);
}
inline int hbSysFreeMem(TestMemory *m) { return test_free(m); }
inline int hbSysFlushMem(TestMemory *m, int f) { return test_flush(m, f); }
inline int hbUCPMallocCached(TestMemory *m, int n, int) {
  return test_allocate(m, n);
}
inline int hbUCPFree(TestMemory *m) { return test_free(m); }
inline int hbUCPMemFlush(TestMemory *m, int f) { return test_flush(m, f); }
inline int hbDNNInfer(void **t, hbDNNTensor **, hbDNNTensor *, void *,
                      hbDNNInferCtrlParam *) {
  return test_create(t);
}
inline int hbDNNInferV2(void **t, hbDNNTensor *, hbDNNTensor *, void *) {
  return test_create(t);
}
inline int hbDNNWaitTaskDone(void *t, int) { return test_wait(t); }
inline int hbUCPWaitTaskDone(void *t, int) { return test_wait(t); }
inline int hbDNNReleaseTask(void *t) { return test_release(t); }
inline int hbUCPReleaseTask(void *t) { return test_release(t); }
inline int hbUCPSubmitTask(void *t, hbUCPSchedParam *s) {
  return test_submit(t, s);
}
