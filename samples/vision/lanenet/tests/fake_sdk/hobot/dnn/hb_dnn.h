#pragma once
#include "hobot/hb_ucp.h"
using hbDNNPackedHandle_t = void *;
using hbDNNHandle_t = void *;
enum {
  HB_DNN_TENSOR_TYPE_F32 = 1,
  HB_DNN_TENSOR_TYPE_S64,
  HB_DNN_TENSOR_TYPE_S32,
  HB_DNN_TENSOR_TYPE_S16,
  HB_DNN_TENSOR_TYPE_S8,
  HB_DNN_TENSOR_TYPE_U8
};
struct hbDNNShape {
  int numDimensions = 0;
  int dimensionSize[8]{};
};
struct hbDNNTensorProperties {
  hbDNNShape validShape;
  std::int64_t stride[8]{};
  std::int64_t alignedByteSize = 0;
  int tensorType = 0;
};
struct hbDNNTensor {
  hbDNNTensorProperties properties;
  hbUCPSysMem sysMem;
};
int hbDNNInitializeFromFiles(hbDNNPackedHandle_t *, const char **, int);
int hbDNNGetModelNameList(const char ***, int *, hbDNNPackedHandle_t);
int hbDNNGetModelHandle(hbDNNHandle_t *, hbDNNPackedHandle_t, const char *);
int hbDNNGetInputCount(int32_t *, hbDNNHandle_t);
int hbDNNGetOutputCount(int32_t *, hbDNNHandle_t);
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t, int);
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t, int);
int hbDNNRelease(hbDNNPackedHandle_t);
int hbDNNInferV2(hbUCPTaskHandle_t *, hbDNNTensor *, hbDNNTensor *,
                 hbDNNHandle_t);
