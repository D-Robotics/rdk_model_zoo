#pragma once
#include "hobot/hb_ucp.h"
using hbDNNPackedHandle_t = void *;
using hbDNNHandle_t = void *;
enum {
  HB_DNN_TENSOR_TYPE_BOOL8,
  HB_DNN_TENSOR_TYPE_S8,
  HB_DNN_TENSOR_TYPE_U8,
  HB_DNN_TENSOR_TYPE_F16,
  HB_DNN_TENSOR_TYPE_S16,
  HB_DNN_TENSOR_TYPE_U16,
  HB_DNN_TENSOR_TYPE_F32,
  HB_DNN_TENSOR_TYPE_S32,
  HB_DNN_TENSOR_TYPE_U32,
  HB_DNN_TENSOR_TYPE_F64,
  HB_DNN_TENSOR_TYPE_S64,
  HB_DNN_TENSOR_TYPE_U64
};
struct hbDNNShape {
  int numDimensions = 1;
  int32_t dimensionSize[8]{4};
};
struct hbDNNTensorProperties {
  hbDNNShape validShape;
  int64_t stride[8]{4};
  int64_t alignedByteSize = 16;
  int tensorType = HB_DNN_TENSOR_TYPE_F32;
};
struct hbDNNTensor {
  hbDNNTensorProperties properties;
  hbUCPSysMem sysMem;
};
int hbDNNInitializeFromFiles(hbDNNPackedHandle_t *, const char **, int);
int hbDNNGetModelHandle(hbDNNHandle_t *, hbDNNPackedHandle_t, const char *);
int hbDNNGetInputCount(int *, hbDNNHandle_t);
int hbDNNGetOutputCount(int *, hbDNNHandle_t);
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t, int);
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t, int);
int hbDNNGetTaskOutputTensorProperties(hbDNNTensorProperties *,
                                       hbUCPTaskHandle_t, int, int);
int hbDNNGetCompileBpuCoreNum(int32_t *, hbDNNHandle_t);
int hbDNNRelease(hbDNNPackedHandle_t);
int hbDNNInferV2(hbUCPTaskHandle_t *, hbDNNTensor *, const hbDNNTensor *,
                 hbDNNHandle_t);
const char *hbDNNGetErrorDesc(int);
