#pragma once
#include "hb_sys.h"
using hbPackedDNNHandle_t = void *;
using hbDNNHandle_t = void *;
using hbDNNTaskHandle_t = void *;
constexpr int HB_DNN_TENSOR_MAX_DIMENSIONS = 8;
enum { HB_DNN_TENSOR_TYPE_F32 = 3, NONE = 0 };
struct hbDNNTensorShape {
  int numDimensions = 4;
  int dimensionSize[8]{};
};
struct hbDNNTensorProperties {
  hbDNNTensorShape validShape, alignedShape;
  int tensorLayout = 0, tensorType = HB_DNN_TENSOR_TYPE_F32, quantiType = NONE;
  int alignedByteSize = 0;
};
struct hbDNNTensor {
  hbDNNTensorProperties properties;
  hbSysMem sysMem[4];
};
struct hbDNNInferCtrlParam {
  int priority = 0;
};
#define HB_DNN_INITIALIZE_INFER_CTRL_PARAM(p) (*(p) = hbDNNInferCtrlParam{})
int hbDNNInitializeFromFiles(hbPackedDNNHandle_t *, const char **, int);
int hbDNNRelease(hbPackedDNNHandle_t);
int hbDNNGetModelNameList(const char ***, int *, hbPackedDNNHandle_t);
int hbDNNGetModelHandle(hbDNNHandle_t *, hbPackedDNNHandle_t, const char *);
int hbDNNGetInputCount(int *, hbDNNHandle_t);
int hbDNNGetOutputCount(int *, hbDNNHandle_t);
int hbDNNGetInputName(const char **, hbDNNHandle_t, int);
int hbDNNGetOutputName(const char **, hbDNNHandle_t, int);
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t, int);
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t, int);
const char *hbDNNGetVersion();
int hbDNNInfer(hbDNNTaskHandle_t *, hbDNNTensor **, hbDNNTensor *,
               hbDNNHandle_t, hbDNNInferCtrlParam *);
int hbDNNWaitTaskDone(hbDNNTaskHandle_t, int);
int hbDNNReleaseTask(hbDNNTaskHandle_t);
