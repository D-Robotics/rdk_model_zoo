#pragma once
#include "hb_sys.h"
using hbPackedDNNHandle_t=void*;
using hbDNNHandle_t=void*;
using hbDNNTaskHandle_t=void*;
constexpr int HB_DNN_TENSOR_TYPE_F32=1,HB_DNN_IMG_TYPE_NV12=2,NONE=0;
struct hbDNNTensorShape { int numDimensions=0; int dimensionSize[4]{}; };
struct hbDNNTensorProperties { int tensorType=0,quantiType=0; hbDNNTensorShape validShape{},alignedShape{}; int alignedByteSize=0; int stride[4]{}; };
struct hbDNNTensor { hbDNNTensorProperties properties{}; hbSysMem sysMem[4]{}; };
struct hbDNNInferCtrlParam { int placeholder=0; };
#define HB_DNN_INITIALIZE_INFER_CTRL_PARAM(p) (*(p)={})
int hbDNNInitializeFromFiles(hbPackedDNNHandle_t*,const char**,int);
int hbDNNGetModelNameList(const char***,int*,hbPackedDNNHandle_t);
int hbDNNGetModelHandle(hbDNNHandle_t*,hbPackedDNNHandle_t,const char*);
int hbDNNGetInputCount(int*,hbDNNHandle_t);
int hbDNNGetOutputCount(int*,hbDNNHandle_t);
int hbDNNGetInputTensorProperties(hbDNNTensorProperties*,hbDNNHandle_t,int);
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties*,hbDNNHandle_t,int);
int hbDNNInfer(hbDNNTaskHandle_t*,hbDNNTensor**,hbDNNTensor*,hbDNNHandle_t,hbDNNInferCtrlParam*);
int hbDNNWaitTaskDone(hbDNNTaskHandle_t,int);
int hbDNNReleaseTask(hbDNNTaskHandle_t);
int hbDNNRelease(hbPackedDNNHandle_t);
