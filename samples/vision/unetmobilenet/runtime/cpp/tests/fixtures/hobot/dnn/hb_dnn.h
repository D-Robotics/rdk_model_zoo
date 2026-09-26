// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "hobot/hb_ucp.h"
using hbDNNPackedHandle_t=void*;
using hbDNNHandle_t=void*;
enum { HB_DNN_TENSOR_TYPE_U8=1, HB_DNN_TENSOR_TYPE_S32=2, HB_DNN_TENSOR_TYPE_F32=3 };
enum { NONE=0, SCALE=1 };
struct hbDNNTensorShape { int numDimensions=4; int dimensionSize[4]{}; };
struct hbDNNScale { int scaleLen=0; float* scaleData=nullptr; int zeroPointLen=0; int* zeroPointData=nullptr; };
struct hbDNNTensorProperties {
    hbDNNTensorShape validShape;
    int tensorType=0;
    int quantiType=NONE;
    int quantizeAxis=3;
    std::int64_t stride[4]{};
    int alignedByteSize=0;
    hbDNNScale scale;
};
struct hbDNNTensor { hbDNNTensorProperties properties; hbUCPSysMem sysMem; };
int hbDNNInitializeFromFiles(hbDNNPackedHandle_t*,const char**,int);
int hbDNNGetModelNameList(const char***,int*,hbDNNPackedHandle_t);
int hbDNNGetModelHandle(hbDNNHandle_t*,hbDNNPackedHandle_t,const char*);
int hbDNNGetInputCount(int32_t*,hbDNNHandle_t);
int hbDNNGetOutputCount(int32_t*,hbDNNHandle_t);
int hbDNNGetInputTensorProperties(hbDNNTensorProperties*,hbDNNHandle_t,int);
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties*,hbDNNHandle_t,int);
int hbDNNInferV2(hbUCPTaskHandle_t*,hbDNNTensor*,hbDNNTensor*,hbDNNHandle_t);
int hbDNNRelease(hbDNNPackedHandle_t);
