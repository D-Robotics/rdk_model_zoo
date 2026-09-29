// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// Host failure injection against the real ModelRunner implementation.
// These fake interfaces do NOT verify real SDK ABI, linkage or execution.
#include "model_runner.hpp"
#include "hobot/dnn/hb_dnn.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
using namespace unetmobilenet;
namespace {
std::string failure;
int buffers=0,tasks=0,models=0,alloc_calls=0;
int status(const std::string& name){return failure==name?17:0;}
}
int hbDNNInitializeFromFiles(hbDNNPackedHandle_t* out,const char**,int){*out=(void*)1;++models;return status("initialize");}
int hbDNNGetModelNameList(const char*** out,int* count,hbDNNPackedHandle_t){static const char* names[]={"model"};*out=names;*count=1;return status("names");}
int hbDNNGetModelHandle(hbDNNHandle_t* out,hbDNNPackedHandle_t,const char*){*out=(void*)2;return status("handle");}
int hbDNNGetInputCount(int32_t* count,hbDNNHandle_t){*count=2;return status("input_count");}
int hbDNNGetOutputCount(int32_t* count,hbDNNHandle_t){*count=1;return status("output_count");}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties* p,hbDNNHandle_t,int index){
    *p={};p->tensorType=HB_DNN_TENSOR_TYPE_U8;
    int shape[4]={1,index?512:1024,index?1024:2048,index?2:1};
    for(int i=0;i<4;++i){p->validShape.dimensionSize[i]=shape[i];p->stride[i]=-1;}
    if(failure=="bad_input")p->validShape.dimensionSize[2]=1;
    return status("input_meta");
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties* p,hbDNNHandle_t,int){
    *p={};p->tensorType=HB_DNN_TENSOR_TYPE_S32;p->alignedByteSize=76;
    int shape[4]={1,1,1,19};std::int64_t stride[4]={76,76,76,4};
    for(int i=0;i<4;++i){p->validShape.dimensionSize[i]=shape[i];p->stride[i]=stride[i];}
    if(failure=="bad_output")p->alignedByteSize=4;
    return status("output_meta");
}
int hbUCPMallocCached(hbUCPSysMem* mem,int size,int){mem->virAddr=new unsigned char[size]{};++buffers;++alloc_calls;return status("allocate"+std::to_string(alloc_calls));}
int hbUCPFree(hbUCPSysMem* mem){assert(mem->virAddr);delete[] static_cast<unsigned char*>(mem->virAddr);mem->virAddr=nullptr;--buffers;return 0;}
int hbUCPMemFlush(hbUCPSysMem*,int kind){return status(kind==HB_SYS_MEM_CACHE_CLEAN?"clean":"invalidate");}
int hbDNNInferV2(hbUCPTaskHandle_t* out,hbDNNTensor* output,hbDNNTensor*,hbDNNHandle_t){
    *out=(void*)3;++tasks;std::int32_t value=7;std::memcpy(static_cast<unsigned char*>(output->sysMem.virAddr)+16,&value,4);
    return status("infer");
}
int hbUCPSubmitTask(hbUCPTaskHandle_t,hbUCPSchedParam* schedule){assert(schedule->priority==3);assert(schedule->backend==2);return status("submit");}
int hbUCPWaitTaskDone(hbUCPTaskHandle_t,int){return status("wait");}
int hbUCPReleaseTask(hbUCPTaskHandle_t){--tasks;return 0;}
int hbDNNRelease(hbDNNPackedHandle_t){assert(tasks==0 && buffers==0);--models;return 0;}
int main(){
    const char* cases[]={"initialize","names","handle","input_count","output_count","input_meta","output_meta",
                         "bad_input","bad_output","allocate1","allocate2","allocate3","clean","infer","submit","wait","invalidate",""};
    for(const auto* entry:cases){
        failure=entry;alloc_calls=0;bool threw=false;
        try{
            ModelRunner runner("fixture.hbm","s100",3,1,[](const std::string&,const std::string&){});
            auto raw=runner.run(cv::Mat(1024,2048,CV_8UC1),cv::Mat(512,1024,CV_8UC2));
            assert(tasks==0);
            const auto mask=decode_scores(raw,2,3);
            for(auto id:mask)assert(id==4);
        }catch(const std::exception&){threw=true;}
        assert(threw==!failure.empty());
        assert(models==0 && tasks==0 && buffers==0);
    }
    std::cout<<"17 SDK failure paths and successful resource cleanup passed (fake interfaces only)\n";
}
