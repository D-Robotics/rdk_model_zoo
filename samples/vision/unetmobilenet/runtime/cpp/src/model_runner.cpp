// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "model_runner.hpp"
#include "target_identity.hpp"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifndef UNETMOBILENET_TARGET_NAME
#define UNETMOBILENET_TARGET_NAME "unknown"
#endif
namespace unetmobilenet {
namespace {
void checked(int rc, const char* operation) {
    if(rc!=0)throw std::runtime_error(std::string(operation)+" failed: "+std::to_string(rc));
}
std::size_t allocation_size(std::int64_t value) {
    if(value<=0 || value>std::numeric_limits<int>::max())
        throw std::invalid_argument("Invalid SDK allocation size");
    return static_cast<std::size_t>(value);
}
std::size_t bind_plane(hbDNNTensorProperties& p, int h, int w, int channels, int alignment) {
    if(p.validShape.numDimensions!=4 || p.tensorType!=HB_DNN_TENSOR_TYPE_U8)
        throw std::invalid_argument("Split NV12 inputs require rank-4 uint8");
    const std::array<int,4> expected{1,h,w,channels};
    for(int i=0;i<4;++i)
        if(p.validShape.dimensionSize[i]!=expected[i])throw std::invalid_argument("Wrong split NV12 geometry/order");
    if(p.stride[3]==-1)p.stride[3]=1;
    if(p.stride[2]==-1)p.stride[2]=channels;
    if(p.stride[1]==-1)p.stride[1]=((w*channels+alignment-1)/alignment)*alignment;
    if(p.stride[1]<w*channels || p.stride[1]>std::numeric_limits<int>::max()/h)
        throw std::invalid_argument("Invalid NV12 row pitch");
    if(p.stride[0]==-1)p.stride[0]=p.stride[1]*h;
    if(p.stride[3]!=1 || p.stride[2]!=channels || p.stride[0]<p.stride[1]*h)
        throw std::invalid_argument("Unsupported NV12 byte strides");
    return allocation_size(p.stride[0]);
}
ScoreSpec bind_scores(const hbDNNTensorProperties& p) {
    if(p.validShape.numDimensions!=4)throw std::invalid_argument("Scores must be rank-4 NHWC");
    ScoreSpec result;
    for(int i=0;i<4;++i) { result.shape[i]=p.validShape.dimensionSize[i];result.stride[i]=p.stride[i]; }
    result.storage_bytes=allocation_size(p.alignedByteSize);
    if(p.tensorType==HB_DNN_TENSOR_TYPE_S32)result.type=ScoreType::Int32;
    else if(p.tensorType==HB_DNN_TENSOR_TYPE_F32)result.type=ScoreType::Float32;
    else throw std::invalid_argument("Scores must be int32 or float32");
    validate_scores(result);  // validate geometry/capacity before copying descriptor arrays
    if(result.type==ScoreType::Int32) {
        if(p.quantiType!=NONE && p.quantiType!=SCALE)throw std::invalid_argument("Unsupported integer quantization");
        result.scaled=p.quantiType==SCALE;
        if(result.scaled) {
            result.axis=p.quantizeAxis;
            const int axis=result.axis<0?result.axis+4:result.axis;
            const auto count=p.scale.scaleLen;
            if(count<=0 || !p.scale.scaleData || (count!=1 && (axis<0 || axis>=4 || count!=result.shape[axis])))
                throw std::invalid_argument("Invalid scale pointer/length/axis");
            const auto zeros=p.scale.zeroPointLen;
            if(zeros<0 || (zeros!=0 && zeros!=1 && zeros!=count) || (zeros>0 && !p.scale.zeroPointData))
                throw std::invalid_argument("Invalid zero-point pointer/length");
            result.scales.assign(p.scale.scaleData,p.scale.scaleData+count);
            if(zeros>0)result.zero_points.assign(p.scale.zeroPointData,p.scale.zeroPointData+zeros);
        }
    }
    validate_scores(result);
    return result;
}
struct TaskGuard {
    hbUCPTaskHandle_t handle=nullptr;
    ~TaskGuard(){if(handle)hbUCPReleaseTask(handle);}
};
}

struct ModelRunner::Impl {
    hbDNNPackedHandle_t packed=nullptr;
    hbDNNHandle_t model=nullptr;
    std::array<hbDNNTensor,2> inputs{};
    hbDNNTensor output{};
    std::array<std::size_t,2> input_bytes{};
    ScoreSpec scores;
    int priority=0;
    unsigned long long backend=1ULL<<7;
    ~Impl() {
        for(auto& input:inputs)if(input.sysMem.virAddr)hbUCPFree(&input.sysMem);
        if(output.sysMem.virAddr)hbUCPFree(&output.sysMem);
        if(packed)hbDNNRelease(packed);
    }
};

ModelRunner::ModelRunner(const std::string& path,const std::string& target,int priority,int core,ExecutionGate gate)
    :impl_(std::make_unique<Impl>()) {
    if(gate)gate(target,UNETMOBILENET_TARGET_NAME);
    else require_native_target(target,UNETMOBILENET_TARGET_NAME);
    if(priority<0 || priority>255 || core< -1 || core>3)
        throw std::invalid_argument("priority must be 0..255 and bpu-core -1 or 0..3");
    impl_->priority=priority;
    impl_->backend=core==-1?(1ULL<<7):(1ULL<<core);
    const char* filename=path.c_str();
    checked(hbDNNInitializeFromFiles(&impl_->packed,&filename,1),"model initialization");
    const char** names=nullptr;int count=0;
    checked(hbDNNGetModelNameList(&names,&count,impl_->packed),"model name query");
    if(count!=1 || !names || !names[0])throw std::invalid_argument("Expected one model in HBM");
    checked(hbDNNGetModelHandle(&impl_->model,impl_->packed,names[0]),"model handle query");
    int32_t input_count=0,output_count=0;
    checked(hbDNNGetInputCount(&input_count,impl_->model),"input count query");
    checked(hbDNNGetOutputCount(&output_count,impl_->model),"output count query");
    if(input_count!=2 || output_count!=1)throw std::invalid_argument("Expected two inputs and one output");
    for(int i=0;i<2;++i)
        checked(hbDNNGetInputTensorProperties(&impl_->inputs[i].properties,impl_->model,i),"input metadata");
    checked(hbDNNGetOutputTensorProperties(&impl_->output.properties,impl_->model,0),"output metadata");
    const int alignment=target=="s600"?64:32;
    impl_->input_bytes[0]=bind_plane(impl_->inputs[0].properties,1024,2048,1,alignment);
    impl_->input_bytes[1]=bind_plane(impl_->inputs[1].properties,512,1024,2,alignment);
    impl_->scores=bind_scores(impl_->output.properties);
    for(int i=0;i<2;++i) {
        checked(hbUCPMallocCached(&impl_->inputs[i].sysMem,static_cast<int>(impl_->input_bytes[i]),0),"input allocation");
        if(!impl_->inputs[i].sysMem.virAddr)throw std::runtime_error("Null input allocation");
        std::memset(impl_->inputs[i].sysMem.virAddr,0,impl_->input_bytes[i]);
    }
    checked(hbUCPMallocCached(&impl_->output.sysMem,static_cast<int>(impl_->scores.storage_bytes),0),"output allocation");
    if(!impl_->output.sysMem.virAddr)throw std::runtime_error("Null output allocation");
}
ModelRunner::~ModelRunner()=default;

RawScores ModelRunner::run(const cv::Mat& y,const cv::Mat& uv) {
    if(y.rows!=1024 || y.cols!=2048 || y.type()!=CV_8UC1 || uv.rows!=512 || uv.cols!=1024 || uv.type()!=CV_8UC2)
        throw std::invalid_argument("Prepared NV12 planes have wrong geometry/type");
    const std::array<const cv::Mat*,2> planes{&y,&uv};
    for(int i=0;i<2;++i) {
        auto& tensor=impl_->inputs[i];
        auto* base=static_cast<unsigned char*>(tensor.sysMem.virAddr);
        for(int row=0;row<planes[i]->rows;++row)
            std::memcpy(base+row*tensor.properties.stride[1],planes[i]->ptr(row),2048);
        checked(hbUCPMemFlush(&tensor.sysMem,HB_SYS_MEM_CACHE_CLEAN),"input cache clean");
    }
    TaskGuard task;
    checked(hbDNNInferV2(&task.handle,&impl_->output,impl_->inputs.data(),impl_->model),"inference creation");
    hbUCPSchedParam schedule;HB_UCP_INITIALIZE_SCHED_PARAM(&schedule);
    schedule.backend=impl_->backend;schedule.priority=impl_->priority;
    checked(hbUCPSubmitTask(task.handle,&schedule),"task submission");
    checked(hbUCPWaitTaskDone(task.handle,0),"task wait");
    checked(hbUCPMemFlush(&impl_->output.sysMem,HB_SYS_MEM_CACHE_INVALIDATE),"output cache invalidate");
    RawScores result{impl_->scores,{}};
    result.bytes.resize(result.spec.storage_bytes);
    std::memcpy(result.bytes.data(),impl_->output.sysMem.virAddr,result.bytes.size());
    return result;
}
}  // namespace unetmobilenet
