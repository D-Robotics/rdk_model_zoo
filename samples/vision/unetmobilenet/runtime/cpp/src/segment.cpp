// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// UnetMobileNet S100/S600 model: board identity, tensor contract, runtime
// ownership and stage math. DNN/UCP handles and buffers live in the private
// Impl in this file; the public header exposes only owned stage-data types.
#include "segment.hpp"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <opencv2/imgproc.hpp>

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
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max()/b)
        throw std::invalid_argument("Tensor size overflow");
    return a*b;
}
std::size_t add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max()-b)
        throw std::invalid_argument("Tensor offset overflow");
    return a+b;
}
std::size_t axis_index(const ScoreSpec& spec, std::size_t y, std::size_t x, int c) {
    const int axis=spec.axis<0?spec.axis+4:spec.axis;
    const std::array<std::size_t,4> coordinates{0,y,x,static_cast<std::size_t>(c)};
    return spec.scales.size()==1?0:coordinates[axis];
}
std::string normalize_identity(std::string value) {
    const auto begin=value.find_first_not_of(" \t\r\n");
    if(begin==std::string::npos)return {};
    value=value.substr(begin,value.find_last_not_of(" \t\r\n")-begin+1);
    std::transform(value.begin(),value.end(),value.begin(),[](unsigned char c){return std::tolower(c);});
    return value;
}
std::string read_identity(const char* path) {
    std::ifstream stream(path);
    return std::string(std::istreambuf_iterator<char>(stream),{});
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
    TaskGuard() = default;
    hbUCPTaskHandle_t handle=nullptr;
    ~TaskGuard(){if(handle)hbUCPReleaseTask(handle);}
    TaskGuard(const TaskGuard&)=delete;
    TaskGuard& operator=(const TaskGuard&)=delete;
};
}  // namespace

// ===========================================================================
// Board identity
// ===========================================================================
std::string match_native_target(std::string soc, std::string board) {
    soc=normalize_identity(soc);board=normalize_identity(board);
    if(soc=="s100" && (board=="s100p" || board=="rdk s100p"))return "s100p";
    if(soc=="s100" || soc=="s100p" || soc=="s600")return soc;
    return {};
}
void require_native_target(const std::string& requested, const std::string& built) {
    if(requested!=built || (requested!="s100" && requested!="s600"))
        throw std::invalid_argument("Requested target does not match the S100/S600 build");
    const auto actual=match_native_target(read_identity("/sys/class/boardinfo/soc_name"),
                                          read_identity("/sys/class/boardinfo/board_type"));
    if(actual!=requested)throw std::invalid_argument("Local board identity does not match requested target");
}

// ===========================================================================
// Score tensor contract
// ===========================================================================
void validate_scores(const ScoreSpec& spec) {
    if (spec.shape[0]!=1 || spec.shape[3]!=19 || spec.shape[1]<=0 || spec.shape[2]<=0)
        throw std::invalid_argument("Expected NHWC [1,H,W,19] scores");
    for (auto stride:spec.stride)
        if (stride<=0) throw std::invalid_argument("Unresolved or invalid output stride");
    if (spec.stride[3]<4) throw std::invalid_argument("Score elements overlap");
    for (int i=0;i<3;++i)
        if (static_cast<std::size_t>(spec.stride[i]) < multiply(spec.stride[i+1],spec.shape[i+1]))
            throw std::invalid_argument("Tensor rows or channels overlap");
    std::size_t span=4;
    for (int i=0;i<4;++i) span=add(span,multiply(spec.shape[i]-1,spec.stride[i]));
    if (span>spec.storage_bytes) throw std::invalid_argument("Score tensor exceeds allocation");
    if (spec.type==ScoreType::Int32 && spec.scaled) {
        if (spec.scales.empty()) throw std::invalid_argument("Missing SCALE values");
        for (float scale:spec.scales)
            if (!std::isfinite(scale) || scale<=0) throw std::invalid_argument("Invalid SCALE value");
        if (spec.scales.size()==1) {
            if (spec.zero_points.size()>1) throw std::invalid_argument("Scalar scale needs scalar offset");
        } else {
            int axis=spec.axis<0?spec.axis+4:spec.axis;
            if (axis<0 || axis>=4 || spec.scales.size()!=static_cast<std::size_t>(spec.shape[axis]))
                throw std::invalid_argument("SCALE axis/length mismatch");
            if (!spec.zero_points.empty() && spec.zero_points.size()!=1 && spec.zero_points.size()!=spec.scales.size())
                throw std::invalid_argument("Zero-point length mismatch");
        }
    }
}

std::vector<std::int32_t> decode_scores(const RawScores& raw, int height, int width) {
    validate_scores(raw.spec);
    if (height<=0 || width<=0 || raw.bytes.size()<raw.spec.storage_bytes)
        throw std::invalid_argument("Invalid original geometry or truncated scores");
    const auto& spec=raw.spec;
    std::vector<std::int32_t> small(multiply(spec.shape[1],spec.shape[2]));
    for (std::size_t y=0;y<static_cast<std::size_t>(spec.shape[1]);++y) {
        for (std::size_t x=0;x<static_cast<std::size_t>(spec.shape[2]);++x) {
            double best=-std::numeric_limits<double>::infinity();
            int winner=0;
            for (int c=0;c<19;++c) {
                const auto offset=y*spec.stride[1]+x*spec.stride[2]+c*spec.stride[3];
                double value;
                if (spec.type==ScoreType::Int32) {
                    std::int32_t integer=0;
                    std::memcpy(&integer,raw.bytes.data()+offset,4);
                    value=integer;
                    if (spec.scaled) {
                        const auto qi=axis_index(spec,y,x,c);
                        const auto zero=spec.zero_points.empty()?0:spec.zero_points[spec.zero_points.size()==1?0:qi];
                        value=(value-static_cast<double>(zero))*spec.scales[qi];
                    }
                } else {
                    float floating=0;
                    std::memcpy(&floating,raw.bytes.data()+offset,4);
                    value=floating;
                }
                if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite output score");
                if (value>best) { best=value;winner=c; }
            }
            small[y*spec.shape[2]+x]=winner;
        }
    }
    std::vector<std::int32_t> restored(multiply(height,width));
    for (int y=0;y<height;++y) {
        const auto source_y=multiply(y,spec.shape[1])/height;
        for (int x=0;x<width;++x) {
            const auto source_x=multiply(x,spec.shape[2])/width;
            restored[static_cast<std::size_t>(y)*width+x]=small[source_y*spec.shape[2]+source_x];
        }
    }
    return restored;
}

// ===========================================================================
// UnetMobileNet model
// ===========================================================================
struct UnetMobileNet::Impl {
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

UnetMobileNet::UnetMobileNet(const std::string& path,const std::string& target,int priority,int core,ExecutionGate gate)
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
UnetMobileNet::~UnetMobileNet()=default;

PreparedInput UnetMobileNet::preprocess(const cv::Mat& image) const {
    if(image.empty() || image.type()!=CV_8UC3)throw std::invalid_argument("Expected nonempty BGR uint8 image");
    cv::Mat resized,i420;
    cv::resize(image,resized,cv::Size(2048,1024),0,0,cv::INTER_AREA);
    cv::cvtColor(resized,i420,cv::COLOR_BGR2YUV_I420);
    PreparedInput result{cv::Mat(1024,2048,CV_8UC1),cv::Mat(512,1024,CV_8UC2),{image.rows,image.cols}};
    const auto* source=i420.ptr<unsigned char>();
    std::memcpy(result.y.data,source,1024*2048);
    const auto* u=source+1024*2048;
    const auto* v=u+512*1024;
    auto* uv=result.uv.ptr<unsigned char>();
    for(int i=0;i<512*1024;++i) { uv[2*i]=u[i];uv[2*i+1]=v[i]; }
    return result;
}

RawScores UnetMobileNet::infer(const PreparedInput& prepared) {
    const cv::Mat& y=prepared.y;
    const cv::Mat& uv=prepared.uv;
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
    // Release the finished task and report a failed release; the guard covers
    // every exceptional path above and never releases twice.
    const int rc=hbUCPReleaseTask(task.handle);
    task.handle=nullptr;
    checked(rc,"task release");
    return result;
}

cv::Mat UnetMobileNet::postprocess(const RawScores& raw,const ImageContext& context) const {
    const auto labels=decode_scores(raw,context.original_height,context.original_width);
    cv::Mat result(context.original_height,context.original_width,CV_32S);
    std::memcpy(result.data,labels.data(),labels.size()*sizeof(std::int32_t));
    return result;
}

cv::Mat UnetMobileNet::predict(const cv::Mat& image) {
    const auto prepared=preprocess(image);
    return postprocess(infer(prepared),prepared.context);
}
const ScoreSpec& UnetMobileNet::score_spec() const { return impl_->scores; }
}  // namespace unetmobilenet
