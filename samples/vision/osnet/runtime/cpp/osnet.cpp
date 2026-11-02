// RDK X5 OSNet synchronous batch-1 runner and reproducible E2E benchmark.
// Original experiment: delivery/scripts/osnet_e2e.cpp.
// Build with main.cpp, or use run.sh (see README.md).
// Input: already decoded, packed RGB crops; decoder/detector are outside timing.
// Bilinear filter uses 22-bit coefficients and two-pass rounding for Pillow
// compatibility, verified byte-for-byte before any timing is reported.
// Mathematical reference: https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef OSNET_UCP
#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>
#else
#include <dnn/hb_dnn.h>
#endif
#include <arm_neon.h>

constexpr int W=128, H=256, N=W*H, D=512, Q=22;
using Clock=std::chrono::steady_clock;
using Feature=std::array<float,D>;
void check(int status,const char* operation) {
    if(status) throw std::runtime_error(std::string(operation)+": "+std::to_string(status));
}
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
template<class T> void read(std::ifstream& f,T* ptr,size_t n) {
    f.read(reinterpret_cast<char*>(ptr),sizeof(T)*n);
    require(bool(f),"Truncated or unreadable fixture/reference");
}
struct Crop { uint32_t w,h; std::vector<uint8_t> rgb; std::vector<int8_t> expected; };
struct Tap { int begin; std::vector<int> weights; };
std::vector<Tap> coefficients(int source,int target) {
    std::vector<Tap> taps(target);
    const double scale=double(source)/target, support=std::max(1.0,scale);
    for(int i=0;i<target;++i) {
        const double center=(i+0.5)*scale;
        const int lo=std::max(0,int(center-support+0.5));
        const int hi=std::min(source,int(center+support+0.5));
        auto& t=taps[i];t.begin=lo;t.weights.resize(hi-lo);
        std::vector<double> weights(hi-lo);double total=0;
        for(int j=lo;j<hi;++j) total+=(weights[j-lo]=std::max(0.0,1.0-std::abs((j+0.5-center)/support)));
        for(int j=lo;j<hi;++j) t.weights[j-lo]=int(weights[j-lo]/total*(1<<Q)+0.5);
    }
    return taps;
}
class Resize {
    struct Plan {std::vector<Tap> x,y;};
    std::map<std::pair<int,int>,Plan> plans;
    std::vector<uint8_t> scratch;
    static uint8x16_t blend31(uint8x16_t a,uint8x16_t b) {
        auto lo=vaddw_u8(vmull_u8(vget_low_u8(a),vdup_n_u8(3)),vget_low_u8(b));
        auto hi=vaddw_u8(vmull_u8(vget_high_u8(a),vdup_n_u8(3)),vget_high_u8(b));
        return vcombine_u8(vshrn_n_u16(vaddq_u16(lo,vdupq_n_u16(2)),2),
                           vshrn_n_u16(vaddq_u16(hi,vdupq_n_u16(2)),2));
    }
    void twice(const uint8_t* rgb,int8_t* output) {
        // Exact 2x bilinear fast path, retaining horizontal then vertical rounding.
        constexpr int sw=W/2,sh=H/2;scratch.resize(sh*W*3);
        for(int y=0;y<sh;++y) {
            const auto* src=rgb+y*sw*3;auto* dst=scratch.data()+y*W*3;
            std::memcpy(dst,src,3);std::memcpy(dst+(W-1)*3,src+(sw-1)*3,3);
            for(int x=0;x<sw-1;++x) for(int c=0;c<3;++c) {
                const int a=src[x*3+c],b=src[(x+1)*3+c];
                dst[(2*x+1)*3+c]=uint8_t((3*a+b+2)>>2);
                dst[(2*x+2)*3+c]=uint8_t((a+3*b+2)>>2);
            }
        }
        const auto offset=vdupq_n_u8(128);
        for(int x=0;x<W;x+=16) {
            auto first=vld3q_u8(scratch.data()+x*3);
            auto last=vld3q_u8(scratch.data()+((sh-1)*W+x)*3);
            for(int c=0;c<3;++c) {
                vst1q_s8(output+c*N+x,vreinterpretq_s8_u8(vsubq_u8(first.val[c],offset)));
                vst1q_s8(output+c*N+(H-1)*W+x,vreinterpretq_s8_u8(vsubq_u8(last.val[c],offset)));
            }
        }
        for(int y=0;y<sh-1;++y) for(int x=0;x<W;x+=16) {
            auto a=vld3q_u8(scratch.data()+(y*W+x)*3);
            auto b=vld3q_u8(scratch.data()+((y+1)*W+x)*3);
            for(int c=0;c<3;++c) {
                vst1q_s8(output+c*N+(2*y+1)*W+x,vreinterpretq_s8_u8(vsubq_u8(blend31(a.val[c],b.val[c]),offset)));
                vst1q_s8(output+c*N+(2*y+2)*W+x,vreinterpretq_s8_u8(vsubq_u8(blend31(b.val[c],a.val[c]),offset)));
            }
        }
    }
public:
    // Plans and scratch capacity are retained for recurring crop dimensions.
    void run(const uint8_t* rgb,int w,int h,int8_t* output) {
        require(w>0 && h>0 && w<=16384 && h<=16384,"Invalid crop dimensions");
        if(w==W/2 && h==H/2) {twice(rgb,output);return;}
        auto key=std::make_pair(w,h);auto it=plans.find(key);
        if(it==plans.end()) it=plans.emplace(key,Plan{coefficients(w,W),coefficients(h,H)}).first;
        const auto& p=it->second;scratch.resize(size_t(h)*W*3);
        for(int y=0;y<h;++y) for(int x=0;x<W;++x) {
            const auto& t=p.x[x];int sums[3]={1<<(Q-1),1<<(Q-1),1<<(Q-1)};
            for(size_t k=0;k<t.weights.size();++k) {
                const auto* src=rgb+(size_t(y)*w+t.begin+k)*3;
                for(int c=0;c<3;++c) sums[c]+=src[c]*t.weights[k];
            }
            for(int c=0;c<3;++c) scratch[(size_t(y)*W+x)*3+c]=uint8_t(sums[c]>>Q);
        }
        for(int y=0;y<H;++y) for(int x=0;x<W;++x) {
            const auto& t=p.y[y];int sums[3]={1<<(Q-1),1<<(Q-1),1<<(Q-1)};
            for(size_t k=0;k<t.weights.size();++k) {
                const auto* src=scratch.data()+((t.begin+k)*W+x)*3;
                for(int c=0;c<3;++c) sums[c]+=src[c]*t.weights[k];
            }
            for(int c=0;c<3;++c) output[c*N+y*W+x]=int8_t((sums[c]>>Q)-128);
        }
    }
};
#ifdef OSNET_UCP
#include "ucp_backend.inc"
#else
constexpr int OSNET_CORE_ZERO=HB_BPU_CORE_0;
class OSNet {
    hbPackedDNNHandle_t packed=nullptr;
    hbDNNHandle_t handle=nullptr;
    hbDNNTensor input{},output{};
    int core_id;
    void cleanup() noexcept {
        if(output.sysMem[0].virAddr) hbSysFreeMem(&output.sysMem[0]);
        if(input.sysMem[0].virAddr) hbSysFreeMem(&input.sysMem[0]);
        if(packed) hbDNNRelease(packed);
    }
public:
    explicit OSNet(const char* path, int core=HB_BPU_CORE_ANY): core_id(core) {
        try {
            check(hbDNNInitializeFromFiles(&packed,&path,1),"load model");
            const char** names=nullptr;int count=0;
            check(hbDNNGetModelNameList(&names,&count,packed),"model names");require(count==1,"Expected one model");
            check(hbDNNGetModelHandle(&handle,packed,names[0]),"model handle");
            check(hbDNNGetInputCount(&count,handle),"input count");require(count==1,"Expected one input");
            check(hbDNNGetOutputCount(&count,handle),"output count");require(count==1,"Expected one output");
            check(hbDNNGetInputTensorProperties(&input.properties,handle,0),"input metadata");
            check(hbDNNGetOutputTensorProperties(&output.properties,handle,0),"output metadata");
            require(input.properties.tensorType==HB_DNN_IMG_TYPE_RGB,"Expected compiled RGB input");
            require(input.properties.alignedByteSize==N*3,"Unexpected input byte stride/padding");
            require(output.properties.tensorType==HB_DNN_TENSOR_TYPE_F32,"Expected float32 output");
            require(output.properties.alignedByteSize==D*int(sizeof(float)),"Unexpected output padding");
            check(hbSysAllocCachedMem(&input.sysMem[0],input.properties.alignedByteSize),"allocate input");
            check(hbSysAllocCachedMem(&output.sysMem[0],output.properties.alignedByteSize),"allocate output");
        } catch(...) { cleanup();throw; }
    }
    OSNet(const OSNet&)=delete;OSNet& operator=(const OSNet&)=delete;
    ~OSNet(){cleanup();}
    int8_t* input_data(){return static_cast<int8_t*>(input.sysMem[0].virAddr);}
    void prepare_input() {}
    void infer(Feature& result,bool normalize) {
        check(hbSysFlushMem(&input.sysMem[0],HB_SYS_MEM_CACHE_CLEAN),"flush input");
        hbDNNInferCtrlParam control;HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
        control.bpuCoreId=core_id;
        hbDNNTaskHandle_t task=nullptr;hbDNNTensor* out=&output;
        check(hbDNNInfer(&task,&out,&input,handle,&control),"infer");
        const int status=hbDNNWaitTaskDone(task,10000);
        const int release_status=hbDNNReleaseTask(task);
        check(status,"wait for inference");check(release_status,"release task");
        check(hbSysFlushMem(&output.sysMem[0],HB_SYS_MEM_CACHE_INVALIDATE),"invalidate output");
        std::memcpy(result.data(),output.sysMem[0].virAddr,D*sizeof(float));
        if(normalize) {
            float sum=0;for(float v:result) sum+=v*v;
            const float divisor=std::max(std::sqrt(sum),1e-12f);
            for(float& v:result) v/=divisor;
        }
    }
};
#endif
double elapsed(Clock::time_point begin) {
    return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
}
double percentile(const std::vector<double>& sorted,double p) {
    const double x=(sorted.size()-1)*p;const size_t lo=size_t(x),hi=std::min(lo+1,sorted.size()-1);
    return sorted[lo]+(sorted[hi]-sorted[lo])*(x-lo);
}
void stats(std::ostream& out,std::vector<double> values) {
    const double mean=std::accumulate(values.begin(),values.end(),0.0)/values.size();
    std::sort(values.begin(),values.end());
    out<<"{\"mean_ms\":"<<mean<<",\"median_ms\":"<<percentile(values,.5)
       <<",\"p95_ms\":"<<percentile(values,.95)<<",\"p99_ms\":"<<percentile(values,.99)
       <<",\"min_ms\":"<<values.front()<<",\"max_ms\":"<<values.back()<<"}";
}
int osnet_main(int argc,char** argv) {
    try {
        require(argc==5 || argc==6,"Usage: osnet_e2e MODEL images.dat reference.f32 result.json [runs]");
        const int runs=argc==6?std::stoi(argv[5]):1000;require(runs>0,"runs must be positive");
        std::ifstream input(argv[2],std::ios::binary),reference(argv[3],std::ios::binary);
        uint32_t n=0;read(input,&n,1);require(n>0 && n<=1000,"Invalid fixture count");
        std::vector<Crop> images(n);std::vector<Feature> expected(n);
        for(auto& im:images) {
            read(input,&im.w,1);read(input,&im.h,1);
            require(im.w>0 && im.h>0 && im.w<=16384 && im.h<=16384,"Invalid fixture dimensions");
            im.rgb.resize(size_t(im.w)*im.h*3);im.expected.resize(N*3);
            read(input,im.rgb.data(),im.rgb.size());read(input,im.expected.data(),im.expected.size());
        }
        for(auto& f:expected) read(reference,f.data(),D);
        OSNet model(argv[1]);Resize resize;Feature feature{};double max_error=0,max_norm_error=0;
        for(size_t i=0;i<images.size();++i) {
            const auto& im=images[i];resize.run(im.rgb.data(),im.w,im.h,model.input_data());model.prepare_input();
            require(std::memcmp(model.input_data(),im.expected.data(),N*3)==0,"C++ preprocessing differs from PIL");
            model.infer(feature,false);
            for(int j=0;j<D;++j) {
                require(std::isfinite(feature[j]),"Non-finite output");
                const double error=std::abs(feature[j]-expected[i][j]);max_error=std::max(max_error,error);
                if(error>5e-4+1e-4*std::abs(expected[i][j])) {
                    std::cerr<<"Output mismatch image="<<i<<" dim="<<j<<" actual="<<feature[j]<<" reference="<<expected[i][j]<<"\n";
                    throw std::runtime_error("C++ output differs from quantized simulator");
                }
            }
            model.infer(feature,true);double norm=0;for(float v:feature) norm+=v*v;
            max_norm_error=std::max(max_norm_error,std::abs(std::sqrt(norm)-1));
        }
        require(max_norm_error<1e-5,"L2 normalization failed");
        std::vector<double> native(runs),pipeline(runs),preprocess(runs);
        for(int i=0;i<50;++i) model.infer(feature,false);
        for(int i=0;i<runs;++i) {auto start=Clock::now();model.infer(feature,false);native[i]=elapsed(start);}
        for(int i=0;i<50;++i) {
            const auto& im=images[i%n];resize.run(im.rgb.data(),im.w,im.h,model.input_data());model.prepare_input();model.infer(feature,true);
        }
        for(int i=0;i<runs;++i) {
            const auto& im=images[i%n];auto start=Clock::now();
            resize.run(im.rgb.data(),im.w,im.h,model.input_data());model.prepare_input();
            model.infer(feature,true);pipeline[i]=elapsed(start);
        }
        for(int i=0;i<runs;++i) {
            const auto& im=images[i%n];auto start=Clock::now();
            resize.run(im.rgb.data(),im.w,im.h,model.input_data());model.prepare_input();preprocess[i]=elapsed(start);
        }
        std::ofstream out(argv[4]);require(bool(out),"Cannot open result path");out.precision(9);
        out<<"{\n\"scope\":\"decoded RGB crop -> PIL-equivalent resize -> signed planar input -> cache clean -> synchronous infer -> cache invalidate -> output copy -> L2\",\n"
           <<"\"excludes\":\"decode, detector, tracking/matching, model loading\",\n"
           <<"\"batch\":1,\"threads\":1,\"warmup\":50,\"runs\":"<<runs<<",\"verification_samples\":"<<n
           <<",\"preprocess_exact\":true,\"feature_max_abs_error\":"<<max_error<<",\"l2_norm_max_error\":"<<max_norm_error
           <<",\n\"native_forward\":";stats(out,native);
        out<<",\n\"pipeline\":";stats(out,pipeline);
        out<<",\n\"preprocess\":";stats(out,preprocess);out<<"\n}\n";
        std::cout<<"Verified "<<n<<" crops; wrote "<<argv[4]<<"\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}

#include "benchmark.inc"
