// S-series UCP backend: normalized F32 NCHW input, owned buffers per worker.
constexpr int OSNET_CORE_ZERO=HB_UCP_BPU_CORE_0;
class OSNet {
    hbDNNPackedHandle_t packed=nullptr;
    hbDNNHandle_t handle=nullptr;
    hbDNNTensor input{},output{};
    std::vector<int8_t> rgb=std::vector<int8_t>(N*3);
    std::array<size_t,D> output_offsets{};
    int core_id;
    void cleanup() noexcept {
        if(output.sysMem.virAddr) hbUCPFree(&output.sysMem);
        if(input.sysMem.virAddr) hbUCPFree(&input.sysMem);
        if(packed) hbDNNRelease(packed);
    }
public:
    explicit OSNet(const char* path,int core=HB_UCP_BPU_CORE_ANY):core_id(core) {
        try {
            check(hbDNNInitializeFromFiles(&packed,&path,1),"load model");
            const char** names=nullptr;int count=0;
            check(hbDNNGetModelNameList(&names,&count,packed),"model names");require(count==1,"Expected one model");
            check(hbDNNGetModelHandle(&handle,packed,names[0]),"model handle");
            check(hbDNNGetInputCount(&count,handle),"input count");require(count==1,"Expected one input");
            check(hbDNNGetOutputCount(&count,handle),"output count");require(count==1,"Expected one output");
            check(hbDNNGetInputTensorProperties(&input.properties,handle,0),"input metadata");
            check(hbDNNGetOutputTensorProperties(&output.properties,handle,0),"output metadata");
            require(input.properties.tensorType==HB_DNN_TENSOR_TYPE_F32,"Expected F32 featuremap input");
            const auto& shape=input.properties.validShape;
            require(shape.numDimensions==4&&shape.dimensionSize[0]==1&&shape.dimensionSize[1]==3&&shape.dimensionSize[2]==H&&shape.dimensionSize[3]==W,"Expected [1,3,256,128] input");
            for(int i=0;i<4;++i)require(input.properties.stride[i]>0,"Invalid input stride");
            const size_t last=2*input.properties.stride[1]+(H-1)*input.properties.stride[2]+(W-1)*input.properties.stride[3]+sizeof(float);
            require(last<=input.properties.alignedByteSize,"Input stride exceeds allocation");
            require(output.properties.tensorType==HB_DNN_TENSOR_TYPE_F32,"Expected F32 embedding");
            const auto& out=output.properties.validShape;
            require(out.numDimensions>0&&out.numDimensions<=8,"Invalid output rank");
            size_t values=1;for(int i=0;i<out.numDimensions;++i){require(out.dimensionSize[i]>0,"Invalid output extent");values*=out.dimensionSize[i];require(values<=D,"Unexpected output size");}
            require(values==D,"Expected 512 output values");
            for(int j=0;j<D;++j){size_t remainder=j,offset=0;for(int i=out.numDimensions-1;i>=0;--i){require(output.properties.stride[i]>0,"Invalid output stride");offset+=(remainder%out.dimensionSize[i])*output.properties.stride[i];remainder/=out.dimensionSize[i];}require(offset+4<=output.properties.alignedByteSize,"Output stride exceeds allocation");output_offsets[j]=offset;}
            check(hbUCPMallocCached(&input.sysMem,input.properties.alignedByteSize,0),"allocate input");
            check(hbUCPMallocCached(&output.sysMem,output.properties.alignedByteSize,0),"allocate output");
            std::memset(input.sysMem.virAddr,0,input.properties.alignedByteSize);
        } catch(...) {cleanup();throw;}
    }
    OSNet(const OSNet&)=delete;OSNet& operator=(const OSNet&)=delete;
    ~OSNet(){cleanup();}
    int8_t* input_data(){return rgb.data();}
    void prepare_input(){
        constexpr float mean[]={.485f,.456f,.406f},stddev[]={.229f,.224f,.225f};
        auto* base=static_cast<uint8_t*>(input.sysMem.virAddr);
        for(int c=0;c<3;++c) for(int y=0;y<H;++y) for(int x=0;x<W;++x){
            const float value=((float(int(rgb[c*N+y*W+x])+128)/255.0f)-mean[c])/stddev[c];
            std::memcpy(base+c*input.properties.stride[1]+y*input.properties.stride[2]+x*input.properties.stride[3],&value,sizeof(value));
        }
    }
    void infer(Feature& result,bool normalize){
        check(hbUCPMemFlush(&input.sysMem,HB_SYS_MEM_CACHE_CLEAN),"flush input");
        hbUCPTaskHandle_t task=nullptr;
        try {
            check(hbDNNInferV2(&task,&output,&input,handle),"create inference");
            hbUCPSchedParam sched;HB_UCP_INITIALIZE_SCHED_PARAM(&sched);sched.backend=core_id;sched.priority=0;
            check(hbUCPSubmitTask(task,&sched),"submit inference");
            check(hbUCPWaitTaskDone(task,10000),"wait inference");
        } catch(...) {if(task)hbUCPReleaseTask(task);throw;}
        check(hbUCPReleaseTask(task),"release task");
        check(hbUCPMemFlush(&output.sysMem,HB_SYS_MEM_CACHE_INVALIDATE),"invalidate output");
        const auto* base=static_cast<const uint8_t*>(output.sysMem.virAddr);
        for(int j=0;j<D;++j)std::memcpy(&result[j],base+output_offsets[j],sizeof(float));
        if(normalize){float sum=0;for(float v:result)sum+=v*v;const float d=std::max(std::sqrt(sum),1e-12f);for(float& v:result)v/=d;}
    }
};
