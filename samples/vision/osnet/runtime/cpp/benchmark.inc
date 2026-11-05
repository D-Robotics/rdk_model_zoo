// Same-frame timings; independent model, tensor buffers and resize plan per worker.
#include <condition_variable>
#include <exception>
#include <iomanip>
#include <mutex>
#include <memory>
#include <thread>
#include <unistd.h>

struct BenchFrame { double preprocess, runtime, postprocess, total; };
class StartGate {
    std::mutex mutex; std::condition_variable cv;
    int count=0, generation=0; const int workers;
public:
    explicit StartGate(int n): workers(n) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutex); const int current=generation;
        if(++count==workers) {count=0; ++generation; cv.notify_all();}
        else cv.wait(lock,[&]{return generation!=current;});
    }
};
static void normalize_feature(Feature& feature) {
    float sum=0; for(float v:feature) sum+=v*v;
    const float divisor=std::max(std::sqrt(sum),1e-12f);
    for(float& v:feature) v/=divisor;
}
static double millis(Clock::time_point begin,Clock::time_point end) {
    return std::chrono::duration<double,std::milli>(end-begin).count();
}
static void bench_stats(std::ostream& out,std::vector<double> values) {
    const double mean=std::accumulate(values.begin(),values.end(),0.0)/values.size();
    std::sort(values.begin(),values.end());
    out<<"{\"mean\":"<<mean<<",\"p50\":"<<percentile(values,.5)
       <<",\"p95\":"<<percentile(values,.95)<<",\"min\":"<<values.front()<<",\"max\":"<<values.back()<<"}";
}
int osnet_benchmark_main(int argc,char** argv) {
    try {
        require(argc==9,"Usage: osnet --benchmark MODEL images.dat reference.f32 result.json STREAMS WARMUP RUNS ROUNDS");
        const int streams=std::stoi(argv[5]),warmup=std::stoi(argv[6]),runs=std::stoi(argv[7]),rounds=std::stoi(argv[8]);
        require((streams==1||streams==2)&&warmup>=0&&runs>0&&rounds>0,"Invalid benchmark counts");
        std::ifstream input(argv[2],std::ios::binary),reference(argv[3],std::ios::binary);
        uint32_t n=0;read(input,&n,1);require(n>0&&n<=1000,"Invalid fixture count");
        std::vector<Crop> images(n);std::vector<Feature> expected(n);
        for(auto& im:images) {
            read(input,&im.w,1);read(input,&im.h,1);
            require(im.w>0&&im.h>0&&im.w<=16384&&im.h<=16384,"Invalid fixture dimensions");
            im.rgb.resize(size_t(im.w)*im.h*3);im.expected.resize(N*3);
            read(input,im.rgb.data(),im.rgb.size());read(input,im.expected.data(),im.expected.size());
        }
        for(auto& f:expected) read(reference,f.data(),D);
        // Initialize and verify every worker before creating threads or timing.
        std::vector<std::unique_ptr<OSNet>> models;
        std::vector<Resize> resizers(streams); std::vector<Feature> features(streams);
        double max_error=0,max_norm_error=0;
        for(int w=0;w<streams;++w) {
            models.emplace_back(new OSNet(argv[1], OSNET_CORE_ZERO));
            for(size_t i=0;i<images.size();++i) {
                const auto& im=images[i];resizers[w].run(im.rgb.data(),im.w,im.h,models[w]->input_data());models[w]->prepare_input();
                require(std::memcmp(models[w]->input_data(),im.expected.data(),N*3)==0,"Preprocessing differs from Pillow");
                models[w]->infer(features[w],false);
                for(int j=0;j<D;++j) {
                    require(std::isfinite(features[w][j]),"Non-finite feature");
                    const double error=std::abs(features[w][j]-expected[i][j]);max_error=std::max(max_error,error);
                    require(error<=5e-4+1e-4*std::abs(expected[i][j]),"Feature differs from reference");
                }
                normalize_feature(features[w]);double norm=0;for(float v:features[w]) norm+=v*v;
                max_norm_error=std::max(max_norm_error,std::abs(std::sqrt(norm)-1));
            }
        }
        require(max_norm_error<1e-5,"L2 normalization failed");
        std::vector<BenchFrame> frames(size_t(streams)*runs*rounds);
        std::vector<double> walls(rounds);
        for(int round=0;round<rounds;++round) {
            // Warm every instance outside timed work. Errors here cannot strand a barrier.
            for(int w=0;w<streams;++w) for(int i=0;i<warmup;++i) {
                const auto& im=images[i%n];resizers[w].run(im.rgb.data(),im.w,im.h,models[w]->input_data());models[w]->prepare_input();
                models[w]->infer(features[w],false);normalize_feature(features[w]);
            }
            StartGate gate(streams); std::vector<std::thread> workers;
            std::vector<Clock::time_point> starts(streams),ends(streams);
            std::vector<std::exception_ptr> errors(streams);
            for(int w=0;w<streams;++w) workers.emplace_back([&,w,round]{
                gate.wait(); starts[w]=Clock::now();
                try {
                    for(int i=0;i<runs;++i) {
                        const auto& im=images[i%n];const auto t0=Clock::now();
                        resizers[w].run(im.rgb.data(),im.w,im.h,models[w]->input_data());models[w]->prepare_input();const auto t1=Clock::now();
                        models[w]->infer(features[w],false);const auto t2=Clock::now();
                        normalize_feature(features[w]);const auto t3=Clock::now();
                        frames[(size_t(round)*streams+w)*runs+i]={millis(t0,t1),millis(t1,t2),millis(t2,t3),millis(t0,t3)};
                    }
                } catch(...) {errors[w]=std::current_exception();}
                ends[w]=Clock::now();
            });
            for(auto& worker:workers) worker.join();
            for(auto& error:errors) if(error) std::rethrow_exception(error);
            walls[round]=millis(*std::min_element(starts.begin(),starts.end()),*std::max_element(ends.begin(),ends.end()));
        }
        const double wall=std::accumulate(walls.begin(),walls.end(),0.0);
        std::ofstream out(argv[4]);require(bool(out),"Cannot write result");out<<std::setprecision(12);
        out<<"{\"tool\":\"osnet\",\"implementation\":\"native_cpp_neon\",\"timing_scope\":\"decoded_rgb_to_l2_feature\","
           <<"\"core_id\":1,\"pipeline_streams\":"<<streams<<",\"runtime_submission_threads\":"<<streams
           <<",\"cpu_backend\":\"native\",\"cpu_thread_policy\":\"fixed\",\"cpu_worker_threads\":"<<streams
           <<",\"online_cpu_threads\":"<<sysconf(_SC_NPROCESSORS_ONLN)
           <<",\"warmup_frames_per_round\":"<<warmup<<",\"warmup_scope\":\"per_worker_in_process\",\"rounds\":"<<rounds
           <<",\"frames_per_round\":"<<runs<<",\"timed_frames\":"<<frames.size()<<",\"aggregate_wall_ms\":"<<wall
           <<",\"throughput_fps\":"<<frames.size()*1000.0/wall<<",\"functional\":{\"fixtures_per_worker\":"<<n
           <<",\"preprocess_byte_exact\":true,\"max_feature_abs_error\":"<<max_error<<",\"max_l2_norm_error\":"<<max_norm_error<<"},\"metrics_ms\":{";
        const char* names[]={"preprocess","runtime","postprocess","end_to_end"};
        for(int stage=0;stage<4;++stage) {
            std::vector<double> values;for(const auto& f:frames) values.push_back(stage==0?f.preprocess:stage==1?f.runtime:stage==2?f.postprocess:f.total);
            if(stage) out<<',';
            out<<'"'<<names[stage]<<"\":";bench_stats(out,values);
        }
        out<<"},\"round_wall_ms\":[";for(size_t i=0;i<walls.size();++i) {if(i)out<<',';out<<walls[i];}
        out<<"],\"raw_frame_ms\":[";
        for(size_t i=0;i<frames.size();++i) {const auto& f=frames[i];if(i)out<<',';out<<'['<<f.preprocess<<','<<f.runtime<<','<<f.postprocess<<','<<f.total<<']';}
        out<<"],\"raw_frame_order\":\"round,worker,frame\"}\n";
        require(bool(out),"Failed writing result");return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
