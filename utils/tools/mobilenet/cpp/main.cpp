// SPDX-License-Identifier: Apache-2.0
// Candidate classification benchmark; batch one, exact frozen geometry.
#include "geometry.hpp"
#ifdef TARGET_S100
#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>
#include <hobot/hb_ucp_sys.h>
#else
#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#endif
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>

using Clock = std::chrono::steady_clock;
static void check(int code, const char* call) {
  if (code) throw std::runtime_error(std::string(call) + " returned " + std::to_string(code));
}
#define CHECK(call) check((call), #call)
// Row byte alignment the S-series runtime requires for dynamic NV12 strides:
// 32 on nash-e/nash-m (224-byte rows stay packed), 64 on nash-p (rows pad to 256).
#ifndef NV12_ROW_ALIGN
#define NV12_ROW_ALIGN 32
#endif
#ifdef TARGET_S100
using Memory = hbUCPSysMem;
static Memory& memory(hbDNNTensor& t) { return t.sysMem; }
static int allocate(Memory* m, int64_t bytes) { return hbUCPMallocCached(m, bytes, 0); }
static int release(Memory* m) { return hbUCPFree(m); }
static int flush(Memory* m, int flag) { return hbUCPMemFlush(m, flag); }
#else
using Memory = hbSysMem;
using hbDNNPackedHandle_t = hbPackedDNNHandle_t;
static Memory& memory(hbDNNTensor& t) { return t.sysMem[0]; }
static int allocate(Memory* m, int64_t bytes) { return hbSysAllocCachedMem(m, bytes); }
static int release(Memory* m) { return hbSysFreeMem(m); }
static int flush(Memory* m, int flag) { return hbSysFlushMem(m, flag); }
#endif

/** Own one loaded model and one set of reusable tensors per pipeline stream. */
class Session {
 public:
  void init(const std::string& path) {
    const char* file = path.c_str(); CHECK(hbDNNInitializeFromFiles(&packed_, &file, 1));
    const char** names = nullptr; int count = 0;
    CHECK(hbDNNGetModelNameList(&names, &count, packed_));
    if (count != 1) throw std::runtime_error("Expected one compiled model");
    CHECK(hbDNNGetModelHandle(&model_, packed_, names[0]));
    CHECK(hbDNNGetInputCount(&count, model_));
#ifdef TARGET_S100
    if (count != 2) throw std::runtime_error("Expected split NV12 inputs");
#else
    if (count != 1) throw std::runtime_error("Expected packed NV12 input");
#endif
    inputs_.resize(count);
    for (int i = 0; i < count; ++i) {
      auto& t = inputs_[i]; CHECK(hbDNNGetInputTensorProperties(&t.properties, model_, i));
#ifdef TARGET_S100
      auto& p = t.properties;
      const std::array<int, 4> expected = i == 0 ? std::array<int,4>{1,224,224,1} : std::array<int,4>{1,112,112,2};
      if (p.tensorType != HB_DNN_TENSOR_TYPE_U8 || p.validShape.numDimensions != 4) throw std::runtime_error("Wrong S100 input type");
      for (int k = 0; k < 4; ++k) if (p.validShape.dimensionSize[k] != expected[k]) throw std::runtime_error("Wrong S100 input shape");
      for (int k = 3; k >= 0; --k) if (p.stride[k] < 0) {
        if (k == 3) throw std::runtime_error("Unknown element stride");
        p.stride[k] = ((p.stride[k + 1] * expected[k + 1] + NV12_ROW_ALIGN - 1) / NV12_ROW_ALIGN) * NV12_ROW_ALIGN;
      }
      const int packedRow = expected[2] * expected[3];
      const int alignedRow = ((packedRow + NV12_ROW_ALIGN - 1) / NV12_ROW_ALIGN) * NV12_ROW_ALIGN;
      if (p.stride[1] != alignedRow || p.stride[2] != expected[3] || p.stride[3] != 1) throw std::runtime_error("Unsupported NV12 stride");
      p.alignedByteSize = p.stride[0];
      rows_.push_back(expected[1]); packedRow_.push_back(packedRow); strideRow_.push_back(alignedRow);
#else
      auto& p = t.properties;
      if (p.tensorType != HB_DNN_IMG_TYPE_NV12 || p.alignedByteSize != 224*224*3/2 || p.tensorLayout != HB_DNN_LAYOUT_NCHW) throw std::runtime_error("Unexpected packed NV12 properties");
      const int dims[] = {1,3,224,224};
      if (p.validShape.numDimensions != 4) throw std::runtime_error("Wrong X5 rank");
      for (int k = 0; k < 4; ++k) if (p.validShape.dimensionSize[k] != dims[k] || p.alignedShape.dimensionSize[k] != dims[k]) throw std::runtime_error("Unsupported X5 padding");
      rows_.push_back(1); packedRow_.push_back(p.alignedByteSize); strideRow_.push_back(p.alignedByteSize);
#endif
      CHECK(allocate(&memory(t), t.properties.alignedByteSize));
      std::memset(memory(t).virAddr, 0, size_t(t.properties.alignedByteSize));
    }
    CHECK(hbDNNGetOutputCount(&count, model_));
    if (count != 1) throw std::runtime_error("Expected one logits output");
    CHECK(hbDNNGetOutputTensorProperties(&output_.properties, model_, 0));
    auto& p = output_.properties; int elements = 1;
    for (int k = 0; k < p.validShape.numDimensions; ++k) elements *= p.validShape.dimensionSize[k];
    if (elements != 1000 || p.tensorType != HB_DNN_TENSOR_TYPE_F32 || p.quantiType != NONE || p.stride[1] != 4) throw std::runtime_error("Expected contiguous 1000 float logits");
    CHECK(allocate(&memory(output_), p.alignedByteSize));
  }
  void upload(const std::vector<uint8_t>& bytes) {
    size_t offset = 0;
    for (size_t i = 0; i < inputs_.size(); ++i) {
      auto& t = inputs_[i];
      const size_t rows = rows_[i], packed = packedRow_[i], stride = strideRow_[i];
      if (offset + rows * packed > bytes.size()) throw std::runtime_error("Input allocation/byte count mismatch");
      auto* target = static_cast<uint8_t*>(memory(t).virAddr);
      if (packed == stride) std::memcpy(target, bytes.data() + offset, rows * packed);
      else for (size_t r = 0; r < rows; ++r) std::memcpy(target + r * stride, bytes.data() + offset + r * packed, packed);
      offset += rows * packed;
      CHECK(flush(&memory(t), HB_SYS_MEM_CACHE_CLEAN));
    }
    if (offset != bytes.size()) throw std::runtime_error("Unconsumed input bytes");
  }
  void infer() {
#ifdef TARGET_S100
    hbUCPTaskHandle_t task = nullptr;
    try {
      CHECK(hbDNNInferV2(&task, &output_, inputs_.data(), model_));
      hbUCPSchedParam param; HB_UCP_INITIALIZE_SCHED_PARAM(&param); param.backend = HB_UCP_BPU_CORE_ANY;
      CHECK(hbUCPSubmitTask(task, &param)); CHECK(hbUCPWaitTaskDone(task, 0));
    } catch (...) { if (task) { hbUCPWaitTaskDone(task, 0); hbUCPReleaseTask(task); } throw; }
    CHECK(hbUCPReleaseTask(task));
#else
    hbDNNTaskHandle_t task = nullptr; hbDNNTensor* output = &output_;
    hbDNNInferCtrlParam param; HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&param);
    try { CHECK(hbDNNInfer(&task, &output, inputs_.data(), model_, &param)); CHECK(hbDNNWaitTaskDone(task, 0)); }
    catch (...) { if (task) { hbDNNWaitTaskDone(task, 0); hbDNNReleaseTask(task); } throw; }
    CHECK(hbDNNReleaseTask(task));
#endif
  }
  std::array<float,1000> logits() {
    CHECK(flush(&memory(output_), HB_SYS_MEM_CACHE_INVALIDATE));
    std::array<float,1000> result; std::memcpy(result.data(), memory(output_).virAddr, sizeof(result));
    for (float value : result) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite logits");
    return result;
  }
  ~Session() {
    for (auto& t : inputs_) if (memory(t).virAddr) { int r = release(&memory(t)); if (r) std::cerr << "input release error " << r << '\n'; }
    if (memory(output_).virAddr) { int r = release(&memory(output_)); if (r) std::cerr << "output release error " << r << '\n'; }
    if (packed_) { int r = hbDNNRelease(packed_); if (r) std::cerr << "model release error " << r << '\n'; }
  }
 private:
  hbDNNPackedHandle_t packed_ = nullptr; hbDNNHandle_t model_ = nullptr;
  std::vector<hbDNNTensor> inputs_; hbDNNTensor output_{};
  std::vector<size_t> rows_, packedRow_, strideRow_;  // per input: rows, packed and padded row bytes
};
static int resize_shorter = 256;  // set once from argv before any worker starts
struct Row { int stream, frame, image; double preprocess, runtime, postprocess, e2e; std::array<int,5> top5; };
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
static Row step(Session& session, const cv::Mat& image, int stream, int frame, int image_id, std::array<float,1000>* saved = nullptr) {
  auto t0 = Clock::now(); auto crop = mobilenet::center_crop(image, 224, resize_shorter); auto bytes = mobilenet::nv12(crop); session.upload(bytes);
  auto t1 = Clock::now(); session.infer(); auto t2 = Clock::now(); auto scores = session.logits();
  std::array<int,1000> ids; std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin()+5, ids.end(), [&](int a,int b) { return scores[a] == scores[b] ? a < b : scores[a] > scores[b]; });
  Row row{stream,frame,image_id,0,0,0,0,{}}; std::copy(ids.begin(),ids.begin()+5,row.top5.begin()); auto t3 = Clock::now();
  row.preprocess=ms(t0,t1); row.runtime=ms(t1,t2); row.postprocess=ms(t2,t3); row.e2e=ms(t0,t3);
  if (saved) *saved = scores;
  return row;
}
/** CLI: model image-list output-prefix streams frames-per-stream warmup cpu-threads. */
int main(int argc, char** argv) {
  try {
    if (argc != 8 && argc != 9) throw std::runtime_error("Usage: benchmark MODEL IMAGE_LIST OUTPUT_PREFIX STREAMS FRAMES WARMUP CPU_THREADS [RESIZE_SHORTER]");
    // Shorter-edge resize before the 224 crop: int(224 / crop_pct), 256 for Small and 235 for Medium-224.
    resize_shorter = argc == 9 ? std::stoi(argv[8]) : 256;
    if (resize_shorter < 224) throw std::runtime_error("RESIZE_SHORTER must be at least 224");
    const int streams=std::stoi(argv[4]), frames=std::stoi(argv[5]), warmup=std::stoi(argv[6]), threads=std::stoi(argv[7]);
    if ((streams != 1 && streams != 2) || frames <= 0 || warmup < 0 || threads <= 0) throw std::runtime_error("Invalid benchmark configuration");
    cv::setNumThreads(threads); std::ifstream list(argv[2]); std::string path; std::vector<cv::Mat> images;
    while (std::getline(list,path)) { if (path.empty()) continue; auto image=cv::imread(path); if (image.empty()) throw std::runtime_error("Cannot decode "+path); images.push_back(image); }
    if (images.empty()) throw std::runtime_error("Empty image list");
    const std::string prefix=argv[3];
    for (size_t i=0;i<images.size();++i) cv::imwrite(prefix+"-crop-"+std::to_string(i)+".png",mobilenet::center_crop(images[i], 224, resize_shorter));
    std::vector<std::unique_ptr<Session>> sessions;
    for (int s=0;s<streams;++s) { auto ptr=std::make_unique<Session>();ptr->init(argv[1]);sessions.push_back(std::move(ptr)); }
    std::vector<std::vector<Row>> rows(streams); std::vector<std::vector<std::array<float,1000>>> saved(streams);
    std::vector<std::exception_ptr> failures(streams); std::mutex mutex; std::condition_variable cv; int ready=0; bool start=false;
    std::vector<std::thread> workers;
    for (int s=0;s<streams;++s) workers.emplace_back([&,s] {
      try { rows[s].reserve(frames);saved[s].resize(std::min<int>(frames,images.size())); for(int f=0;f<warmup;++f) step(*sessions[s],images[f%images.size()],s,f,f%images.size()); }
      catch (...) { failures[s]=std::current_exception(); }
      { std::unique_lock<std::mutex> lock(mutex); ++ready;cv.notify_all();cv.wait(lock,[&]{return start;}); }
      if (failures[s]) return;
      try { for(int f=0;f<frames;++f) rows[s].push_back(step(*sessions[s],images[f%images.size()],s,f,f%images.size(),f<int(saved[s].size())?&saved[s][f]:nullptr)); }
      catch (...) { failures[s]=std::current_exception(); }
    });
    Clock::time_point begin;
    {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return ready==streams;});begin=Clock::now();start=true;cv.notify_all();}
    for(auto& worker:workers) worker.join();
    auto end=Clock::now();
    for(auto failure:failures)if(failure)std::rethrow_exception(failure);
    std::ofstream csv(prefix+".csv");csv.exceptions(std::ios::failbit|std::ios::badbit);csv<<std::setprecision(12)<<"stream,frame,image,preprocess_ms,runtime_ms,postprocess_ms,e2e_ms,top1,top2,top3,top4,top5\n";
    for(int s=0;s<streams;++s){for(const auto&r:rows[s]){csv<<r.stream<<','<<r.frame<<','<<r.image<<','<<r.preprocess<<','<<r.runtime<<','<<r.postprocess<<','<<r.e2e;for(int id:r.top5)csv<<','<<id;csv<<'\n';}
      for(size_t i=0;i<saved[s].size();++i){std::ofstream file(prefix+"-s"+std::to_string(s)+"-logits-"+std::to_string(i)+".bin",std::ios::binary);file.exceptions(std::ios::failbit|std::ios::badbit);file.write(reinterpret_cast<const char*>(saved[s][i].data()),sizeof(saved[s][i]));}}
    std::ofstream meta(prefix+".json");meta.exceptions(std::ios::failbit|std::ios::badbit);
    meta<<std::setprecision(12)<<"{\"pipeline_streams\":"<<streams<<",\"runtime_submission_threads\":"<<streams<<",\"frames_per_stream\":"<<frames<<",\"warmup_per_stream\":"<<warmup<<",\"opencv_threads\":"<<cv::getNumThreads()<<",\"completed_frames\":"<<streams*frames<<",\"wall_seconds\":"<<ms(begin,end)/1000<<",\"throughput_fps\":"<<streams*frames*1000/ms(begin,end)<<"}\n";
    std::cout<<"Completed "<<streams*frames<<" frames\n";
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
