// Constructor-only SDK and embedding doubles: never loads weights or infers.
#include "gemma4_text_engine.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

static std::set<void *> buffers, models;
static int step = 0, fail_at = 0;
static std::string malformed;
static int Tick() { return ++step == fail_at ? -1 : 0; }
const char *hbDNNGetErrorDesc(int) { return "injected constructor failure"; }
const char *hbUCPGetErrorDesc(int) { return "injected allocation failure"; }
int hbUCPMallocCached(hbUCPSysMem *mem, int64_t n, int) {
  mem->virAddr = std::malloc(n);
  assert(mem->virAddr);
  buffers.insert(mem->virAddr);
  return Tick(); // Include allocation that returns both memory and an error.
}
int hbUCPFree(hbUCPSysMem *mem) {
  assert(buffers.erase(mem->virAddr) == 1);
  std::free(mem->virAddr);
  mem->virAddr = nullptr;
  return 0;
}
int hbDNNInitializeFromFiles(hbDNNPackedHandle_t *out, const char **, int) {
  if (malformed == "packed") {
    *out = nullptr;
    return 0;
  }
  *out = new int(0);
  models.insert(*out);
  return Tick();
}
int hbDNNRelease(hbDNNPackedHandle_t handle) {
  assert(models.erase(handle) == 1);
  delete static_cast<int *>(handle);
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *out, hbDNNPackedHandle_t,
                        const char *name) {
  // Subgraph handles borrow the name as a stable identity; only the packed
  // model is released.
  *out = malformed == "handle" ? nullptr
                               : static_cast<hbDNNHandle_t>(const_cast<char *>(name));
  return Tick();
}
int hbDNNGetInputCount(int *n, hbDNNHandle_t) {
  *n = malformed == "inputs" ? 0 : 35;
  return Tick();
}
int hbDNNGetOutputCount(int *n, hbDNNHandle_t) {
  *n = malformed == "outputs" ? -1 : 31;
  return Tick();
}
// Contract-valid fixed-export descriptors; the malformed modes degrade one
// field so the constructor must reject without leaking.
int SubgraphSeq(hbDNNHandle_t handle) {
  const char *name = static_cast<const char *>(handle);
  return name != nullptr && name[0] == 'p' ? gemma4::kChunkSize : 1;
}
int KvLayer(int index) {
  return index < 5 + gemma4::kNumKvLayers
             ? index - 5
             : index - 5 - gemma4::kNumKvLayers;
}
hbDNNTensorProperties InputProps(hbDNNHandle_t handle, int i) {
  const int seq = SubgraphSeq(handle);
  hbDNNTensorProperties p{};
  p.quantiType = NONE;
  if (i == 0) {  // inputs_embeds F32 [seq, hidden]
    p.tensorType = HB_DNN_TENSOR_TYPE_F32;
    p.validShape.numDimensions = 2;
    p.validShape.dimensionSize[0] = seq;
    p.validShape.dimensionSize[1] = gemma4::kHiddenSize;
    p.stride[1] = 4;
  } else if (i == 1) {  // token ids S64 [1, seq]
    p.tensorType = HB_DNN_TENSOR_TYPE_S64;
    p.validShape.numDimensions = 2;
    p.validShape.dimensionSize[0] = 1;
    p.validShape.dimensionSize[1] = seq;
    p.stride[1] = 8;
  } else if (i == 2) {  // position ids S32 [seq]
    p.tensorType = HB_DNN_TENSOR_TYPE_S32;
    p.validShape.numDimensions = 1;
    p.validShape.dimensionSize[0] = seq;
    p.stride[0] = 4;
    p.alignedByteSize = static_cast<int64_t>(seq) * 4;
    return p;
  } else if (i <= 4) {  // masks S16 [seq, cache_len]
    p.tensorType = HB_DNN_TENSOR_TYPE_S16;
    p.validShape.numDimensions = 2;
    p.validShape.dimensionSize[0] = seq;
    p.validShape.dimensionSize[1] = gemma4::kCacheLen;
    p.stride[1] = 2;
  } else {  // KV cache inputs S8 [cache_len, 1, head_dim]
    const int head = gemma4::kHeadDims[KvLayer(i)];
    p.tensorType = HB_DNN_TENSOR_TYPE_S8;
    p.validShape.numDimensions = 3;
    p.validShape.dimensionSize[0] = gemma4::kCacheLen;
    p.validShape.dimensionSize[1] = 1;
    p.validShape.dimensionSize[2] = head;
    p.stride[2] = 1;
    p.stride[1] = head;
    p.stride[0] = head;
    p.alignedByteSize = static_cast<int64_t>(head) * gemma4::kCacheLen;
    return p;
  }
  p.stride[0] = static_cast<int64_t>(p.validShape.dimensionSize[1]) *
                p.stride[1];
  p.alignedByteSize =
      p.stride[0] * p.validShape.dimensionSize[0];
  return p;
}
hbDNNTensorProperties OutputProps(hbDNNHandle_t handle, int j) {
  const int seq = SubgraphSeq(handle);
  hbDNNTensorProperties p{};
  p.quantiType = NONE;
  if (j == 0) {  // logits S16 [seq, vocab]
    p.tensorType = HB_DNN_TENSOR_TYPE_S16;
    p.validShape.numDimensions = 2;
    p.validShape.dimensionSize[0] = seq;
    p.validShape.dimensionSize[1] = gemma4::kVocabSize;
    p.stride[1] = 2;
  } else {  // KV outputs S8 [seq, head_dim]
    const int layer =
        j < 1 + gemma4::kNumKvLayers ? j - 1 : j - 1 - gemma4::kNumKvLayers;
    p.tensorType = HB_DNN_TENSOR_TYPE_S8;
    p.validShape.numDimensions = 2;
    p.validShape.dimensionSize[0] = seq;
    p.validShape.dimensionSize[1] = gemma4::kHeadDims[layer];
    p.stride[1] = 1;
  }
  p.stride[0] =
      static_cast<int64_t>(p.validShape.dimensionSize[1]) * p.stride[1];
  p.alignedByteSize = p.stride[0] * p.validShape.dimensionSize[0];
  return p;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t h,
                                  int i) {
  *p = InputProps(h, i);
  if (malformed == "sequence")
    p->validShape.dimensionSize[0] = 0;
  if (malformed == "rank")
    p->validShape.numDimensions = 0;
  return Tick();
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t h,
                                   int j) {
  *p = OutputProps(h, j);
  if (malformed == "sequence")
    p->validShape.dimensionSize[0] = 0;
  if (malformed == "rank")
    p->validShape.numDimensions = 0;
  return Tick();
}
// Any inference call in this constructor test is a test failure.
int hbUCPMemFlush(hbUCPSysMem *, int) { std::abort(); }
int hbUCPSubmitTask(hbUCPTaskHandle_t, hbUCPSchedParam *) { std::abort(); }
int hbUCPWaitTaskDone(hbUCPTaskHandle_t, int) { std::abort(); }
int hbUCPReleaseTask(hbUCPTaskHandle_t) { std::abort(); }
int hbDNNInferV2(hbUCPTaskHandle_t *, hbDNNTensor *, const hbDNNTensor *,
                 hbDNNHandle_t) {
  std::abort();
}
int hbDNNGetTaskOutputTensorProperties(hbDNNTensorProperties *,
                                       hbUCPTaskHandle_t, int, int) {
  std::abort();
}
int hbDNNGetCompileBpuCoreNum(int32_t *, hbDNNHandle_t) { std::abort(); }
namespace gemma4 {
TokenEmbeddings::TokenEmbeddings(const std::string &) {}
void TokenEmbeddings::Lookup(const std::vector<int64_t> &, float *) const {
  std::abort();
}
const float *TokenEmbeddings::GetRow(int64_t) const { std::abort(); }
std::vector<float>
TokenEmbeddings::BuildPromptHidden(const std::vector<int64_t> &,
                                   const std::vector<float> &) const {
  std::abort();
}
} // namespace gemma4
int main() {
  {
    gemma4::TextEngine engine("fixture", "fixture");
  }
  assert(buffers.empty() && models.empty());
  const int calls = step;
  for (fail_at = 1; fail_at <= calls; ++fail_at) {
    step = 0;
    bool rejected = false;
    try {
      gemma4::TextEngine engine("fixture", "fixture");
    } catch (const std::exception &) {
      rejected = true;
    }
    assert(rejected);
    assert(buffers.empty() && models.empty());
  }
  fail_at = 0;
  for (const auto *mode :
       {"packed", "handle", "inputs", "outputs", "sequence", "rank"}) {
    malformed = mode;
    bool rejected = false;
    try {
      gemma4::TextEngine engine("fixture", "fixture");
    } catch (const std::exception &) {
      rejected = true;
    }
    assert(rejected && buffers.empty() && models.empty());
  }
  std::cout << "6 invalid descriptor cases passed; " << calls
            << " constructor failure points + normal teardown passed\n";
}
