// Baseline/green driver for the Text descriptor seq_len contract.
//
// The fixture reports a 512-position prefill export (chunk 256 is the fixed
// contract). Compiled against the pre-package sources, the constructor
// accepts it and FillCommonInputs writes 512 token ids into a 256-entry
// stack buffer (ASan stack-buffer-overflow). Compiled against the current
// sources, the constructor rejects the export before any allocation.
//
// Not a vendor SDK: host allocation double only, inference never runs.
#include "gemma4_text_engine.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

std::map<void *, int64_t> g_buffers;
std::map<const void *, int> g_models;

const char *hbDNNGetErrorDesc(int) { return "fixture dnn error"; }
const char *hbUCPGetErrorDesc(int) { return "fixture ucp error"; }

int hbDNNInitializeFromFiles(hbDNNPackedHandle_t *out, const char **, int) {
  static char model_slot = 0;
  *out = &model_slot;
  ++g_models[*out];
  return 0;
}
int hbDNNRelease(hbDNNPackedHandle_t handle) {
  assert(g_models.erase(handle) == 1);
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *out, hbDNNPackedHandle_t,
                        const char *name) {
  *out = static_cast<hbDNNHandle_t>(const_cast<char *>(name));
  return 0;
}
int hbDNNGetInputCount(int *count, hbDNNHandle_t) { *count = 35; return 0; }
int hbDNNGetOutputCount(int *count, hbDNNHandle_t) { *count = 31; return 0; }

// A 512-position export: every ordinary binding sized for seq 512.
hbDNNTensorProperties SeqProps(int type, int element, int seq, int64_t cols) {
  hbDNNTensorProperties properties{};
  properties.quantiType = NONE;
  properties.tensorType = type;
  properties.validShape.numDimensions = 2;
  properties.validShape.dimensionSize[0] = seq;
  properties.validShape.dimensionSize[1] = static_cast<int32_t>(cols);
  properties.stride[1] = element;
  properties.stride[0] = cols * element;
  properties.alignedByteSize = properties.stride[0] * seq;
  return properties;
}

int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t h,
                                  int i) {
  const int seq = static_cast<const char *>(h)[0] == 'p' ? 512 : 1;
  if (i == 0) *p = SeqProps(HB_DNN_TENSOR_TYPE_F32, 4, seq, 1536);
  else if (i == 1) *p = SeqProps(HB_DNN_TENSOR_TYPE_S64, 8, seq, seq);
  else if (i == 2) *p = SeqProps(HB_DNN_TENSOR_TYPE_S32, 4, seq, seq);
  else if (i <= 4) *p = SeqProps(HB_DNN_TENSOR_TYPE_S16, 2, seq, 4096);
  else {  // KV inputs: 512-position KV storage.
    const int layer = i < 20 ? i - 5 : i - 20;
    const int head = gemma4::kHeadDims[layer];
    *p = SeqProps(HB_DNN_TENSOR_TYPE_S8, 1, 512, static_cast<int64_t>(head) * 512);
  }
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t h,
                                   int j) {
  const int seq = static_cast<const char *>(h)[0] == 'p' ? 512 : 1;
  if (j == 0) *p = SeqProps(HB_DNN_TENSOR_TYPE_S16, 2, seq, 262144);
  else {
    const int layer = j < 16 ? j - 1 : j - 16;
    *p = SeqProps(HB_DNN_TENSOR_TYPE_S8, 1, seq, gemma4::kHeadDims[layer]);
  }
  return 0;
}
int hbDNNGetCompileBpuCoreNum(int32_t *count, hbDNNHandle_t) {
  *count = 1;
  return 0;
}
int hbUCPMallocCached(hbUCPSysMem *mem, int64_t bytes, int) {
  mem->virAddr = std::malloc(static_cast<size_t>(bytes));
  assert(mem->virAddr);
  g_buffers[mem->virAddr] = bytes;
  return 0;
}
int hbUCPFree(hbUCPSysMem *mem) {
  assert(g_buffers.erase(mem->virAddr) == 1);
  std::free(mem->virAddr);
  mem->virAddr = nullptr;
  return 0;
}
int hbUCPMemFlush(hbUCPSysMem *, int) { return 0; }
// The baseline must overflow before any inference call; if it is reached the
// driver reports it instead of simulating a model.
int hbDNNInferV2(hbUCPTaskHandle_t *task, hbDNNTensor *, const hbDNNTensor *,
                 hbDNNHandle_t) {
  *task = nullptr;
  std::printf("UNEXPECTED: inference reached\n");
  std::fflush(stdout);
  return -7;
}
int hbUCPSubmitTask(hbUCPTaskHandle_t, hbUCPSchedParam *) { return 0; }
int hbUCPWaitTaskDone(hbUCPTaskHandle_t, int) { return 0; }
int hbUCPReleaseTask(hbUCPTaskHandle_t) { return 0; }
int hbDNNGetTaskOutputTensorProperties(hbDNNTensorProperties *, hbUCPTaskHandle_t,
                                       int, int) {
  return 0;
}

namespace gemma4 {
TokenEmbeddings::TokenEmbeddings(const std::string &) {}
void TokenEmbeddings::Lookup(const std::vector<int64_t> &ids, float *out) const {
  for (size_t i = 0; i < ids.size(); ++i)
    for (int c = 0; c < kHiddenSize; ++c)
      out[i * kHiddenSize + c] = static_cast<float>(ids[i]);
}
const float *TokenEmbeddings::GetRow(int64_t) const { std::abort(); }
std::vector<float>
TokenEmbeddings::BuildPromptHidden(const std::vector<int64_t> &,
                                   const std::vector<float> &) const {
  std::abort();
}
}  // namespace gemma4

int main() {
  try {
    gemma4::TextEngine engine("fixture.hbm", "unused");
    std::printf("constructor accepted the 512-position export\n");
    std::fflush(stdout);
    const auto out = engine.Generate({11, 22, 33, 44, 55}, 1);
    std::printf("UNEXPECTED: generation returned %zu tokens\n", out.size());
    return 1;
  } catch (const std::exception &error) {
    std::printf("rejected at construction: %s\n", error.what());
    return 0;
  }
}
