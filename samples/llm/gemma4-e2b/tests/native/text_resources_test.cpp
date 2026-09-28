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
int hbDNNGetModelHandle(hbDNNHandle_t *out, hbDNNPackedHandle_t p,
                        const char *) {
  *out = malformed == "handle" ? nullptr : p;
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
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                  int i) {
  *p = {};
  p->validShape.dimensionSize[0] = malformed == "sequence" ? 0 : 256;
  if (malformed == "rank")
    p->validShape.numDimensions = 0;
  p->alignedByteSize = i >= 5 ? 4096 * 512 : 16;
  return Tick();
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                   int) {
  *p = {};
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
    } catch (const std::runtime_error &) {
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
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    assert(rejected && buffers.empty() && models.empty());
  }
  std::cout << "6 invalid descriptor cases passed; " << calls
            << " constructor failure points + normal teardown passed\n";
}
