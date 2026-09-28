// Independent host doubles: no vendor SDK or model execution.
#include "gemma4_vision_engine.hpp"
#include "hb_utils.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
std::string fail;
std::set<void *> buffers, tasks, models;
int allocations = 0, releases = 0, clean = 0, invalidate = 0;
int error(const char *name) { return fail == name ? -1 : 0; }
const char *hbDNNGetErrorDesc(int) { return "fixture"; }
const char *hbUCPGetErrorDesc(int) { return "fixture"; }
int hbUCPMallocCached(hbUCPSysMem *m, int64_t n, int) {
  ++allocations;
  if (fail == "alloc_null")
    return 0;
  m->virAddr = std::malloc(n);
  buffers.insert(m->virAddr);
  return error("alloc") || (fail == "second_alloc" && allocations == 2) ? -1
                                                                        : 0;
}
int hbUCPFree(hbUCPSysMem *m) {
  assert(buffers.erase(m->virAddr) == 1);
  std::free(m->virAddr);
  m->virAddr = nullptr;
  return 0;
}
int hbUCPMemFlush(hbUCPSysMem *, int mode) {
  if (mode == HB_SYS_MEM_CACHE_CLEAN) {
    ++clean;
    return error("clean");
  }
  ++invalidate;
  return error("invalidate");
}
int hbUCPSubmitTask(hbUCPTaskHandle_t, hbUCPSchedParam *) {
  return error("submit");
}
int hbUCPWaitTaskDone(hbUCPTaskHandle_t, int) { return error("wait"); }
int hbUCPReleaseTask(hbUCPTaskHandle_t t) {
  ++releases;
  assert(tasks.erase(t) == 1);
  delete static_cast<int *>(t);
  return error("release");
}
int hbDNNInferV2(hbUCPTaskHandle_t *t, hbDNNTensor *, const hbDNNTensor *,
                 hbDNNHandle_t) {
  if (fail == "infer_null")
    return 0;
  *t = new int;
  tasks.insert(*t);
  return error("infer");
}
int hbDNNGetCompileBpuCoreNum(int32_t *n, hbDNNHandle_t) {
  *n = 2;
  return error("cores");
}
int hbDNNGetTaskOutputTensorProperties(hbDNNTensorProperties *,
                                       hbUCPTaskHandle_t, int, int) {
  return error("task_props");
}
int hbDNNInitializeFromFiles(hbDNNPackedHandle_t *p, const char **, int) {
  if (fail == "model_null")
    return 0;
  *p = new int;
  models.insert(*p);
  return error("load");
}
int hbDNNRelease(hbDNNPackedHandle_t p) {
  assert(models.erase(p) == 1);
  delete static_cast<int *>(p);
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *h, hbDNNPackedHandle_t p, const char *) {
  if (fail != "handle_null")
    *h = p;
  return error("handle");
}
int hbDNNGetInputCount(int *n, hbDNNHandle_t) {
  *n = 1;
  return error("input_count");
}
int hbDNNGetOutputCount(int *n, hbDNNHandle_t) {
  *n = 1;
  return error("output_count");
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                  int) {
  *p = {};
  if (fail == "zero_bytes")
    p->alignedByteSize = 0;
  return error("input_props");
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                   int) {
  *p = {};
  return error("output_props");
}

int main(int argc, char **argv) {
  assert(argc == 3);
  std::string action = argv[1];
  fail = argv[2];
  bool threw = false;
  try {
    if (action == "tensor") {
      auto t = MakeTensor(nullptr, true, 0);
      std::vector<hbDNNTensor> ts{t};
      FreeTensors(ts);
    } else if (action == "constructor") {
      gemma4::VisionEngine engine("fixture.hbm");
    } else {
      std::vector<hbDNNTensor> inputs(2), outputs(2);
      if (action == "all")
        RunInfer(nullptr, inputs, outputs);
      else
        RunInferSelective(nullptr, inputs, outputs, {1}, {0});
    }
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw == (fail != "none"));
  assert(buffers.empty());
  assert(tasks.empty());
  assert(models.empty());
  if ((action == "all" || action == "selective") && fail == "none") {
    assert(releases == 1);
    assert(clean == (action == "all" ? 2 : 1));
    assert(invalidate == (action == "all" ? 2 : 1));
  }
  std::cout << action << "/" << fail << " passed\n";
}
