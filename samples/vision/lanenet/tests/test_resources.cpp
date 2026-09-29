#include "hobot/dnn/hb_dnn.h"
#include "model_runner.hpp"
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
int calls = 0, fail = 0, models = 0, buffers = 0, tasks = 0;
bool bad_metadata = false;
int step() { return ++calls == fail ? -1 : 0; }
int hbDNNInitializeFromFiles(void **p, const char **, int) {
  *p = new int(1);
  ++models;
  return step();
}
int hbDNNGetModelNameList(const char ***n, int *c, void *) {
  static const char *names[] = {"lane"};
  *n = names;
  *c = 1;
  return step();
}
int hbDNNGetModelHandle(void **p, void *model, const char *) {
  *p = model;
  return step();
}
int hbDNNGetInputCount(int32_t *c, void *) {
  *c = 1;
  return step();
}
int hbDNNGetOutputCount(int32_t *c, void *) {
  *c = 3;
  return step();
}
void shape(hbDNNTensorProperties *p, int channels, int type, int bytes) {
  *p = hbDNNTensorProperties{};
  p->validShape.numDimensions = 4;
  int dims[] = {1, channels, 256, 512};
  std::int64_t stride = bytes;
  for (int i = 3; i >= 0; --i) {
    p->validShape.dimensionSize[i] = dims[i];
    p->stride[i] = stride;
    stride *= dims[i];
  }
  p->alignedByteSize = stride;
  p->tensorType = type;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, void *, int) {
  shape(p, 3, HB_DNN_TENSOR_TYPE_F32, 4);
  for (int i = 0; i < 4; i++)
    p->stride[i] = -1;
  return step();
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, void *, int i) {
  if (i == 0)
    shape(p, 1, HB_DNN_TENSOR_TYPE_S64, 8);
  else
    shape(p, i == 1 ? 2 : 3, HB_DNN_TENSOR_TYPE_F32, 4);
  if (bad_metadata && i == 2)
    p->validShape.dimensionSize[3] = 511;
  return step();
}
int hbDNNRelease(void *p) {
  delete static_cast<int *>(p);
  --models;
  return 0;
}
int hbUCPMallocCached(hbUCPSysMem *p, int n, int) {
  p->virAddr = std::calloc(1, n);
  ++buffers;
  return step();
}
int hbUCPFree(hbUCPSysMem *p) {
  assert(p->virAddr);
  std::free(p->virAddr);
  p->virAddr = nullptr;
  --buffers;
  return 0;
}
int hbUCPMemFlush(hbUCPSysMem *, int) { return step(); }
int hbDNNInferV2(void **p, hbDNNTensor *, hbDNNTensor *, void *) {
  *p = new int(2);
  ++tasks;
  return step();
}
int hbUCPSubmitTask(void *, hbUCPSchedParam *p) {
  assert(p->backend == HB_UCP_BPU_CORE_ANY);
  return step();
}
int hbUCPWaitTaskDone(void *, int) { return step(); }
int hbUCPReleaseTask(void *p) {
  delete static_cast<int *>(p);
  --tasks;
  return 0;
}
int main(int argc, char **argv) {
  assert(argc == 2);
  std::ofstream(argv[1]) << "fixture";
  for (int point = 0; point <= 20; point++) {
    calls = 0;
    fail = point;
    bool threw = false;
    try {
      lanenet::ModelRunner runner(argv[1], [] {});
      auto raw = runner.run(std::vector<float>(3 * 256 * 512, .5f));
      assert(raw.size() == 3);
      auto result = lanenet::decode_outputs(raw);
      assert(result.binary[0] == 0);
    } catch (const std::exception &) {
      threw = true;
    }
    assert(threw == (point != 0));
    assert(models == 0 && buffers == 0 && tasks == 0);
  }
  fail = 0;
  calls = 0;
  bool rejected = false;
  try {
    lanenet::ModelRunner runner(
        argv[1], [] { throw std::invalid_argument("wrong board"); });
  } catch (const std::exception &) {
    rejected = true;
  }
  assert(rejected && calls == 0);
  bad_metadata = true;
  rejected = false;
  try {
    lanenet::ModelRunner runner(argv[1], [] {});
  } catch (const std::exception &) {
    rejected = true;
  }
  assert(rejected && models == 0 && buffers == 0 && tasks == 0);
}
