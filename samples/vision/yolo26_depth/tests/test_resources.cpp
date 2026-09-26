#include "model_runner.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <dnn/hb_dnn.h>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>
int fail_at = 0, step = 0, packed = 0, tasks = 0, wrong_shape = 0;
std::set<void *> buffers;
int rc() { return ++step == fail_at ? 7 : 0; }
int hbDNNInitializeFromFiles(void **out, const char **, int) {
  *out = &packed;
  ++packed;
  return rc();
}
int hbDNNGetModelNameList(const char ***out, int *count, void *) {
  static const char *names[] = {"fixture"};
  *out = names;
  *count = 1;
  return rc();
}
int hbDNNGetModelHandle(void **out, void *, const char *) {
  *out = &packed;
  return rc();
}
int hbDNNGetInputCount(int *count, void *) {
  *count = 1;
  return rc();
}
int hbDNNGetOutputCount(int *count, void *) {
  *count = 1;
  return rc();
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, void *, int) {
  p->tensorType = HB_DNN_IMG_TYPE_NV12;
  p->validShape = {4, {1, 3, 768, 768}};
  p->alignedShape = p->validShape;
  p->alignedByteSize = 768 * 768 * 3 / 2;
  return rc();
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, void *, int) {
  p->tensorType = HB_DNN_TENSOR_TYPE_F32;
  p->validShape = {4, {1, 192, 192, 1}};
  p->alignedShape = p->validShape;
  p->alignedByteSize = 192 * 192 * 4;
  if (wrong_shape)
    p->validShape.dimensionSize[3] = 2;
  return rc();
}
int hbSysAllocCachedMem(hbSysMem *p, int size) {
  int status = rc();
  if (status)
    return status;
  p->virAddr = std::calloc(1, size);
  buffers.insert(p->virAddr);
  return 0;
}
int hbSysFreeMem(hbSysMem *p) {
  assert(buffers.erase(p->virAddr) == 1);
  std::free(p->virAddr);
  p->virAddr = nullptr;
  return 0;
}
int hbSysFlushMem(hbSysMem *, int) { return rc(); }
int hbDNNInfer(void **task, hbDNNTensor **output, hbDNNTensor *, void *,
               hbDNNInferCtrlParam *) {
  *task = &tasks;
  ++tasks;
  const float value = 2;
  std::memcpy((*output)->sysMem[0].virAddr, &value, 4);
  return rc();
}
int hbDNNWaitTaskDone(void *, int) { return rc(); }
int hbDNNReleaseTask(void *) {
  assert(tasks == 1);
  --tasks;
  return 0;
}
int hbDNNRelease(void *) {
  assert(packed == 1);
  --packed;
  return 0;
}
int main(int argc, char **argv) {
  assert(argc == 2);
  std::ofstream(argv[1]) << "fixture";
  for (int failure = 0; failure <= 13; ++failure) {
    step = 0;
    fail_at = failure;
    bool threw = false;
    try {
      yolo26_depth::ModelRunner runner(argv[1], [] {});
      auto values = runner.run(std::vector<std::uint8_t>(768 * 768 * 3 / 2));
      assert(values[0] == 2);
    } catch (const std::exception &) {
      threw = true;
    }
    assert(threw == (failure != 0));
    assert(packed == 0 && tasks == 0 && buffers.empty());
  }
  fail_at = 0;
  step = 0;
  wrong_shape = 1;
  bool bad = false;
  try {
    yolo26_depth::ModelRunner runner(argv[1], [] {});
  } catch (const std::exception &) {
    bad = true;
  }
  assert(bad && buffers.empty() && packed == 0 && step == 7);
  step = 0;
  bad = false;
  try {
    yolo26_depth::ModelRunner runner(
        argv[1], [] { throw std::invalid_argument("wrong board"); });
  } catch (const std::exception &) {
    bad = true;
  }
  assert(bad && step == 0);
}
