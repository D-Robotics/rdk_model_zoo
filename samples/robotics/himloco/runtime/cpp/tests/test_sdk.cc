// Explicit link-time SDK and preflight doubles; never linked into production.
#include "dnn/hb_dnn.h"
#include "sdk_runner.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int packs = 0, allocations = 0, tasks = 0, calls = 0, gates = 0,
    alloc_calls = 0;
int fail_alloc = 0, infer_error = 0, wait_error = 0, flush_error = 0;
bool deny = false, bad_output = false;
hbDNNTensorProperties input, output;
void Reset() {
  assert(packs == 0 && allocations == 0 && tasks == 0);
  input = {};
  output = {};
  input.validShape.dimensionSize[0] = 1;
  input.validShape.dimensionSize[1] = 1;
  input.validShape.dimensionSize[2] = 1;
  input.validShape.dimensionSize[3] = 270;
  input.alignedShape = input.validShape;
  input.alignedShape.dimensionSize[3] = 272;
  input.alignedByteSize = 272 * 4;
  output.validShape.dimensionSize[0] = 1;
  output.validShape.dimensionSize[1] = 1;
  output.validShape.dimensionSize[2] = 3;
  output.validShape.dimensionSize[3] = 4;
  output.alignedShape = output.validShape;
  output.alignedShape.dimensionSize[3] = 8;
  output.alignedByteSize = 3 * 8 * 4;
  alloc_calls = 0;
  fail_alloc = 0;
  infer_error = 0;
  wait_error = 0;
  flush_error = 0;
  deny = false;
  bad_output = false;
}
template <class F> void Fails(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::exception &) {
    failed = true;
  }
  assert(failed);
}
} // namespace
namespace himloco {
void verify_native_model(const std::string &) {
  ++gates;
  if (deny)
    throw std::runtime_error("gate denied");
}
} // namespace himloco
int hbDNNInitializeFromFiles(hbPackedDNNHandle_t *p, const char **, int) {
  assert(gates > 0);
  *p = &packs;
  ++packs;
  return 0;
}
int hbDNNRelease(hbPackedDNNHandle_t) {
  --packs;
  return 0;
}
int hbDNNGetModelNameList(const char ***p, int *n, hbPackedDNNHandle_t) {
  static const char *names[] = {"policy"};
  *p = names;
  *n = 1;
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *p, hbPackedDNNHandle_t, const char *) {
  *p = &packs;
  return 0;
}
int hbDNNGetInputCount(int *n, hbDNNHandle_t) {
  *n = 1;
  return 0;
}
int hbDNNGetOutputCount(int *n, hbDNNHandle_t) {
  *n = 1;
  return 0;
}
int hbDNNGetInputName(const char **p, hbDNNHandle_t, int) {
  *p = "obs_history";
  return 0;
}
int hbDNNGetOutputName(const char **p, hbDNNHandle_t, int) {
  *p = "actions";
  return 0;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                  int) {
  *p = input;
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                   int) {
  *p = output;
  return 0;
}
const char *hbDNNGetVersion() { return "fixture-only"; }
int hbSysAllocCachedMem(hbSysMem *p, std::uint32_t size) {
  ++alloc_calls;
  if (alloc_calls == fail_alloc)
    return -1;
  p->virAddr = std::malloc(size);
  ++allocations;
  return 0;
}
int hbSysFreeMem(hbSysMem *p) {
  std::free(p->virAddr);
  p->virAddr = nullptr;
  --allocations;
  return 0;
}
int hbSysFlushMem(hbSysMem *, int) { return flush_error; }
int hbDNNInfer(hbDNNTaskHandle_t *p, hbDNNTensor **out, hbDNNTensor *in,
               hbDNNHandle_t, hbDNNInferCtrlParam *ctl) {
  *p = &tasks;
  ++tasks;
  ++calls;
  assert(ctl->priority == 7);
  assert(in->properties.alignedShape.dimensionSize[3] == 270);
  const float *values = static_cast<float *>(in->sysMem[0].virAddr);
  assert(values[0] == 2 && values[269] == 2 && values[270] == 0);
  auto *result = static_cast<float *>((*out)->sysMem[0].virAddr);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 8; ++c)
      result[r * 8 + c] = c < 4 ? float(r * 4 + c) : 999;
  if (bad_output)
    result[0] = std::numeric_limits<float>::quiet_NaN();
  return infer_error;
}
int hbDNNWaitTaskDone(hbDNNTaskHandle_t, int) { return wait_error; }
int hbDNNReleaseTask(hbDNNTaskHandle_t) {
  --tasks;
  return 0;
}
int main() {
  Reset();
  deny = true;
  Fails([] { himloco::SdkRunner r({"fixture.bin", 7}); });
  assert(!packs && !allocations);
  Reset();
  input.alignedByteSize = 4;
  Fails([] { himloco::SdkRunner r({"fixture.bin", 7}); });
  assert(!packs && !allocations);
  Reset();
  output.alignedShape.dimensionSize[3] = 2;
  Fails([] { himloco::SdkRunner r({"fixture.bin", 7}); });
  assert(!packs && !allocations);
  Reset();
  output.quantiType = 1;
  Fails([] { himloco::SdkRunner r({"fixture.bin", 7}); });
  assert(!packs && !allocations);
  Reset();
  output.tensorType = 99;
  Fails([] { himloco::SdkRunner r({"fixture.bin", 7}); });
  assert(!packs && !allocations);
  Reset();
  output.alignedShape.dimensionSize[0] = std::numeric_limits<int>::max();
  output.alignedShape.dimensionSize[1] = std::numeric_limits<int>::max();
  output.alignedShape.dimensionSize[2] = std::numeric_limits<int>::max();
  Fails([] { himloco::SdkRunner r({"fixture.bin", 7}); });
  assert(!packs && !allocations);
  Reset();
  fail_alloc = 1;
  Fails([] { himloco::SdkRunner r({"fixture.bin", 7}); });
  assert(!packs && !allocations);
  Reset();
  fail_alloc = 2;
  Fails([] { himloco::SdkRunner r({"fixture.bin", 7}); });
  assert(!packs && !allocations);
  Reset();
  {
    himloco::SdkRunner runner({"fixture.bin", 7});
    assert(packs == 1 && allocations == 2);
    himloco::HimLoco task(
        [&](const std::vector<float> &v) { return runner.run(v); });
    auto result = task.predict(std::vector<float>(270, 2));
    for (int i = 0; i < 12; ++i)
      assert(result.actions[i] == i);
    assert(result.latency_ms >= 0 && tasks == 0 &&
           runner.model_name() == "policy");
    assert(runner.input_metadata().aligned_shape[3] == 272);
    int before = calls;
    Fails([&] { runner.run({1}); });
    assert(calls == before);
    infer_error = -7;
    Fails([&] { runner.run(std::vector<float>(270, 2)); });
    assert(tasks == 0);
    infer_error = 0;
    wait_error = -8;
    Fails([&] { runner.run(std::vector<float>(270, 2)); });
    assert(tasks == 0);
    wait_error = 0;
    flush_error = -9;
    Fails([&] { runner.run(std::vector<float>(270, 2)); });
    assert(tasks == 0);
    flush_error = 0;
    bad_output = true;
    Fails([&] { runner.run(std::vector<float>(270, 2)); });
    assert(tasks == 0);
    assert(result.actions[0] == 0);
  }
  assert(!packs && !allocations && !tasks);
  std::cout << "SDK fixture: metadata/capacity, padded output, scheduling, "
               "errors and RAII passed\n";
}
