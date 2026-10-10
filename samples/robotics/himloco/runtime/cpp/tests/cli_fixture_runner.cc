// Host CLI fixture: link-time doubles for the declaration-only D-Robotics DNN
// declarations in tests/fixtures, plus an inert board gate via
// -DHIMLOCO_HOST_FIXTURE. The fixture binary therefore runs the production
// SdkRuntime, run workspace and CLI end to end without a board or model file.
#include "dnn/hb_dnn.h"
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
int runs = 0;
char task_slot; // Non-null task handle stand-in.

hbDNNTensorProperties InputProperties() {
  hbDNNTensorProperties p{};
  p.validShape.dimensionSize[0] = 1;
  p.validShape.dimensionSize[1] = 1;
  p.validShape.dimensionSize[2] = 1;
  p.validShape.dimensionSize[3] = 270;
  p.alignedShape = p.validShape;
  p.alignedShape.dimensionSize[3] = 272;
  p.alignedByteSize = 272 * 4;
  return p;
}
hbDNNTensorProperties OutputProperties() {
  hbDNNTensorProperties p{};
  p.validShape.dimensionSize[0] = 1;
  p.validShape.dimensionSize[1] = 1;
  p.validShape.dimensionSize[2] = 1;
  p.validShape.dimensionSize[3] = 12;
  p.alignedShape = p.validShape;
  p.alignedShape.dimensionSize[3] = 16;
  p.alignedByteSize = 16 * 4;
  return p;
}
} // namespace

int hbDNNInitializeFromFiles(hbPackedDNNHandle_t *p, const char **, int) {
  *p = &task_slot;
  return 0;
}
int hbDNNRelease(hbPackedDNNHandle_t) { return 0; }
int hbDNNGetModelNameList(const char ***p, int *n, hbPackedDNNHandle_t) {
  static const char *names[] = {"fixture"};
  *p = names;
  *n = 1;
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *p, hbPackedDNNHandle_t, const char *) {
  *p = &task_slot;
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
  *p = InputProperties();
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                   int) {
  *p = OutputProperties();
  return 0;
}
const char *hbDNNGetVersion() { return "fixture-only"; }
int hbSysAllocCachedMem(hbSysMem *p, std::uint32_t size) {
  p->virAddr = std::malloc(size);
  return 0;
}
int hbSysFreeMem(hbSysMem *p) {
  std::free(p->virAddr);
  p->virAddr = nullptr;
  return 0;
}
int hbSysFlushMem(hbSysMem *, int) { return 0; }
int hbDNNInfer(hbDNNTaskHandle_t *p, hbDNNTensor **out, hbDNNTensor *,
               hbDNNHandle_t, hbDNNInferCtrlParam *) {
  if (const char *fail = std::getenv("HIMLOCO_FIXTURE_FAIL_AFTER"))
    if (runs >= std::stoi(fail))
      throw std::runtime_error("Injected fixture failure");
  ++runs;
  *p = &task_slot;
  // Valid region holds iota(12); aligned padding stays untouched.
  constexpr int kFixtureActions = 12;
  auto *result = static_cast<float *>((*out)->sysMem[0].virAddr);
  std::vector<float> values(kFixtureActions);
  std::iota(values.begin(), values.end(), 0.f);
  for (std::size_t i = 0; i < values.size(); ++i)
    result[i] = values[i];
  return 0;
}
int hbDNNWaitTaskDone(hbDNNTaskHandle_t, int) { return 0; }
int hbDNNReleaseTask(hbDNNTaskHandle_t) { return 0; }
