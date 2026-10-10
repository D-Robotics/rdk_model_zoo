// Link-only SDK stubs for the native contract and CLI tests: the consolidated
// segment.cpp references these symbols, and those tests never call them.
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
int hbDNNInitializeFromFiles(hbDNNPackedHandle_t *out, const char **, int) {
  *out = nullptr;
  return 0;
}
int hbDNNGetModelNameList(const char ***names, int *count, hbDNNPackedHandle_t) {
  static const char *none[] = {nullptr};
  *names = none;
  *count = 0;
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *out, hbDNNPackedHandle_t, const char *) {
  *out = nullptr;
  return 0;
}
int hbDNNGetInputCount(int32_t *count, hbDNNHandle_t) {
  *count = 0;
  return 0;
}
int hbDNNGetOutputCount(int32_t *count, hbDNNHandle_t) {
  *count = 0;
  return 0;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t, int) {
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t, int) {
  return 0;
}
int hbDNNInferV2(hbUCPTaskHandle_t *task, hbDNNTensor *, hbDNNTensor *,
                 hbDNNHandle_t) {
  *task = nullptr;
  return 0;
}
int hbDNNRelease(hbDNNPackedHandle_t) { return 0; }
int hbUCPMallocCached(hbUCPSysMem *mem, int, int) {
  mem->virAddr = nullptr;
  return 0;
}
int hbUCPFree(hbUCPSysMem *mem) {
  mem->virAddr = nullptr;
  return 0;
}
int hbUCPMemFlush(hbUCPSysMem *, int) { return 0; }
int hbUCPSubmitTask(hbUCPTaskHandle_t, hbUCPSchedParam *) { return 0; }
int hbUCPWaitTaskDone(hbUCPTaskHandle_t, int) { return 0; }
int hbUCPReleaseTask(hbUCPTaskHandle_t) { return 0; }
