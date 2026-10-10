// Link-only SDK stubs for the host contract test: satisfy the fake ABI with
// unnamed parameters (no -Wunused-parameter) and never touch state; the
// contract executable only links them, it does not call them.
#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
int hbDNNInitializeFromFiles(hbPackedDNNHandle_t *, const char **, int) {
  return 0;
}
int hbDNNGetModelNameList(const char ***, int *, hbPackedDNNHandle_t) {
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *, hbPackedDNNHandle_t, const char *) {
  return 0;
}
int hbDNNGetInputCount(int *, hbDNNHandle_t) { return 0; }
int hbDNNGetOutputCount(int *, hbDNNHandle_t) { return 0; }
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t,
                                  int) {
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t,
                                   int) {
  return 0;
}
int hbDNNInfer(hbDNNTaskHandle_t *, hbDNNTensor **, hbDNNTensor *, hbDNNHandle_t,
               hbDNNInferCtrlParam *) {
  return 0;
}
int hbDNNWaitTaskDone(hbDNNTaskHandle_t, int) { return 0; }
int hbDNNReleaseTask(hbDNNTaskHandle_t) { return 0; }
int hbDNNRelease(hbPackedDNNHandle_t) { return 0; }
int hbSysAllocCachedMem(hbSysMem *, int) { return 0; }
int hbSysFreeMem(hbSysMem *) { return 0; }
int hbSysFlushMem(hbSysMem *, int) { return 0; }
