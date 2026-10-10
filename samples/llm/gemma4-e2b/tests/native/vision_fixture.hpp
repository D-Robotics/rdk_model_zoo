// Link satisfaction for host test binaries that compile the folded Vision
// engine source (gemma4_vision_engine.cpp: preprocessing, tensor contract and
// the SDK-driven engine in one translation unit). NOT vendor ABI, NOT a
// model: the tests exercise only the SDK-free stages, so these stubs merely
// satisfy the linker — every entry point returns an error and none is ever
// called. Single translation unit per test binary, following the same
// pattern as text_fixture.hpp.
#pragma once

#include "gemma4_vision_engine.hpp"

int hbDNNInitializeFromFiles(hbDNNPackedHandle_t *, const char **, int) {
  return -1;
}
int hbDNNRelease(hbDNNPackedHandle_t) { return 0; }
int hbDNNGetModelHandle(hbDNNHandle_t *, hbDNNPackedHandle_t, const char *) {
  return -1;
}
int hbDNNGetInputCount(int *, hbDNNHandle_t) { return -1; }
int hbDNNGetOutputCount(int *, hbDNNHandle_t) { return -1; }
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t,
                                  int) {
  return -1;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t,
                                   int) {
  return -1;
}
int hbDNNGetTaskOutputTensorProperties(hbDNNTensorProperties *, hbDNNHandle_t,
                                       int, int) {
  return -1;
}
const char *hbDNNGetErrorDesc(int) { return "fixture"; }
int hbDNNInferV2(hbUCPTaskHandle_t *, hbDNNTensor *, const hbDNNTensor *,
                 hbDNNHandle_t) {
  return -1;
}
int hbUCPMallocCached(hbUCPSysMem *, int64_t, int) { return -1; }
int hbUCPFree(hbUCPSysMem *) { return 0; }
int hbUCPMemFlush(hbUCPSysMem *, int) { return 0; }
int hbUCPSubmitTask(hbUCPTaskHandle_t, hbUCPSchedParam *) { return -1; }
int hbUCPWaitTaskDone(hbUCPTaskHandle_t, int) { return -1; }
int hbUCPReleaseTask(hbUCPTaskHandle_t) { return 0; }
const char *hbUCPGetErrorDesc(int) { return "fixture"; }
