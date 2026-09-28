#pragma once
#include <cstdint>
struct hbUCPSysMem {
  void *virAddr = nullptr;
};
using hbUCPTaskHandle_t = void *;
struct hbUCPSchedParam {
  uint64_t backend = 0;
};
#define HB_UCP_INITIALIZE_SCHED_PARAM(p) (*(p) = hbUCPSchedParam{})
#define HB_UCP_BPU_CORE_ANY 255
#define HB_UCP_BPU_CORE_0 1ULL
#define HB_SYS_MEM_CACHE_CLEAN 1
#define HB_SYS_MEM_CACHE_INVALIDATE 2
int hbUCPMallocCached(hbUCPSysMem *, int64_t, int);
int hbUCPFree(hbUCPSysMem *);
int hbUCPMemFlush(hbUCPSysMem *, int);
int hbUCPSubmitTask(hbUCPTaskHandle_t, hbUCPSchedParam *);
int hbUCPWaitTaskDone(hbUCPTaskHandle_t, int);
int hbUCPReleaseTask(hbUCPTaskHandle_t);
const char *hbUCPGetErrorDesc(int);
