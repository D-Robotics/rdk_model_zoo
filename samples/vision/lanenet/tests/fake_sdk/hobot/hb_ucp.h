#pragma once
#include <cstdint>
struct hbUCPSysMem {
  void *virAddr = nullptr;
};
using hbUCPTaskHandle_t = void *;
struct hbUCPSchedParam {
  unsigned long long backend = 0;
  int priority = 0;
};
#define HB_UCP_INITIALIZE_SCHED_PARAM(p) (*(p) = hbUCPSchedParam{})
#define HB_UCP_BPU_CORE_ANY (1ULL << 7)
#define HB_SYS_MEM_CACHE_CLEAN 1
#define HB_SYS_MEM_CACHE_INVALIDATE 2
int hbUCPMallocCached(hbUCPSysMem *, int, int);
int hbUCPFree(hbUCPSysMem *);
int hbUCPMemFlush(hbUCPSysMem *, int);
int hbUCPSubmitTask(hbUCPTaskHandle_t, hbUCPSchedParam *);
int hbUCPWaitTaskDone(hbUCPTaskHandle_t, int);
int hbUCPReleaseTask(hbUCPTaskHandle_t);
