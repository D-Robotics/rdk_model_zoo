// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Minimal host test double based on fields used by the source adapter.
#include <cstdint>
struct hbUCPSysMem { void* virAddr=nullptr; };
using hbUCPTaskHandle_t=void*;
struct hbUCPSchedParam { unsigned long long backend=0; int priority=0; };
#define HB_UCP_INITIALIZE_SCHED_PARAM(p) (*(p)=hbUCPSchedParam{})
#define HB_SYS_MEM_CACHE_CLEAN 0
#define HB_SYS_MEM_CACHE_INVALIDATE 1
int hbUCPMallocCached(hbUCPSysMem*,int,int);
int hbUCPFree(hbUCPSysMem*);
int hbUCPMemFlush(hbUCPSysMem*,int);
int hbUCPSubmitTask(hbUCPTaskHandle_t,hbUCPSchedParam*);
int hbUCPWaitTaskDone(hbUCPTaskHandle_t,int);
int hbUCPReleaseTask(hbUCPTaskHandle_t);
