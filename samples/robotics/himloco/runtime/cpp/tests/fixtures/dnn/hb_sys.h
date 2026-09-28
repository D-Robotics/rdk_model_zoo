#pragma once
#include <cstdint>
struct hbSysMem {
  void *virAddr = nullptr;
};
enum { HB_SYS_MEM_CACHE_CLEAN = 1, HB_SYS_MEM_CACHE_INVALIDATE = 2 };
int hbSysAllocCachedMem(hbSysMem *, std::uint32_t);
int hbSysFreeMem(hbSysMem *);
int hbSysFlushMem(hbSysMem *, int);
