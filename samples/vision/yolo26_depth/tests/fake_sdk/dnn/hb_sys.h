#pragma once
// Host-only test ABI model. This is NOT a real SDK header or compatibility proof.
struct hbSysMem { void* virAddr=nullptr; };
constexpr int HB_SYS_MEM_CACHE_CLEAN=1,HB_SYS_MEM_CACHE_INVALIDATE=2;
int hbSysAllocCachedMem(hbSysMem*,int);
int hbSysFreeMem(hbSysMem*);
int hbSysFlushMem(hbSysMem*,int);
