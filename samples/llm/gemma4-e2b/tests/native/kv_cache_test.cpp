#include "gemma4_kv_cache.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
std::map<void *, int64_t> live;
int allocation = 0, fail_at = -1;
bool null_success = false;
const char *hbUCPGetErrorDesc(int) { return "fixture"; }
int hbUCPMallocCached(hbUCPSysMem *m, int64_t size, int) {
  ++allocation;
  if (allocation == fail_at && null_success)
    return 0;
  m->virAddr = std::malloc(size);
  assert(m->virAddr);
  live[m->virAddr] = size;
  return allocation == fail_at ? -1 : 0;
}
int hbUCPFree(hbUCPSysMem *m) {
  assert(live.erase(m->virAddr) == 1);
  std::free(m->virAddr);
  m->virAddr = nullptr;
  return 0;
}
template <class F> void rejects(F action) {
  bool caught = false;
  try {
    action();
  } catch (const std::exception &) {
    caught = true;
  }
  assert(caught);
}
int main(int argc, char **argv) {
  using namespace gemma4;
  assert(argc == 2);
  const std::string mode = argv[1];
  std::vector<int64_t> ks, vs;
  for (int head : kHeadDims) {
    ks.push_back(int64_t(kCacheLen) * head + 32);
    vs.push_back(int64_t(kCacheLen) * head + 16);
  }
  {
    KvCache cache;
    cache.Allocate(ks, vs);
    if (mode == "reset") {
      for (int i = 0; i < kNumKvLayers; ++i) {
        std::memset(cache.KLayer(i), 3, ks[i]);
        std::memset(cache.VLayer(i), 4, vs[i]);
      }
      cache.Reset();
      for (int i = 0; i < kNumKvLayers; ++i) {
        for (int64_t j = 0; j < ks[i]; ++j)
          assert(cache.KLayer(i)[j] == 0);
        for (int64_t j = 0; j < vs[i]; ++j)
          assert(cache.VLayer(i)[j] == 0);
      }
    } else if (mode == "transaction" || mode == "null") {
      auto *old = cache.KLayer(14);
      old[0] = 77;
      fail_at = allocation + 3;
      null_success = mode == "null";
      rejects([&] { cache.Allocate(ks, vs); });
      assert(live.size() == 30);
      assert(cache.KLayer(14) == old && old[0] == 77);
    } else if (mode == "invalid") {
      auto *old = cache.KLayer(0);
      rejects([&] { cache.Allocate({}, vs); });
      auto short_v = vs;
      short_v[14] = 1;
      rejects([&] { cache.Allocate(ks, short_v); });
      assert(cache.KLayer(0) == old && live.size() == 30);
      rejects([&] { cache.KLayer(-1); });
      rejects([&] { cache.VLayer(15); });
    } else {
      std::vector<std::vector<int8_t>> k(kNumKvLayers), v(kNumKvLayers);
      const int8_t *kp[kNumKvLayers];
      const int8_t *vp[kNumKvLayers];
      int64_t strides[kNumKvLayers];
      for (int i = 0; i < kNumKvLayers; ++i) {
        strides[i] = kHeadDims[i] + 8;
        k[i].assign(2 * strides[i], 0);
        v[i].assign(2 * strides[i], 0);
        for (int row = 0; row < 2; ++row)
          for (int col = 0; col < kHeadDims[i]; ++col) {
            k[i][row * strides[i] + col] = row + 1;
            v[i][row * strides[i] + col] = row + 11;
          }
        kp[i] = k[i].data();
        vp[i] = v[i].data();
      }
      cache.AppendPrefillChunk(kp, vp, strides, 0, 2);
      assert(cache.OccupiedLen() == 2 && cache.PhysicalIndex(0) == 4094);
      for (int i = 0; i < kNumKvLayers; ++i) {
        assert(cache.KLayer(i)[4094 * kHeadDims[i]] == 1);
        assert(cache.VLayer(i)[4095 * kHeadDims[i]] == 12);
        assert(cache.KLayer(i)[4096 * kHeadDims[i]] ==
               0); // aligned trailing padding isn't a KV row
      }
      if (mode == "append") {
        auto *saved = kp[14];
        kp[14] = nullptr;
        rejects([&] { cache.AppendDecodeStep(kp, vp, strides, 2); });
        kp[14] = saved;
        assert(cache.OccupiedLen() == 2 && cache.PhysicalIndex(0) == 4094);
        assert(cache.KLayer(0)[4094 * kHeadDims[0]] == 1);
        rejects([&] { cache.AppendPrefillChunk(kp, vp, strides, -1, 2); });
        rejects([&] { cache.AppendPrefillChunk(kp, vp, strides, 2, 257); });
        cache.AppendDecodeStep(kp, vp, strides, 2);
        assert(cache.OccupiedLen() == 3 && cache.PhysicalIndex(0) == 4093);
        cache.CompactShift(1, 2);
        assert(cache.OccupiedLen() == 1 && cache.PhysicalIndex(0) == 4095);
        assert(cache.KLayer(0)[4095 * kHeadDims[0]] == 1);
        cache.Reset();
        assert(cache.OccupiedLen() == 0 && cache.PhysicalIndex(0) == -1);
      } else {
        cache.Allocate(ks, vs);
        assert(cache.OccupiedLen() == 0 && cache.PhysicalIndex(0) == -1);
      }
    }
  }
  assert(live.empty());
  std::cout << mode << " passed\n";
}
