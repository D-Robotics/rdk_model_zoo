#include "gemma4_kv_cache.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "hb_utils.h"

namespace gemma4 {

KvCache::KvCache() {
  k_mem_.resize(kNumKvLayers);
  v_mem_.resize(kNumKvLayers);
}

KvCache::~KvCache() {
  FreeMem();
}

void KvCache::FreeMem() {
  for (auto& m : k_mem_) {
    if (m.virAddr != nullptr) {
      hbUCPFree(&m);
      m.virAddr = nullptr;
    }
  }
  for (auto& m : v_mem_) {
    if (m.virAddr != nullptr) {
      hbUCPFree(&m);
      m.virAddr = nullptr;
    }
  }
  layer_bytes_.clear();
}

void KvCache::Allocate(const std::vector<int64_t>& k_bytes,
                       const std::vector<int64_t>& v_bytes) {
  FreeMem();
  k_mem_.resize(kNumKvLayers);
  v_mem_.resize(kNumKvLayers);
  layer_bytes_.resize(kNumKvLayers);
  for (int i = 0; i < kNumKvLayers; ++i) {
    layer_bytes_[i] = k_bytes[i];
    HBUCP_CHECK(hbUCPMallocCached(&k_mem_[i], k_bytes[i], 0),
                "alloc kv k cache");
    HBUCP_CHECK(hbUCPMallocCached(&v_mem_[i], v_bytes[i], 0),
                "alloc kv v cache");
    std::memset(k_mem_[i].virAddr, 0, static_cast<size_t>(k_bytes[i]));
    std::memset(v_mem_[i].virAddr, 0, static_cast<size_t>(v_bytes[i]));
  }
}

void KvCache::Reset() {
  // Zero out existing BPU buffers without freeing/reallocating
  for (int i = 0; i < kNumKvLayers; ++i) {
    if (k_mem_[i].virAddr != nullptr && !layer_bytes_.empty()) {
      std::memset(k_mem_[i].virAddr, 0, static_cast<size_t>(layer_bytes_[i]));
    }
    if (v_mem_[i].virAddr != nullptr && !layer_bytes_.empty()) {
      std::memset(v_mem_[i].virAddr, 0, static_cast<size_t>(layer_bytes_[i]));
    }
  }
  phys_of_global_.clear();
  cache_start_ = 0;
  occupied_len_ = 0;
}

void KvCache::ShiftPhysicalIndices(int delta) {
  for (int& phys : phys_of_global_) {
    if (phys >= 0) {
      phys += delta;
    }
  }
}

void KvCache::RollAppendLayer(int layer, const int8_t* k_src, const int8_t* v_src,
                              int rows, int64_t row_stride) {
  if (rows <= 0) {
    return;
  }
  const int hd = kHeadDims[layer];
  const int shift = rows * hd;
  int8_t* k_past = KLayer(layer);
  int8_t* v_past = VLayer(layer);
  const int keep = static_cast<int>(layer_bytes_[layer]) - shift;
  if (keep < 0) {
    throw std::runtime_error("KV roll exceeds cache size");
  }
  std::memmove(k_past, k_past + shift, static_cast<size_t>(keep));
  std::memmove(v_past, v_past + shift, static_cast<size_t>(keep));
  for (int t = 0; t < rows; ++t) {
    std::memcpy(k_past + keep + t * hd,
                k_src + static_cast<size_t>(t) * static_cast<size_t>(row_stride),
                static_cast<size_t>(hd));
    std::memcpy(v_past + keep + t * hd,
                v_src + static_cast<size_t>(t) * static_cast<size_t>(row_stride),
                static_cast<size_t>(hd));
  }
}

void KvCache::AppendPrefillChunk(const int8_t* const* k_outs,
                                 const int8_t* const* v_outs,
                                 const int64_t* row_strides, int chunk_start,
                                 int chunk_valid) {
  ShiftPhysicalIndices(-chunk_valid);
  for (int layer = 0; layer < kNumKvLayers; ++layer) {
    RollAppendLayer(layer, k_outs[layer], v_outs[layer], chunk_valid,
                    row_strides[layer]);
  }

  if (static_cast<int>(phys_of_global_.size()) < chunk_start + chunk_valid) {
    phys_of_global_.resize(static_cast<size_t>(chunk_start + chunk_valid), -1);
  }
  for (int t = 0; t < chunk_valid; ++t) {
    phys_of_global_[static_cast<size_t>(chunk_start + t)] =
        kCacheLen - chunk_valid + t;
  }

  occupied_len_ = std::max(occupied_len_, chunk_start + chunk_valid);
  if (occupied_len_ > kCacheLen) {
    cache_start_ = occupied_len_ - kCacheLen;
  }
}

void KvCache::AppendDecodeStep(const int8_t* const* k_outs,
                               const int8_t* const* v_outs,
                               const int64_t* row_strides, int global_pos) {
  ShiftPhysicalIndices(-1);
  for (int layer = 0; layer < kNumKvLayers; ++layer) {
    RollAppendLayer(layer, k_outs[layer], v_outs[layer], 1, row_strides[layer]);
  }

  if (static_cast<int>(phys_of_global_.size()) <= global_pos) {
    phys_of_global_.resize(static_cast<size_t>(global_pos + 1), -1);
  }
  phys_of_global_[static_cast<size_t>(global_pos)] = kCacheLen - 1;

  occupied_len_ = std::max(occupied_len_, global_pos + 1);
  if (occupied_len_ > kCacheLen) {
    cache_start_ = occupied_len_ - kCacheLen;
  }
}

int KvCache::PhysicalIndex(int global_pos) const {
  if (global_pos < 0 ||
      global_pos >= static_cast<int>(phys_of_global_.size())) {
    return -1;
  }
  return phys_of_global_[static_cast<size_t>(global_pos)];
}

void KvCache::CompactShift(int n_keep, int discard) {
  if (discard <= 0) return;

  // Truncate the KV cache to only keep the first n_keep tokens.
  // The caller will re-prefill the recent history with correct positions.
  // This is necessary because RoPE positions are baked into the int8 K cache.
  
  if (n_keep <= 0) {
    // Keep nothing
    phys_of_global_.clear();
    occupied_len_ = 0;
    cache_start_ = 0;
    return;
  }

  // Truncate phys_of_global_ to only keep [0, n_keep)
  if (static_cast<int>(phys_of_global_.size()) > n_keep) {
    phys_of_global_.resize(n_keep, -1);
  }
  occupied_len_ = n_keep;
  cache_start_ = 0;
  
  // Note: We don't zero out the physical buffer beyond n_keep because
  // it will be overwritten when the caller re-prefills the tail.
}

}  // namespace gemma4
