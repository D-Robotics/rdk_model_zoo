/**
 * @file gemma4_kv_cache.cpp
 * @brief Manage Gemma4-E2B BPU key-value cache storage and compaction.
 *
 * The implementation allocates cache tensors, appends prefill/decode results,
 * and shifts retained tokens when conversation history is truncated.
 *
 * @note KvCache instances are not thread-safe.
 */

#include "gemma4_kv_cache.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "hb_utils.hpp"

namespace gemma4 {

KvCache::KvCache() {
  k_mem_.resize(kNumKvLayers);
  v_mem_.resize(kNumKvLayers);
}

KvCache::~KvCache() { FreeMem(); }

void KvCache::FreeMem() noexcept {
  for (auto &m : k_mem_) {
    if (m.virAddr != nullptr) {
      hbUCPFree(&m);
      m.virAddr = nullptr;
    }
  }
  for (auto &m : v_mem_) {
    if (m.virAddr != nullptr) {
      hbUCPFree(&m);
      m.virAddr = nullptr;
    }
  }
  k_bytes_.clear();
  v_bytes_.clear();
  phys_of_global_.clear();
  cache_start_ = occupied_len_ = 0;
}

void KvCache::Allocate(const std::vector<int64_t> &k_bytes,
                       const std::vector<int64_t> &v_bytes) {
  if (k_bytes.size() != kNumKvLayers || v_bytes.size() != kNumKvLayers)
    throw std::invalid_argument("KV allocation requires 15 K and V sizes");
  for (int i = 0; i < kNumKvLayers; ++i) {
    const int64_t logical = int64_t(kCacheLen) * kHeadDims[i];
    if (k_bytes[i] < logical || v_bytes[i] < logical ||
        static_cast<uint64_t>(k_bytes[i]) >
            std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(v_bytes[i]) > std::numeric_limits<size_t>::max())
      throw std::invalid_argument(
          "KV allocation is smaller than its contiguous matrix");
  }
  KvCache next;
  next.k_bytes_ = k_bytes;
  next.v_bytes_ = v_bytes;
  for (int i = 0; i < kNumKvLayers; ++i) {
    HBUCP_CHECK(hbUCPMallocCached(&next.k_mem_[i], k_bytes[i], 0),
                "alloc kv k cache");
    if (!next.k_mem_[i].virAddr)
      throw std::runtime_error("SDK returned null K cache");
    HBUCP_CHECK(hbUCPMallocCached(&next.v_mem_[i], v_bytes[i], 0),
                "alloc kv v cache");
    if (!next.v_mem_[i].virAddr)
      throw std::runtime_error("SDK returned null V cache");
    std::memset(next.k_mem_[i].virAddr, 0, static_cast<size_t>(k_bytes[i]));
    std::memset(next.v_mem_[i].virAddr, 0, static_cast<size_t>(v_bytes[i]));
  }
  k_mem_.swap(next.k_mem_);
  v_mem_.swap(next.v_mem_);
  k_bytes_.swap(next.k_bytes_);
  v_bytes_.swap(next.v_bytes_);
  phys_of_global_.clear();
  cache_start_ = occupied_len_ = 0;
  // next now owns the previous allocations; its destructor releases them.
}

void KvCache::Reset() {
  // Zero out existing BPU buffers without freeing/reallocating
  for (int i = 0; i < kNumKvLayers; ++i) {
    if (k_mem_[i].virAddr != nullptr && !k_bytes_.empty()) {
      std::memset(k_mem_[i].virAddr, 0, static_cast<size_t>(k_bytes_[i]));
    }
    if (v_mem_[i].virAddr != nullptr && !v_bytes_.empty()) {
      std::memset(v_mem_[i].virAddr, 0, static_cast<size_t>(v_bytes_[i]));
    }
  }
  phys_of_global_.clear();
  cache_start_ = 0;
  occupied_len_ = 0;
}

void KvCache::ValidateAppend(const int8_t *const *keys,
                             const int8_t *const *values,
                             const int64_t *strides, int start,
                             int rows) const {
  if (!keys || !values || !strides || rows <= 0 || rows > kChunkSize ||
      start != occupied_len_ || start < 0 ||
      start > std::numeric_limits<int>::max() - rows ||
      k_bytes_.size() != kNumKvLayers || v_bytes_.size() != kNumKvLayers)
    throw std::invalid_argument(
        "KV append requires allocated buffers and a contiguous valid chunk");
  for (int layer = 0; layer < kNumKvLayers; ++layer) {
    if (!keys[layer] || !values[layer] || !k_mem_[layer].virAddr ||
        !v_mem_[layer].virAddr || strides[layer] < kHeadDims[layer] ||
        (rows > 1 &&
         strides[layer] >
             (std::numeric_limits<std::ptrdiff_t>::max() - kHeadDims[layer]) /
                 (rows - 1)))
      throw std::invalid_argument(
          "KV source pointers or row strides are invalid");
  }
}

void KvCache::RollAppendLayer(int layer, const int8_t *k_src,
                              const int8_t *v_src, int rows,
                              int64_t row_stride) {
  if (rows <= 0) {
    return;
  }
  const int hd = kHeadDims[layer];
  const int shift = rows * hd;
  int8_t *k_past = KLayer(layer);
  int8_t *v_past = VLayer(layer);
  const int keep = kCacheLen * hd - shift;
  if (keep < 0) {
    throw std::runtime_error("KV roll exceeds cache size");
  }
  std::memmove(k_past, k_past + shift, static_cast<size_t>(keep));
  std::memmove(v_past, v_past + shift, static_cast<size_t>(keep));
  for (int t = 0; t < rows; ++t) {
    std::memcpy(k_past + keep + t * hd,
                k_src +
                    static_cast<size_t>(t) * static_cast<size_t>(row_stride),
                static_cast<size_t>(hd));
    std::memcpy(v_past + keep + t * hd,
                v_src +
                    static_cast<size_t>(t) * static_cast<size_t>(row_stride),
                static_cast<size_t>(hd));
  }
}

void KvCache::AppendPrefillChunk(const int8_t *const *k_outs,
                                 const int8_t *const *v_outs,
                                 const int64_t *row_strides, int chunk_start,
                                 int chunk_valid) {
  ValidateAppend(k_outs, v_outs, row_strides, chunk_start, chunk_valid);
  // Prepare potentially allocating bookkeeping before changing any live rows.
  auto positions = phys_of_global_;
  positions.resize(static_cast<size_t>(chunk_start + chunk_valid), -1);
  for (int &position : positions)
    if (position >= 0)
      position = std::max(-1, position - chunk_valid);
  for (int t = 0; t < chunk_valid; ++t)
    positions[static_cast<size_t>(chunk_start + t)] =
        kCacheLen - chunk_valid + t;
  for (int layer = 0; layer < kNumKvLayers; ++layer)
    RollAppendLayer(layer, k_outs[layer], v_outs[layer], chunk_valid,
                    row_strides[layer]);
  phys_of_global_.swap(positions);
  occupied_len_ = chunk_start + chunk_valid;
  cache_start_ = std::max(0, occupied_len_ - kCacheLen);
}

void KvCache::AppendDecodeStep(const int8_t *const *k_outs,
                               const int8_t *const *v_outs,
                               const int64_t *row_strides, int global_pos) {
  AppendPrefillChunk(k_outs, v_outs, row_strides, global_pos, 1);
}

int KvCache::PhysicalIndex(int global_pos) const {
  if (global_pos < 0 ||
      global_pos >= static_cast<int>(phys_of_global_.size())) {
    return -1;
  }
  return phys_of_global_[static_cast<size_t>(global_pos)];
}

void KvCache::CompactShift(int n_keep, int discard) {
  if (discard < 0 || n_keep < 0)
    throw std::invalid_argument("KV compaction arguments must be nonnegative");
  if (discard == 0)
    return;
  const int cached_tokens = std::min(occupied_len_, kCacheLen);
  if (k_bytes_.size() != kNumKvLayers || n_keep > cached_tokens ||
      discard != occupied_len_ - n_keep)
    throw std::invalid_argument(
        "KV compaction retains a prefix and discards the entire suffix");
  if (n_keep == 0) {
    Reset();
    return;
  }
  std::vector<int> positions(static_cast<size_t>(n_keep));
  for (int index = 0; index < n_keep; ++index)
    positions[index] = kCacheLen - n_keep + index;

  // KV rows are right-aligned. Move the retained prefix from the beginning of
  // the occupied range to the end of the cache before replaying the suffix.
  for (int layer = 0; layer < kNumKvLayers; ++layer) {
    const size_t row_bytes = static_cast<size_t>(kHeadDims[layer]);
    const size_t keep_bytes = static_cast<size_t>(n_keep) * row_bytes;
    const size_t total_bytes = static_cast<size_t>(kCacheLen) * row_bytes;
    const size_t source_offset =
        total_bytes - static_cast<size_t>(cached_tokens) * row_bytes;
    const size_t target_offset = total_bytes - keep_bytes;
    std::memmove(KLayer(layer) + target_offset, KLayer(layer) + source_offset,
                 keep_bytes);
    std::memmove(VLayer(layer) + target_offset, VLayer(layer) + source_offset,
                 keep_bytes);
    std::memset(KLayer(layer), 0, target_offset);
    std::memset(VLayer(layer), 0, target_offset);
  }

  phys_of_global_.swap(positions);
  occupied_len_ = n_keep;
  cache_start_ = 0;
}

} // namespace gemma4
