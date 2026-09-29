/**
 * @file gemma4_kv_cache.hpp
 * @brief Zero-copy KV cache for the Gemma4-E2B text decoder.
 *
 * Owns BPU-allocated K/V tensors for every decoder layer with per-layer
 * aligned byte sizes. Pointers are shared with the model's prefill / decode
 * input slots. CPU append/compaction moves cache rows; sharing the input
 * slots avoids a separate prefill-to-decode buffer copy.
 *
 * @note Not thread-safe; one cache per text engine.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#include "gemma4_config.hpp"

namespace gemma4 {

/**
 * @brief Own the per-layer BPU key-value cache for one text session.
 *
 * The cache exposes model input buffers, appends prefill/decode outputs, and
 * compacts retained tokens after whole conversation turns are removed.
 *
 * @note Instances are not thread-safe.
 */
class KvCache {
public:
  KvCache();
  ~KvCache();

  KvCache(const KvCache &) = delete;
  KvCache &operator=(const KvCache &) = delete;

  /// Global position of the oldest resident row (zero until the window rolls).
  int CacheStart() const { return cache_start_; }
  /// Logical end position; at most kCacheLen rows are physically resident.
  int OccupiedLen() const { return occupied_len_; }

  /// Mutable key-tensor pointer for decoder layer @p i.
  int8_t *KLayer(int i) { return static_cast<int8_t *>(k_mem_.at(i).virAddr); }
  /// Mutable value-tensor pointer for decoder layer @p i.
  int8_t *VLayer(int i) { return static_cast<int8_t *>(v_mem_.at(i).virAddr); }
  /// Const key-tensor pointer for decoder layer @p i.
  const int8_t *KLayer(int i) const {
    return static_cast<const int8_t *>(k_mem_.at(i).virAddr);
  }
  /// Const value-tensor pointer for decoder layer @p i.
  const int8_t *VLayer(int i) const {
    return static_cast<const int8_t *>(v_mem_.at(i).virAddr);
  }

  /// Mutable UCP memory struct for decoder layer @p i (keys).
  hbUCPSysMem &KMem(int i) { return k_mem_.at(i); }
  /// Mutable UCP memory struct for decoder layer @p i (values).
  hbUCPSysMem &VMem(int i) { return v_mem_.at(i); }

  /// Zero K/V buffers and clear positions; retain allocations and aliases.
  void Reset();

  /**
   * @brief Replace all KV allocations transactionally and reset positions.
   * Failed allocation preserves old buffers/state. Successful replacement
   * invalidates old aliases: only allocate before binding model input slots.
   * Each matrix is contiguous S8 [4096, head_dim], plus optional trailing
   * padding.
   *
   * @param k_bytes Per-layer aligned key-byte sizes (from model inputs).
   * @param v_bytes Per-layer aligned value-byte sizes (from model inputs).
   */
  void Allocate(const std::vector<int64_t> &k_bytes,
                const std::vector<int64_t> &v_bytes);

  /**
   * @brief Copy a prefill chunk's K/V outputs into the cache.
   *
   * @param k_outs Per-layer key output pointers.
   * @param v_outs Per-layer value output pointers.
   * @param row_strides Per-layer output row stride in bytes.
   * @param chunk_start Global token offset where the chunk begins.
   * @param chunk_valid Number of valid tokens in the chunk.
   */
  void AppendPrefillChunk(const int8_t *const *k_outs,
                          const int8_t *const *v_outs,
                          const int64_t *row_strides, int chunk_start,
                          int chunk_valid);

  /**
   * @brief Copy one decode step's K/V output into the cache.
   *
   * @param k_outs Per-layer key output pointers.
   * @param v_outs Per-layer value output pointers.
   * @param row_strides Per-layer output row stride in bytes.
   * @param global_pos Global token position being decoded.
   */
  void AppendDecodeStep(const int8_t *const *k_outs,
                        const int8_t *const *v_outs, const int64_t *row_strides,
                        int global_pos);

  /// Map a global token position to its physical cache index.
  int PhysicalIndex(int global_pos) const;

  /**
   * @brief Retain the leading n_keep resident rows and reset their positions.
   * The suffix is discarded and must be replayed by the caller. This is not
   * an arbitrary middle-range deletion. discard must equal
   * OccupiedLen()-n_keep.
   *
   * @param n_keep Number of leading tokens preserved.
   * @param discard Number of tokens discarded after the preserved prefix.
   */
  void CompactShift(int n_keep, int discard);

private:
  void RollAppendLayer(int layer, const int8_t *k_src, const int8_t *v_src,
                       int rows, int64_t row_stride);
  void ValidateAppend(const int8_t *const *k_outs, const int8_t *const *v_outs,
                      const int64_t *strides, int start, int rows) const;
  void FreeMem() noexcept;

  std::vector<hbUCPSysMem> k_mem_;
  std::vector<hbUCPSysMem> v_mem_;
  std::vector<int64_t> k_bytes_, v_bytes_;
  std::vector<int> phys_of_global_;
  int cache_start_ = 0;
  int occupied_len_ = 0;
};

} // namespace gemma4
