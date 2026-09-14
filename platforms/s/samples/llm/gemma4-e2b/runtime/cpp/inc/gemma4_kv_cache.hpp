/**
 * @file gemma4_kv_cache.hpp
 * @brief Zero-copy KV cache for the Gemma4-E2B text decoder.
 *
 * Owns BPU-allocated K/V tensors for every decoder layer with per-layer
 * aligned byte sizes. Pointers are shared with the model's prefill / decode
 * input slots, so prefill writes the cache in-place and decode reads from
 * it without any memcpy between steps.
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

  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;

  /// Physical cache offset where the next token will be written.
  int CacheStart() const { return cache_start_; }
  /// Number of tokens currently occupied in the cache.
  int OccupiedLen() const { return occupied_len_; }

  /// Mutable key-tensor pointer for decoder layer @p i.
  int8_t* KLayer(int i) { return static_cast<int8_t*>(k_mem_[i].virAddr); }
  /// Mutable value-tensor pointer for decoder layer @p i.
  int8_t* VLayer(int i) { return static_cast<int8_t*>(v_mem_[i].virAddr); }
  /// Const key-tensor pointer for decoder layer @p i.
  const int8_t* KLayer(int i) const {
    return static_cast<const int8_t*>(k_mem_[i].virAddr);
  }
  /// Const value-tensor pointer for decoder layer @p i.
  const int8_t* VLayer(int i) const {
    return static_cast<const int8_t*>(v_mem_[i].virAddr);
  }

  /// Mutable UCP memory struct for decoder layer @p i (keys).
  hbUCPSysMem& KMem(int i) { return k_mem_[i]; }
  /// Mutable UCP memory struct for decoder layer @p i (values).
  hbUCPSysMem& VMem(int i) { return v_mem_[i]; }

  /// Clear all cache state and release layer memory.
  void Reset();

  /**
   * @brief Allocate BPU-compatible memory for all KV layers.
   *
   * @param k_bytes Per-layer aligned key-byte sizes (from model inputs).
   * @param v_bytes Per-layer aligned value-byte sizes (from model inputs).
   */
  void Allocate(const std::vector<int64_t>& k_bytes,
                const std::vector<int64_t>& v_bytes);

  /**
   * @brief Copy a prefill chunk's K/V outputs into the cache.
   *
   * @param k_outs Per-layer key output pointers.
   * @param v_outs Per-layer value output pointers.
   * @param row_strides Per-layer output row stride in bytes.
   * @param chunk_start Global token offset where the chunk begins.
   * @param chunk_valid Number of valid tokens in the chunk.
   */
  void AppendPrefillChunk(const int8_t* const* k_outs, const int8_t* const* v_outs,
                          const int64_t* row_strides, int chunk_start,
                          int chunk_valid);

  /**
   * @brief Copy one decode step's K/V output into the cache.
   *
   * @param k_outs Per-layer key output pointers.
   * @param v_outs Per-layer value output pointers.
   * @param row_strides Per-layer output row stride in bytes.
   * @param global_pos Global token position being decoded.
   */
  void AppendDecodeStep(const int8_t* const* k_outs, const int8_t* const* v_outs,
                        const int64_t* row_strides, int global_pos);

  /// Map a global token position to its physical cache index.
  int PhysicalIndex(int global_pos) const;

  /// Set the number of occupied tokens directly.
  void SetOccupiedLen(int len) { occupied_len_ = len; }

  /**
   * @brief Compact the cache to drop a middle range of tokens.
   *
   * Keeps [0, n_keep) and [n_keep+discard, occupied_len_), moving the tail
   * physically to start at index @p n_keep and updating the global→physical
   * mapping.
   *
   * @param n_keep Number of leading tokens preserved.
   * @param discard Number of tokens discarded after the preserved prefix.
   */
  void CompactShift(int n_keep, int discard);

 private:
  void RollAppendLayer(int layer, const int8_t* k_src, const int8_t* v_src,
                       int rows, int64_t row_stride);
  void ShiftPhysicalIndices(int delta);
  void FreeMem();

  std::vector<hbUCPSysMem> k_mem_;
  std::vector<hbUCPSysMem> v_mem_;
  std::vector<int64_t> layer_bytes_;
  std::vector<int> phys_of_global_;
  int cache_start_ = 0;
  int occupied_len_ = 0;
};

}  // namespace gemma4
