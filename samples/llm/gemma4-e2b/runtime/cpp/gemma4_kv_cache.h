#pragma once

#include <cstdint>
#include <vector>

#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#include "gemma4_config.h"

namespace gemma4 {

class KvCache {
 public:
  KvCache();
  ~KvCache();

  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;

  int CacheStart() const { return cache_start_; }
  int OccupiedLen() const { return occupied_len_; }

  int8_t* KLayer(int i) { return static_cast<int8_t*>(k_mem_[i].virAddr); }
  int8_t* VLayer(int i) { return static_cast<int8_t*>(v_mem_[i].virAddr); }
  const int8_t* KLayer(int i) const {
    return static_cast<const int8_t*>(k_mem_[i].virAddr);
  }
  const int8_t* VLayer(int i) const {
    return static_cast<const int8_t*>(v_mem_[i].virAddr);
  }

  hbUCPSysMem& KMem(int i) { return k_mem_[i]; }
  hbUCPSysMem& VMem(int i) { return v_mem_[i]; }

  void Reset();

  // Allocate BPU-compatible memory for all KV layers using per-layer aligned sizes.
  // k_bytes[i] and v_bytes[i] are the alignedByteSize from the model's input tensor.
  void Allocate(const std::vector<int64_t>& k_bytes,
                const std::vector<int64_t>& v_bytes);

  void AppendPrefillChunk(const int8_t* const* k_outs, const int8_t* const* v_outs,
                          const int64_t* row_strides, int chunk_start,
                          int chunk_valid);

  void AppendDecodeStep(const int8_t* const* k_outs, const int8_t* const* v_outs,
                        const int64_t* row_strides, int global_pos);

  int PhysicalIndex(int global_pos) const;

  void SetOccupiedLen(int len) { occupied_len_ = len; }

  // Compact KV cache: keep [0, n_keep) and [n_keep+discard, occupied_len_).
  // Physically moves the tail portion to start at physical index n_keep.
  // Updates occupied_len_ and phys_of_global_ accordingly.
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
