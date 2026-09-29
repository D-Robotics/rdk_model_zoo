/** @file gemma4_model_io.hpp
 * @brief Own text-subgraph tensor buffers and track borrowed KV inputs.
 */
#pragma once
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gemma4 {
// The subgraph handle is borrowed from the packed model. Destroy/clear these
// tensor owners before releasing that packed model. KV input memory is borrowed
// from KvCache and is never released by this class.
struct ModelIo {
  hbDNNHandle_t handle = nullptr;
  std::vector<hbDNNTensor> inputs;
  std::vector<hbDNNTensor> outputs;
  int seq_len = 0;

  ModelIo() = default;
  ~ModelIo() { Clear(); }
  ModelIo(const ModelIo &) = delete;
  ModelIo &operator=(const ModelIo &) = delete;
  ModelIo(ModelIo &&other) noexcept { Swap(other); }
  ModelIo &operator=(ModelIo &&other) noexcept {
    if (this != &other) {
      Clear();
      Swap(other);
    }
    return *this;
  }

  // Adopt an allocated tensor and record its buffer capacity. The SDK can
  // refresh output descriptors after inference; the recorded capacity is the
  // allocation the engine actually owns, not the refreshed claim.
  void AddInput(hbDNNTensor tensor) {
    input_capacity_.push_back(tensor.properties.alignedByteSize);
    inputs.push_back(std::move(tensor));
  }
  void AddOutput(hbDNNTensor tensor) {
    output_capacity_.push_back(tensor.properties.alignedByteSize);
    outputs.push_back(std::move(tensor));
  }

  // Recorded allocation capacity, or 0 for tensors adopted directly through
  // the public vectors (untracked).
  int64_t InputCapacity(std::size_t index) const {
    return index < input_capacity_.size() ? input_capacity_[index] : 0;
  }
  int64_t OutputCapacity(std::size_t index) const {
    return index < output_capacity_.size() ? output_capacity_[index] : 0;
  }

  void BindBorrowedInput(std::size_t index, const hbUCPSysMem &memory,
                         int64_t capacity_bytes = 0) {
    auto &tensor = inputs.at(index);
    if (!memory.virAddr)
      throw std::invalid_argument("null borrowed KV buffer");
    if (capacity_bytes > 0 &&
        capacity_bytes < tensor.properties.alignedByteSize)
      throw std::invalid_argument(
          "borrowed KV buffer is smaller than the declared allocation");
    // Allocate bookkeeping before changing any ownership.
    borrowed_.resize(inputs.size(), false);
    input_capacity_.resize(inputs.size(), 0);
    if (!borrowed_[index] && tensor.sysMem.virAddr) {
      hbUCPFree(&tensor.sysMem);
    }
    tensor.sysMem = memory;
    borrowed_[index] = true;
    input_capacity_[index] = capacity_bytes > 0 ? capacity_bytes : 0;
  }

  void Clear() noexcept {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if ((i >= borrowed_.size() || !borrowed_[i]) && inputs[i].sysMem.virAddr)
        hbUCPFree(&inputs[i].sysMem);
    }
    for (auto &tensor : outputs)
      if (tensor.sysMem.virAddr)
        hbUCPFree(&tensor.sysMem);
    inputs.clear();
    outputs.clear();
    borrowed_.clear();
    input_capacity_.clear();
    output_capacity_.clear();
    handle = nullptr;
    seq_len = 0;
  }

private:
  void Swap(ModelIo &other) noexcept {
    std::swap(handle, other.handle);
    inputs.swap(other.inputs);
    outputs.swap(other.outputs);
    borrowed_.swap(other.borrowed_);
    input_capacity_.swap(other.input_capacity_);
    output_capacity_.swap(other.output_capacity_);
    std::swap(seq_len, other.seq_len);
  }
  std::vector<bool> borrowed_;
  std::vector<int64_t> input_capacity_;
  std::vector<int64_t> output_capacity_;
};
} // namespace gemma4
