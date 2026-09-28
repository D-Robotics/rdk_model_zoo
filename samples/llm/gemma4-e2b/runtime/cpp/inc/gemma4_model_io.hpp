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

  void BindBorrowedInput(std::size_t index, const hbUCPSysMem &memory) {
    auto &tensor = inputs.at(index);
    if (!memory.virAddr)
      throw std::invalid_argument("null borrowed KV buffer");
    // Allocate bookkeeping before changing any ownership.
    borrowed_.resize(inputs.size(), false);
    if (!borrowed_[index] && tensor.sysMem.virAddr) {
      hbUCPFree(&tensor.sysMem);
    }
    tensor.sysMem = memory;
    borrowed_[index] = true;
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
    handle = nullptr;
    seq_len = 0;
  }

private:
  void Swap(ModelIo &other) noexcept {
    std::swap(handle, other.handle);
    inputs.swap(other.inputs);
    outputs.swap(other.outputs);
    borrowed_.swap(other.borrowed_);
    std::swap(seq_len, other.seq_len);
  }
  std::vector<bool> borrowed_;
};
} // namespace gemma4
