// Independent host allocation double: no model, SDK, or board is used.
#include "gemma4_model_io.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <type_traits>
#include <utility>

static std::set<void *> live;
int hbUCPFree(hbUCPSysMem *mem) {
  assert(live.erase(mem->virAddr) == 1);
  std::free(mem->virAddr);
  mem->virAddr = nullptr;
  return 0;
}
static hbDNNTensor Tensor() {
  hbDNNTensor t{};
  t.sysMem.virAddr = std::malloc(16);
  assert(t.sysMem.virAddr);
  live.insert(t.sysMem.virAddr);
  return t;
}
int main() {
  static_assert(!std::is_copy_constructible_v<gemma4::ModelIo>);
  static_assert(std::is_nothrow_move_constructible_v<gemma4::ModelIo>);
  // An exception midway through initialization must free every completed
  // buffer.
  try {
    gemma4::ModelIo partial;
    partial.inputs.push_back(Tensor());
    partial.outputs.push_back(Tensor());
    throw 7;
  } catch (int) {
  }
  assert(live.empty());
  auto cache = Tensor();
  {
    gemma4::ModelIo first;
    first.inputs.push_back(Tensor());
    first.inputs.push_back(Tensor());
    first.outputs.push_back(Tensor());
    first.BindBorrowedInput(1, cache.sysMem);
    assert(live.size() == 3);
    gemma4::ModelIo second;
    second.outputs.push_back(Tensor());
    second = std::move(first); // Releases the destination's old buffers.
    assert(live.size() == 3);
    gemma4::ModelIo third(std::move(second));
    third = std::move(third); // Self move preserves ownership.
    third.Clear();
    third.Clear();
    assert(live.size() == 1 && live.count(cache.sysMem.virAddr));
  }
  hbUCPFree(&cache.sysMem);
  assert(live.empty());
  std::cout << "model IO ownership passed\n";
}
