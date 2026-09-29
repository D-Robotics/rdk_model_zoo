// GEMMA-TEXT-R2 regression: ModelIo tensor adoption is transactional under
// allocation failure at each bookkeeping step. Independent host allocation
// double; no SDK, model, or board. Global operator new is replaced to
// inject a deterministic bad_alloc at a chosen allocation index;
// hbUCPFree asserts every buffer is released exactly once.
#include "gemma4_model_io.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <set>

namespace {

std::set<void *> live;
int new_count = 0;
int fail_new_at = -1;  // 1-based allocation index from the last reset.

hbDNNTensor MakeTensor(int64_t bytes) {
  hbDNNTensor tensor{};
  tensor.properties.alignedByteSize = bytes;
  tensor.sysMem.virAddr = std::malloc(static_cast<size_t>(bytes));
  assert(tensor.sysMem.virAddr != nullptr);
  live.insert(tensor.sysMem.virAddr);
  return tensor;
}

}  // namespace

void *operator new(std::size_t size) {
  if (fail_new_at > 0 && ++new_count == fail_new_at) {
    fail_new_at = -1;
    throw std::bad_alloc();
  }
  void *pointer = std::malloc(size);
  if (pointer == nullptr) throw std::bad_alloc();
  return pointer;
}
void operator delete(void *pointer) noexcept { std::free(pointer); }

int hbUCPFree(hbUCPSysMem *mem) {
  assert(mem->virAddr != nullptr);
  assert(live.erase(mem->virAddr) == 1);  // A second release fails here.
  std::free(mem->virAddr);
  mem->virAddr = nullptr;
  return 0;
}

int main() {
  // (1) AddInput failing while pushing the capacity entry: the guard
  // releases the buffer, no vector changes, nothing leaks.
  {
    gemma4::ModelIo io;
    hbDNNTensor tensor = MakeTensor(16);
    new_count = 0;
    fail_new_at = 1;
    bool caught = false;
    try {
      io.AddInput(tensor);
    } catch (const std::bad_alloc &) {
      caught = true;
    }
    assert(caught);
    assert(live.empty());  // Exactly one release: the adoption guard's.
    assert(io.inputs.empty());
    io.Clear();  // Destructor state on an empty owner stays harmless.
    assert(live.empty());
  }

  // (2) AddOutput failing while pushing the tensor: the capacity entry is
  // rolled back and the buffer released; a subsequent successful adoption
  // lands at index 0 with its own capacity.
  {
    gemma4::ModelIo io;
    hbDNNTensor tensor = MakeTensor(32);
    new_count = 0;
    fail_new_at = 2;
    bool caught = false;
    try {
      io.AddOutput(tensor);
    } catch (const std::bad_alloc &) {
      caught = true;
    }
    assert(caught);
    assert(live.empty());
    assert(io.outputs.empty());
    hbDNNTensor next = MakeTensor(48);
    io.AddOutput(next);
    assert(io.outputs.size() == 1);
    assert(io.OutputCapacity(0) == 48);  // No stale entry from the failure.
    io.Clear();
    assert(live.empty());
    assert(io.OutputCapacity(0) == 0);
  }

  // (3) Successful adoptions record capacities; Clear frees each buffer
  // exactly once; repeated Clear stays harmless.
  {
    gemma4::ModelIo io;
    hbDNNTensor input = MakeTensor(16);
    hbDNNTensor output = MakeTensor(24);
    fail_new_at = -1;
    io.AddInput(input);
    io.AddOutput(output);
    assert(live.size() == 2);
    assert(io.InputCapacity(0) == 16);
    assert(io.OutputCapacity(0) == 24);
    io.Clear();
    assert(live.empty());
    assert(io.InputCapacity(0) == 0 && io.OutputCapacity(0) == 0);
    io.Clear();
    assert(live.empty());
  }

  // (4) Both failure stages in sequence leave the owner clean and reusable.
  {
    gemma4::ModelIo io;
    hbDNNTensor first = MakeTensor(8);
    new_count = 0;
    fail_new_at = 1;
    try {
      io.AddInput(first);
    } catch (const std::bad_alloc &) {
    }
    assert(live.empty());
    hbDNNTensor second = MakeTensor(8);
    new_count = 0;
    fail_new_at = 2;
    try {
      io.AddInput(second);
    } catch (const std::bad_alloc &) {
    }
    assert(live.empty());
    hbDNNTensor third = MakeTensor(8);
    fail_new_at = -1;
    io.AddInput(third);
    assert(live.size() == 1);
    assert(io.InputCapacity(0) == 8);
    io.Clear();
    assert(live.empty());
  }

  std::printf("model io adoption transaction passed\n");
  return 0;
}
