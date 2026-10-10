// Contract tests for the folded Vision engine's descriptor/IO helpers: host
// buffers only; the SDK link stubs come from vision_fixture.hpp.
#include "gemma4_vision_engine.hpp"
#include "vision_fixture.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

template <class F> void rejects(F f) {
  bool threw = false;
  try {
    f();
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw);
}
hbDNNTensorProperties props(bool input, int dtype, int pad = 0) {
  hbDNNTensorProperties p{};
  p.tensorType = dtype;
  p.quantiType = NONE;
  p.validShape.numDimensions = 2;
  p.validShape.dimensionSize[0] = input ? 2520 : 280;
  p.validShape.dimensionSize[1] = input ? 768 : 1536;
  const int bytes = dtype == HB_DNN_TENSOR_TYPE_F16 ? 2 : 4;
  p.stride[1] = bytes;
  p.stride[0] = p.validShape.dimensionSize[1] * bytes + pad;
  p.alignedByteSize = p.stride[0] * p.validShape.dimensionSize[0];
  return p;
}
int main() {
  using namespace gemma4;
  auto p = props(true, HB_DNN_TENSOR_TYPE_F16, 16);
  std::vector<unsigned char> memory(p.alignedByteSize, 0xcd);
  hbDNNTensor input{};
  input.properties = p;
  input.sysMem.virAddr = memory.data();
  std::vector<float> patches(2520 * 768, 0.5f);
  patches[0] = 1.f;
  WriteVisionInput(input, patches, p.alignedByteSize);
  uint16_t first = 0, next = 0;
  std::memcpy(&first, memory.data(), 2);
  std::memcpy(&next, memory.data() + p.stride[0], 2);
  assert(first == 0x3c00 && next == 0x3800);
  assert(memory[768 * 2] == 0);
  // Preserve the source truncation, including a positive half subnormal.
  patches[0] = 0.33333334f;
  patches[1] = std::ldexp(1.f, -24);
  WriteVisionInput(input, patches, p.alignedByteSize);
  std::memcpy(&first, memory.data(), 2);
  std::memcpy(&next, memory.data() + 2, 2);
  assert(first == 0x3555 && next == 1);
  auto bad = patches;
  bad[0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { WriteVisionInput(input, bad, p.alignedByteSize); });
  bad[0] = -0.1f;
  rejects([&] { WriteVisionInput(input, bad, p.alignedByteSize); });
  rejects([&] { WriteVisionInput(input, {1.f}, p.alignedByteSize); });
  for (int dtype : {HB_DNN_TENSOR_TYPE_F16, HB_DNN_TENSOR_TYPE_F32}) {
    auto op = props(false, dtype, 32);
    // Also exercise padding between individual elements, not just rows.
    op.stride[1] *= 2;
    op.stride[0] = 1536 * op.stride[1] + 32;
    op.alignedByteSize = 280 * op.stride[0];
    std::vector<unsigned char> output(op.alignedByteSize, 0xcd);
    for (int row = 0; row < 280; ++row)
      for (int col = 0; col < 1536; ++col) {
        auto *dst = output.data() + row * op.stride[0] + col * op.stride[1];
        if (dtype == HB_DNN_TENSOR_TYPE_F16) {
          uint16_t v = 0xb800;
          std::memcpy(dst, &v, 2);
        } else {
          float v = -0.5f;
          std::memcpy(dst, &v, 4);
        }
      }
    hbDNNTensor tensor{};
    tensor.properties = op;
    tensor.sysMem.virAddr = output.data();
    auto features = ReadVisionOutput(tensor, op.alignedByteSize);
    assert(features.size() == 280u * 1536);
    for (float value : features)
      assert(value == -0.5f);
    output[0] = 0;
    assert(features[0] == -0.5f);
    if (dtype == HB_DNN_TENSOR_TYPE_F16) {
      uint16_t v = 0xfc00;
      std::memcpy(output.data(), &v, 2);
    } else {
      float v = std::numeric_limits<float>::infinity();
      std::memcpy(output.data(), &v, 4);
    }
    rejects([&] { ReadVisionOutput(tensor, op.alignedByteSize); });
    rejects([&] { ReadVisionOutput(tensor, op.alignedByteSize - 1); });
  }
  for (int which = 0; which < 9; ++which) {
    auto invalid = p;
    if (which == 0)
      invalid.tensorType = HB_DNN_TENSOR_TYPE_F32;
    if (which == 1)
      invalid.quantiType = SCALE;
    if (which == 2)
      invalid.validShape.dimensionSize[0] = 2521;
    if (which == 3)
      invalid.validShape.numDimensions = 99;
    if (which == 4)
      invalid.stride[1] = 1;
    if (which == 5)
      invalid.stride[0] = 2;
    if (which == 6)
      invalid.alignedByteSize = 2;
    if (which == 7)
      invalid.stride[0] = std::numeric_limits<int64_t>::max();
    if (which == 8)
      invalid.stride[0] = -1;
    rejects([&] { ValidateVisionTensor(invalid, true); });
  }
  // The same logical matrix may carry leading singleton dimensions.
  auto batched = p;
  batched.validShape.numDimensions = 3;
  batched.validShape.dimensionSize[0] = 1;
  batched.validShape.dimensionSize[1] = 2520;
  batched.validShape.dimensionSize[2] = 768;
  batched.stride[0] = p.alignedByteSize;
  batched.stride[1] = p.stride[0];
  batched.stride[2] = 2;
  ValidateVisionTensor(batched, true);
  batched.validShape.dimensionSize[0] = 2;
  rejects([&] { ValidateVisionTensor(batched, true); });
  auto unknown = props(false, HB_DNN_TENSOR_TYPE_F32);
  unknown.tensorType = HB_DNN_TENSOR_TYPE_S32;
  rejects([&] { ValidateVisionTensor(unknown, false); });
  input.sysMem.virAddr = nullptr;
  rejects([&] { WriteVisionInput(input, patches, p.alignedByteSize); });
  std::cout
      << "vision tensor dtype/shape/padding/capacity/finite checks passed\n";
}
