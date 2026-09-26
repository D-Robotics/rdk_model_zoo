#include "tensor_contract.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace yolo26_depth;
template <class F> void rejects(F f) {
  bool threw = false;
  try {
    f();
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw);
}
int main() {
  auto g = letterbox_geometry(1536, 5);
  assert(g.left == 383 &&
         g.right == 383); // Python ties-to-even: width 2, not 3.
  assert(g.top == 0 && g.bottom == 0);
  rejects([] { letterbox_geometry(1, 10000); });
  rejects([] { letterbox_geometry(0, 10); });
  TensorLayout layout{
      {1, 192, 192, 1}, {1, 192, 192, 4}, {}, 192 * 192 * 4 * sizeof(float)};
  auto strides = validated_strides(layout);
  assert(strides[2] == 16 && strides[1] == 192 * 16);
  std::vector<unsigned char> storage(layout.capacity, 0);
  for (int h = 0; h < 192; ++h)
    for (int w = 0; w < 192; ++w) {
      float value = float(h * 192 + w) / 1000;
      std::memcpy(storage.data() + h * strides[1] + w * strides[2], &value,
                  sizeof(value));
    }
  auto values = read_log_depth(storage.data(), layout);
  assert(values.size() == 192 * 192 && values[193] == .193f);
  TensorLayout nchw{
      {1, 1, 192, 192}, {1, 1, 192, 200}, {}, 192 * 200 * sizeof(float)};
  auto ns = validated_strides(nchw);
  assert(ns[2] == 800 && ns[3] == 4);
  std::vector<unsigned char> nb(nchw.capacity, 0);
  const float marker = -3.25f;
  std::memcpy(nb.data() + ns[2] + 2 * ns[3], &marker, 4);
  assert(read_log_depth(nb.data(), nchw)[194] == marker);
  auto bad = layout;
  bad.capacity = 4;
  rejects([&] { validated_strides(bad); });
  bad = layout;
  bad.strides = {100, 100, 16, 4};
  rejects([&] { validated_strides(bad); });
  bad = layout;
  bad.strides = {0, 100, 0, 4};
  rejects([&] { validated_strides(bad); });
  bad = layout;
  bad.strides = {std::numeric_limits<std::size_t>::max(),
                 std::numeric_limits<std::size_t>::max(), 16, 4};
  rejects([&] { validated_strides(bad); });
  bad = layout;
  bad.valid = {1, 192, 192, 2};
  rejects([&] { validated_strides(bad); });
  float nan = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(storage.data(), &nan, 4);
  rejects([&] { read_log_depth(storage.data(), layout); });
  assert(std::abs(percentile({0, 10, 20, 30}, .02) - .6) < 1e-9);
  assert(std::abs(percentile({0, 10, 20, 30}, .98) - 29.4) < 1e-9);
  rejects([] { percentile({}, .02); });
  assert(match_x5_identity("X5\n", "s600", "") == true);
  assert(match_x5_identity("", "X5U", "") == true);
  assert(match_x5_identity("", "", "D-Robotics RDK X5 V1.0") == true);
  assert(match_x5_identity("s100", "X5U", "") == false);
  assert(match_x5_identity("", "unknown", "D-Robotics RDK X5 V1.0") == false);
}
