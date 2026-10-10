// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "detect.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#define EXPECT(v)                                                              \
  do {                                                                         \
    if (!(v))                                                                  \
      throw std::runtime_error(#v);                                            \
  } while (0)
template <class F> void rejects(F fn) {
  bool failed = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  EXPECT(failed);
}
int main() {
  std::array<std::vector<float>, 10> outputs;
  for (int scale = 0; scale < 3; ++scale) {
    int grid = 80 >> scale;
    outputs[scale * 3].assign(grid * grid * 4585, -100.f);
    outputs[scale * 3 + 1].assign(grid * grid * 4, 1.f);
    outputs[scale * 3 + 2].assign(grid * grid * 32, 0.f);
  }
  outputs[9].assign(160 * 160 * 32, 0.f);
  // Exact ties: lower scale/anchor/class wins; overlapping boxes keep both.
  outputs[0][7] = 2;
  outputs[0][8] = 2;
  outputs[0][4585 + 9] = 2;
  outputs[3][3] = 2;
  outputs[2][0] = 17;
  auto single = yoloe::decode_e26(outputs, 0.25f, 3, true);
  EXPECT(single.size() == 3);
  EXPECT(single[0].label == 7);
  EXPECT(single[1].label == 9);
  EXPECT(single[2].label == 3);
  EXPECT((single[0].box == std::array<float, 4>({-4, -4, 12, 12})));
  EXPECT(single[0].coefficients[0] == 17);
  EXPECT(std::abs(single[0].score - 0.880797f) < 1e-6f);
  auto multi = yoloe::decode_e26(outputs, 0.25f, 3, false);
  EXPECT(multi.size() == 3);
  EXPECT(multi[0].label == 7);
  EXPECT(multi[1].label == 8);
  EXPECT(multi[2].label == 9);
  EXPECT(yoloe::decode_e26(outputs, 0.99f, 3, true).empty());
  outputs[0][7] = 0;
  outputs[0][8] = -100;
  outputs[0][4585 + 9] = -100;
  outputs[3][3] = -100;
  EXPECT(yoloe::decode_e26(outputs, 0.5f, 3, true).empty()); // strict threshold
  rejects([&] { yoloe::decode_e26(outputs, 0.f, 3, true); });
  rejects([&] { yoloe::decode_e26(outputs, 0.25f, 0, true); });
  rejects([&] { yoloe::decode_e26(outputs, 0.25f, 8401, true); });
  outputs[9].pop_back();
  rejects([&] { yoloe::decode_e26(outputs, 0.25f, 3, true); });
  outputs[9].push_back(0);
  outputs[9][0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { yoloe::decode_e26(outputs, 0.25f, 3, true); });
  outputs[9][0] = 0;
  outputs[0][7] = 2;
  outputs[1][0] = std::numeric_limits<float>::max();
  rejects([&] { yoloe::decode_e26(outputs, 0.25f, 3, true); });
}
