// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "e11_decode.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#define EXPECT(v)                                                              \
  do {                                                                         \
    if (!(v))                                                                  \
      throw std::runtime_error(#v);                                            \
  } while (0)
template <class F> void rejects(F fn) {
  bool bad = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    bad = true;
  }
  EXPECT(bad);
}
int main() {
  std::array<std::vector<float>, 10> outputs;
  for (int scale = 0; scale < 3; ++scale) {
    int grid = 80 >> scale;
    outputs[3 * scale].assign(grid * grid * 4585, -100);
    outputs[3 * scale + 1].assign(grid * grid * 64, 0);
    outputs[3 * scale + 2].assign(grid * grid * 32, 0);
  }
  outputs[9].assign(160 * 160 * 32, 0);
  // Uniform DFL yields distance 7.5. Adjacent anchors overlap strongly.
  outputs[0][7] = 2;
  outputs[0][4585 + 7] = 1;
  outputs[0][2 * 4585 + 8] = 3;
  outputs[2][0] = 17;
  outputs[2][32] = 99;
  outputs[2][64] = 42;
  auto result = yoloe::decode_e11(outputs, 0.25f, 0.7f);
  EXPECT(result.size() == 2);
  EXPECT(result[0].label == 7);
  EXPECT(result[1].label == 8);
  EXPECT((result[0].box == std::array<float, 4>{-56, -56, 64, 64}));
  EXPECT(result[0].coefficients[0] == 17);
  EXPECT(result[1].coefficients[0] == 42);
  // Native source retains equality at both score and IoU boundaries.
  outputs[0][7] = 0;
  outputs[0][4585 + 7] = -100;
  outputs[0][2 * 4585 + 8] = -100;
  EXPECT(yoloe::decode_e11(outputs, 0.5f, 0.7f).size() == 1);
  auto a = result[0], b = a;
  b.coefficients[0] = 99;
  auto tied = yoloe::detail::nms_e11({a, b}, 1.f);
  EXPECT(tied.size() == 2);
  tied = yoloe::detail::nms_e11({a, b}, 0.7f);
  EXPECT(tied.size() == 1);
  EXPECT(tied[0].coefficients[0] == 17);
  b.box = {100, 100, 110, 110};
  EXPECT(yoloe::detail::nms_e11({a, b}, 0.f).size() == 2);
  rejects([&] { yoloe::decode_e11(outputs, 0.f, 0.7f); });
  rejects([&] { yoloe::decode_e11(outputs, 0.25f, -1.f); });
  outputs[1].pop_back();
  rejects([&] { yoloe::decode_e11(outputs); });
  outputs[1].push_back(0);
  outputs[9][0] = std::numeric_limits<float>::infinity();
  rejects([&] { yoloe::decode_e11(outputs); });
  outputs[9][0] = 0;
  // Extreme finite logits must not destabilize the shared softmax.
  for (int side = 0; side < 4; ++side) {
    for (int bin = 0; bin < 16; ++bin)
      outputs[1][side * 16 + bin] = -10000;
    outputs[1][side * 16 + 3] = 10000;
  }
  result = yoloe::decode_e11(outputs, 0.5f, 0.7f);
  EXPECT((result[0].box == std::array<float, 4>{-20, -20, 28, 28}));
}
