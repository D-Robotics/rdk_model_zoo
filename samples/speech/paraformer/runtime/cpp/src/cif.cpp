// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cif.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace paraformer {
namespace {
void finite_values(const std::vector<float> &values, size_t expected,
                   const char *label) {
  if (values.size() != expected ||
      std::any_of(values.begin(), values.end(),
                  [](float value) { return !std::isfinite(value); }))
    throw std::invalid_argument(label);
}
} // namespace

CifOutput cif(const std::vector<float> &weights,
              const std::vector<float> &hidden, int real_frames) {
  finite_values(weights, 401, "Expected 401 finite weights");
  finite_values(hidden, 401 * 512, "Expected 401*512 finite hidden values");
  if (real_frames < 0 || real_frames > 400 ||
      std::any_of(weights.begin(), weights.end(),
                  [](float value) { return value < 0.f; }))
    throw std::invalid_argument("Invalid real frame count or negative weight");
  CifOutput result{std::vector<float>(100 * 512, 0.f), 0};
  std::array<double, 512> prefix_hidden{};
  std::array<float, 512> previous_frame{}, previous_remain{};
  double weight_sum = 0;
  float previous_floor = 0;
  for (int t = 0; t < 401; ++t) {
    const float weight = t < real_frames ? weights[t] : 0.f;
    weight_sum = t == 0 ? double(weight) : weight_sum + double(weight);
    const float prefix = static_cast<float>(weight_sum);
    const float current_floor = std::floor(prefix);
    const bool fire = current_floor - previous_floor > 0;
    previous_floor = current_floor;
    // Keep source float32 operation order; simplifying to prefix-floor changes
    // rounding near integer boundaries after adding the fire indicator.
    float fires = (fire ? 1.f : 0.f) + prefix;
    fires -= current_floor;
    const float remainder = fires - std::floor(fires);
    for (size_t h = 0; h < 512; ++h) {
      const double product =
          double(weight) * double(hidden[size_t(t) * 512 + h]);
      prefix_hidden[h] = t == 0 ? product : prefix_hidden[h] + product;
      if (fire && result.token_count < 100) {
        const float frame = static_cast<float>(prefix_hidden[h]);
        const float remain = remainder * hidden[size_t(t) * 512 + h];
        result.acoustic[size_t(result.token_count) * 512 + h] =
            frame - previous_frame[h] + previous_remain[h] - remain;
        previous_frame[h] = frame;
        previous_remain[h] = remain;
      }
    }
    if (fire && result.token_count < 100)
      ++result.token_count;
  }
  return result;
}
} // namespace paraformer
