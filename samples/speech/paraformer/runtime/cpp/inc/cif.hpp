// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <vector>

namespace paraformer {
struct CifOutput {
  std::vector<float> acoustic;
  int32_t token_count;
};
// Fixed batch-one inference contract: weights[401], hidden[401*512].
// real_frames is explicit 0..400; no unmasked calibration mode is inferred.
// Returns owned [100*512] acoustic values and a count capped at 100.
CifOutput cif(const std::vector<float> &weights,
              const std::vector<float> &hidden, int real_frames);
} // namespace paraformer
