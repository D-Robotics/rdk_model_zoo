// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "geometry.h"
#include "nv12.h"
#include <array>
#include <vector>
namespace yoloe {
using Heads = std::array<std::vector<float>, 10>;
// Backend owns model/tensor resources. Return independent compact semantic
// FLOAT32 heads; no borrowed SDK buffers may escape infer(). SDK metadata and
// hardware/artifact identity validation belong to the concrete backend.
class Runner {
public:
  virtual ~Runner() = default;
  virtual Protocol protocol() const = 0;
  virtual Heads infer(const Nv12Input &input) = 0;
};
} // namespace yoloe
