// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "image_ops.h"
#include "runner.h"
#include <memory>
#include <utility>
namespace yoloe {
class YOLOE;
struct StageIdentity {};
class Prepared {
public:
  const Nv12Input &input() const { return input_; }
  const Geometry &geometry() const { return geometry_; }

private:
  friend class YOLOE;
  Prepared(Nv12Input input, Geometry geometry,
           std::shared_ptr<const StageIdentity> owner)
      : input_(std::move(input)), geometry_(geometry),
        owner_(std::move(owner)) {}
  Nv12Input input_;
  Geometry geometry_;
  std::shared_ptr<const StageIdentity> owner_;
};
class RawBatch {
public:
  const Heads &outputs() const { return outputs_; }
  const Geometry &geometry() const { return geometry_; }

private:
  friend class YOLOE;
  RawBatch(Heads outputs, Geometry geometry,
           std::shared_ptr<const StageIdentity> owner)
      : outputs_(std::move(outputs)), geometry_(geometry),
        owner_(std::move(owner)) {}
  Heads outputs_;
  Geometry geometry_;
  std::shared_ptr<const StageIdentity> owner_;
};
struct Instance {
  std::array<float, 4> box;
  float score;
  int label;
  cv::Mat mask;
};
using Result = std::vector<Instance>;
} // namespace yoloe
