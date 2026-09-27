// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "geometry.h"
#include <optional>
namespace yoloe {
struct Config {
  Protocol protocol = Protocol::E11;
  float score_threshold = 0.25f;
  std::optional<float> nms_threshold;
  int max_det = 300;
  bool single_label = true;
  bool do_morph = false;
  int resize_type = 1;
};
inline void validate_config(const Config &cfg) {
  if (cfg.protocol != Protocol::E11 && cfg.protocol != Protocol::E26)
    throw std::invalid_argument("Unknown YOLOE protocol");
  if (!std::isfinite(cfg.score_threshold) || cfg.score_threshold <= 0 ||
      cfg.score_threshold >= 1)
    throw std::invalid_argument("Score threshold must be finite in (0,1)");
  if (cfg.protocol == Protocol::E11) {
    if (cfg.max_det != 300 || !cfg.single_label ||
        (cfg.resize_type != 0 && cfg.resize_type != 1))
      throw std::invalid_argument(
          "E11 requires single-label and no E26 Top-K override");
    if (cfg.nms_threshold && (!std::isfinite(*cfg.nms_threshold) ||
                              *cfg.nms_threshold < 0 || *cfg.nms_threshold > 1))
      throw std::invalid_argument("E11 NMS must be finite in [0,1]");
  } else if (cfg.nms_threshold || cfg.do_morph || cfg.resize_type != 1 ||
             cfg.max_det < 1 || cfg.max_det > 8400)
    throw std::invalid_argument(
        "E26 requires letterbox, no NMS/morphology and max_det in [1,8400]");
}
} // namespace yoloe
