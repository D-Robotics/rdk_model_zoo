// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "config.h"
#include "e11_decode.h"
#include "e26_decode.h"
#include "pipeline_io.h"
namespace yoloe {
inline Result decode_result(const Heads &heads, const Geometry &geometry,
                            const Config &cfg) {
  validate_config(cfg);
  validate_geometry(geometry);
  if (geometry.protocol != cfg.protocol ||
      geometry.resize_type != cfg.resize_type)
    throw std::invalid_argument(
        "Postprocessing geometry/configuration mismatch");
  auto candidates = cfg.protocol == Protocol::E11
                        ? decode_e11(heads, cfg.score_threshold,
                                     cfg.nms_threshold.value_or(0.7f))
                        : decode_e26(heads, cfg.score_threshold, cfg.max_det,
                                     cfg.single_label);
  auto masks =
      cfg.protocol == Protocol::E11
          ? restore_e11_masks(candidates, heads[9], geometry, cfg.do_morph)
          : restore_e26_masks(candidates, heads[9], geometry);
  Result result;
  result.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i)
    result.push_back({masks[i].box, candidates[i].score, candidates[i].label,
                      std::move(masks[i].mask)});
  return result;
}
} // namespace yoloe
