/*
 * Copyright (c) 2025, XiangshunZhao D-Robotics.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file nn_math.cc
 * @brief Provide lightweight neural network related math utilities.
 *
 * This file implements common mathematical helpers used during model
 * postprocessing, such as activation functions and probability normalization.
 * The implementations are designed to be simple, efficient, and reusable
 * across different model samples.
 */

#include <vector>
#include <cmath>

#include "model_types.hpp"
#include "nn_math.hpp"

/**
 * @brief Convert logits to normalized probabilities in-place using softmax (stable).
 * @param[in,out] top_k_cls   Vector of (id, score) pairs; scores are treated as logits, then replaced by probabilities.
 */
void logits_to_probabilities(std::vector<Classification> &top_k_cls) {
  if (top_k_cls.empty()) return;

  // Find max for numerical stability
  float max_logit = top_k_cls[0].score;
  for (const auto &cls : top_k_cls) {
    if (cls.score > max_logit) max_logit = cls.score;
  }

  float sum = 0.f;
  for (auto &cls : top_k_cls) {
    cls.score = std::exp(cls.score - max_logit); // exp-shifted
    sum += cls.score;
  }

  // Normalize to probabilities
  for (auto &cls : top_k_cls) {
    cls.score /= sum;
  }
}
