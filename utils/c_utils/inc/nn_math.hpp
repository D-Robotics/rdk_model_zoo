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
 * @file nn_math.hpp
 * @brief Declare common mathematical helper interfaces for neural network
 *        inference and result processing.
 *
 * This file provides lightweight math utilities shared across inference
 * and postprocessing components.
 */

#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include "model_types.hpp"


/**
 * @brief Fast sigmoid approximation using std::exp.
 * @param[in] x Raw value (logit).
 * @return float Sigmoid(x) in (0,1).
 */
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Numerically stable softmax for a short vector.
 * @param[in]  input  Pointer to input array (length = len).
 * @param[out] output Pointer to output array (length = len).
 * @param[in]  len    Vector length (default 16).
 */
inline void softmax(const float* input, float* output, int len = 16)
{
    if (len <= 0) return;
    float max_val = *std::max_element(input, input + len); // for stability
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    if (sum == 0.0f) sum = 1.0f; // avoid division by zero
    for (int i = 0; i < len; ++i) {
        output[i] /= sum;
    }
}

/**
 * @brief Convert logits into probabilities using softmax normalization.
 *
 * @param[in,out] top_k_cls Vector of classification results to normalize in-place.
 */
void logits_to_probabilities(std::vector<Classification> &top_k_cls);
