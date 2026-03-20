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
 * @file nn_math.cpp
 * @brief Provide lightweight neural network related math utilities.
 *
 * This file implements common mathematical helpers used during model
 * postprocessing, such as activation functions and probability normalization.
 * The implementations are designed to be simple, efficient, and reusable
 * across different model samples.
 */

#include <vector>
#include <cmath>
#include <iostream>

#include "model_types.hpp"
#include "nn_math.hpp"

/**
 * @brief Convert logits to normalized probabilities in-place using softmax (stable).
 * @param[in,out] top_k_cls   Vector of (id, score) pairs; scores are treated as logits, then replaced by probabilities.
 */
// void logits_to_probabilities(std::vector<Classification> &top_k_cls) {
//   if (top_k_cls.empty()) return;

//   // Find max for numerical stability
//   float max_logit = top_k_cls[0].score;
//   for (const auto &cls : top_k_cls) {
//     if (cls.score > max_logit) max_logit = cls.score;
//   }

//   float sum = 0.f;
//   for (auto &cls : top_k_cls) {
//     cls.score = std::exp(cls.score - max_logit); // exp-shifted
//     sum += cls.score;
//   }

//   // Normalize to probabilities
//   for (auto &cls : top_k_cls) {
//     cls.score /= sum;
//   }
// }

/**
 * @brief Convert classification logits to probabilities using stable softmax.
 *
 * This function applies softmax over the entire logit vector and outputs
 * per-class probabilities without sorting or top-k selection.
 *
 * @param[in]  tensor    Output tensor containing float logits.
 * @param[out] results   Classification results in (class_id, probability) form.
 *
 * @note
 * - Assumes tensor output type is float32 and quantiType == NONE.
 * - Dequantization is not handled here.
 */
void logits_to_probabilities(hbDNNTensor& tensor,
                             std::vector<Classification>& results)
{
    results.clear();

    auto* data = reinterpret_cast<float*>(tensor.sysMem.virAddr);
    auto quanti_type = tensor.properties.quantiType;

    if (quanti_type != hbDNNQuantiType::NONE) {
        std::cerr << "Warning: quanti_type != NONE, output may require dequantization\n";
    }

    /* -------- infer tensor length -------- */
    int tensor_len = 0;
    const auto& shape = tensor.properties.validShape;

    if (shape.numDimensions > 0) {
        tensor_len = 1;
        for (int i = 0; i < shape.numDimensions; ++i) {
            tensor_len *= shape.dimensionSize[i];
        }
    }

    if (tensor_len <= 0) {
        std::cerr << "Invalid tensor length\n";
        return;
    }

    /* -------- stable softmax -------- */
    float max_logit = data[0];
    for (int i = 1; i < tensor_len; ++i) {
        if (data[i] > max_logit) {
            max_logit = data[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < tensor_len; ++i) {
        sum_exp += std::exp(static_cast<double>(data[i] - max_logit));
    }

    if (sum_exp <= 0.0) {
        std::cerr << "Softmax sum is zero\n";
        return;
    }

    /* -------- output probabilities -------- */
    results.reserve(tensor_len);
    for (int i = 0; i < tensor_len; ++i) {
        Classification cls;
        cls.class_id = i;
        cls.probability = static_cast<float>(
            std::exp(static_cast<double>(data[i] - max_logit)) / sum_exp
        );
        results.emplace_back(cls);
    }
}

/**
 * @brief Read float tensor data directly as classification results without softmax.
 *
 * Use this function when the model output node already contains post-softmax
 * probabilities (e.g., MobileNetV2 "prob" output), so no further normalization
 * is needed.
 *
 * @param[in]  tensor    Output tensor containing float probability values.
 * @param[out] results   Classification results in (class_id, probability) form.
 *
 * @note
 * - Assumes tensor output type is float32 and quantiType == NONE.
 * - No softmax or any other transformation is applied.
 */
void tensor_to_cls_results(hbDNNTensor& tensor,
                            std::vector<Classification>& results)
{
    results.clear();

    auto* data = reinterpret_cast<float*>(tensor.sysMem.virAddr);
    auto quanti_type = tensor.properties.quantiType;

    if (quanti_type != hbDNNQuantiType::NONE) {
        std::cerr << "Warning: quanti_type != NONE, output may require dequantization\n";
    }

    /* -------- infer tensor length -------- */
    int tensor_len = 0;
    const auto& shape = tensor.properties.validShape;

    if (shape.numDimensions > 0) {
        tensor_len = 1;
        for (int i = 0; i < shape.numDimensions; ++i) {
            tensor_len *= shape.dimensionSize[i];
        }
    }

    if (tensor_len <= 0) {
        std::cerr << "Invalid tensor length\n";
        return;
    }

    /* -------- read probabilities directly -------- */
    results.reserve(tensor_len);
    for (int i = 0; i < tensor_len; ++i) {
        Classification cls;
        cls.class_id = i;
        cls.probability = data[i];
        results.emplace_back(cls);
    }
}
