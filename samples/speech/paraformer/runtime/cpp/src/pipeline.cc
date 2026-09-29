// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "pipeline.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace paraformer {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
void validate(const std::vector<float> &values, size_t count) {
  if (values.size() != count ||
      std::any_of(values.begin(), values.end(),
                  [](float value) { return !std::isfinite(value); }))
    throw std::invalid_argument("Invalid finite float32 tensor geometry");
}
} // namespace

Pipeline::Pipeline(Encoder encoder, Predictor predictor, Decoder decoder,
                   std::vector<std::string> vocabulary)
    : encoder_(std::move(encoder)), predictor_(std::move(predictor)),
      decoder_(std::move(decoder)), vocabulary_(std::move(vocabulary)) {
  if (!encoder_ || !predictor_ || !decoder_)
    throw std::invalid_argument("Each model requires a raw runner");
  validate_vocabulary(vocabulary_);
}

Prediction Pipeline::predict(const std::vector<float> &features,
                             int valid_frames) const {
  validate(features, 400 * 560);
  if (valid_frames < 1 || valid_frames > 400)
    throw std::invalid_argument("valid_frames must be in [1,400]");
  Prediction result;
  auto start = Clock::now();
  auto context = encoder_(features);
  result.timings.encoder_ms = milliseconds(start);
  validate(context, 400 * 512);
  start = Clock::now();
  auto predictor = predictor_(context);
  result.timings.predictor_ms = milliseconds(start);
  validate(predictor.weights, 401);
  validate(predictor.hidden, 401 * 512);
  start = Clock::now();
  auto acoustic = cif(predictor.weights, predictor.hidden, valid_frames);
  result.timings.cif_ms = milliseconds(start);
  result.token_count = acoustic.token_count;
  if (result.token_count == 0)
    return result;
  validate(acoustic.acoustic, 100 * 512);
  start = Clock::now();
  auto logits = decoder_(
      DecoderInput{context, acoustic.acoustic, result.token_count, bias_});
  result.timings.decoder_ms = milliseconds(start);
  auto decoded = decode(logits, result.token_count, vocabulary_);
  result.text = std::move(decoded.text);
  result.token_ids = std::move(decoded.token_ids);
  result.decoder_executed = true;
  return result;
}
} // namespace paraformer
