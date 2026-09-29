/**
 * @file gemma4_vision_engine.cpp
 * @brief Execute the Gemma4-E2B vision HBM and return image features.
 *
 * The implementation consumes prepared patches, submits BPU inference, and
 * converts the encoder output into features consumed by the text runtime.
 *
 * @note VisionEngine instances are not thread-safe.
 */

#include "gemma4_vision_engine.hpp"

#include <chrono>
#include <stdexcept>
#include <vector>

#include "gemma4_config.hpp"
#include "gemma4_vision_debug.hpp"
#include "gemma4_vision_tensor.hpp"
#include "hb_utils.hpp"

namespace gemma4 {

VisionEngine::VisionEngine(const std::string &vision_hbm) {
  try {
    const char *path = vision_hbm.c_str();
    const auto start = std::chrono::steady_clock::now();
    HBDNN_CHECK(hbDNNInitializeFromFiles(&packed_, &path, 1),
                "load vision hbm");
    if (!packed_)
      throw std::runtime_error("SDK returned null packed model");
    load_ms_ = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count();
    HBDNN_CHECK(hbDNNGetModelHandle(&handle_, packed_, "Gemma4VisionModel"),
                "get Gemma4VisionModel");
    if (!handle_)
      throw std::runtime_error("SDK returned null model handle");
    int input_count = 0, output_count = 0;
    HBDNN_CHECK(hbDNNGetInputCount(&input_count, handle_),
                "vision input count");
    HBDNN_CHECK(hbDNNGetOutputCount(&output_count, handle_),
                "vision output count");
    if (input_count != 1 || output_count != 1)
      throw std::runtime_error(
          "Vision requires exactly one input and one output");
    hbDNNTensorProperties input{}, output{};
    HBDNN_CHECK(hbDNNGetInputTensorProperties(&input, handle_, 0),
                "get input tensor props");
    HBDNN_CHECK(hbDNNGetOutputTensorProperties(&output, handle_, 0),
                "get output tensor props");
    ValidateVisionTensor(input, true);
    ValidateVisionTensor(output, false);
    input_capacity_ = input.alignedByteSize;
    output_capacity_ = output.alignedByteSize;
    inputs_.reserve(1);
    outputs_.reserve(1);
    inputs_.push_back(AllocateTensor(input));
    outputs_.push_back(AllocateTensor(output));
  } catch (...) {
    Release();
    throw;
  }
}

VisionEngine::~VisionEngine() { Release(); }

void VisionEngine::Release() noexcept {
  FreeTensors(inputs_);
  FreeTensors(outputs_);
  if (packed_)
    hbDNNRelease(packed_);
  packed_ = nullptr;
  handle_ = nullptr;
}

std::vector<float> VisionEngine::Infer(const std::vector<float> &patches) {
  WriteVisionInput(inputs_[0], patches, input_capacity_);
  LogVisionValues("patches", patches);
  LogVisionTensor("input tensor", inputs_[0].properties);
  RunInfer(handle_, inputs_, outputs_);
  auto features = ReadVisionOutput(outputs_[0], output_capacity_);
  LogVisionTensor("output tensor", outputs_[0].properties);
  LogVisionValues("vision output", features);
  return features;
}

} // namespace gemma4
