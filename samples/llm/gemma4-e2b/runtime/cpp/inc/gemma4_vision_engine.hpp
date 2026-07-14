/**
 * @file gemma4_vision_engine.h
 * @brief Vision ViT engine for Gemma4-E2B.
 *
 * Loads the Vision HBM and runs the ViT encoder to produce 280 soft image
 * tokens per image, which are injected into the text decoder's input
 * embeddings at image soft-token positions.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hobot/dnn/hb_dnn.h"

namespace gemma4 {

/**
 * @brief Vision ViT inference engine for Gemma4-E2B.
 *
 * Loads the vision HBM model, preprocesses the input image, and returns
 * the ViT output features to be injected into the text decoder.
 */
class VisionEngine {
 public:
  explicit VisionEngine(const std::string& vision_hbm);
  ~VisionEngine();

  VisionEngine(const VisionEngine&) = delete;
  VisionEngine& operator=(const VisionEngine&) = delete;

  std::vector<float> Infer(const std::string& image_path);

  double LoadMs() const { return load_ms_; }

 private:
  hbDNNPackedHandle_t packed_ = nullptr;
  hbDNNHandle_t handle_ = nullptr;
  std::vector<hbDNNTensor> inputs_;
  std::vector<hbDNNTensor> outputs_;
  double load_ms_ = 0;
};

}  // namespace gemma4
