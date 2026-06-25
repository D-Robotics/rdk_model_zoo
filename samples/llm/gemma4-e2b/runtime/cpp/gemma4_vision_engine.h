#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hobot/dnn/hb_dnn.h"

namespace gemma4 {

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
