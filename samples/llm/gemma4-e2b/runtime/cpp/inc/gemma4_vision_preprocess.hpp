/**
 * @file gemma4_vision_preprocess.hpp
 * @brief Image preprocessing for Gemma4-E2B's Vision ViT.
 *
 * Loads an image from disk, resizes it to the ViT's expected resolution,
 * normalizes pixel values to [0, 1], and patchifies it into the
 * [num_patches, 3 * patch_size * patch_size] tensor format consumed by the
 * vision engine.
 */
#pragma once

#include <string>
#include <vector>

namespace gemma4 {

/**
 * @brief Load and transform an image into the vision HBM input tensor.
 * @param image_path Path to an image readable by OpenCV.
 * @return Flattened normalized patch tensor expected by the vision model.
 * @throws std::runtime_error If the image cannot be loaded or transformed.
 */
std::vector<float> PreprocessImage(const std::string& image_path);

}  // namespace gemma4
