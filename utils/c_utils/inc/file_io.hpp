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
 * @file file_io.hpp
 * @brief Provide reusable interfaces for file-based resource loading.
 *
 * This file declares common utilities for loading external resources
 * required at runtime, such as images, labels, or other data files.
 */

#pragma once

#include <map>
#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * @brief Load an image from disk in BGR color space.
 *
 * @param[in] image_file Absolute or relative path to the image file.
 * @return cv::Mat Loaded image in BGR (empty if loading fails).
 */
cv::Mat load_bgr_image(const std::string& image_file);

/**
 * @brief Load an ImageNet-1000 label map dumped as "{123: 'label'}" text.
 *
 * Expected file format (single object): {0: 'label0', 1: 'label1', ...}
 *
 * @param[in] label_path Path to label file.
 * @return std::map<int,std::string> Mapping from class id to label string. Returns empty map on failure.
 */
std::map<int, std::string> load_imagenet1000_label_map(const std::string &label_path);

/**
 * @brief Load labels line-by-line from a plain text file.
 *
 * Each non-empty line is considered one label. Trailing '\r' is stripped.
 *
 * @param[in] filename Path to the label text file.
 * @return std::vector<std::string> All labels in order of appearance.
 */
std::vector<std::string> load_linewise_labels(const std::string& filename);

/**
 * @brief Load a token->id vocabulary from JSON and produce id->token table.
 *
 * The JSON is expected to be an object: {"<token>": <id>, ... }.
 *
 * @param[in] vocab_file Path to JSON vocabulary file.
 * @return std::vector<std::string> Vector where index is token id and value is token string.
 *
 * @throws std::runtime_error If the file cannot be opened or parsed.
 */
std::vector<std::string> load_id2token(const std::string& vocab_file);
