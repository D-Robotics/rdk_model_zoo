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
 * @file file_io.cc
 * @brief Provide common file input utilities for loading images, labels, and related resources.
 *
 * This file implements a set of generic file I/O helpers used by samples and runtime code.
 * It focuses on reading external data (such as images and label files) into in-memory
 * representations required by the inference pipeline.
 */

#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>
#include <opencv2/imgcodecs.hpp>

#include "file_io.hpp"

/**
 * @brief Load an image from disk in BGR color space.
 *
 * @param[in] image_file Absolute or relative path to the image file.
 * @return cv::Mat Loaded image in BGR (empty if loading fails).
 */
cv::Mat load_bgr_image(const std::string& image_file) {
    // Read as 3-channel BGR image
    cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);

    // Warn if failed
    if (bgr_mat.empty()) {
        std::cerr << "ERROR: Failed to load image: " << image_file << std::endl;
    }
    return bgr_mat;
}

/**
 * @brief Load an ImageNet-1000 label map dumped as "{123: 'label'}" text.
 *
 * Expected file format (single object): {0: 'label0', 1: 'label1', ...}
 *
 * @param[in] label_path Path to label file.
 * @return std::map<int,std::string> Mapping from class id to label string. Returns empty map on failure.
 */
std::map<int, std::string> load_imagenet1000_label_map(const std::string &label_path) {
  std::map<int, std::string> label_map;

  std::ifstream infile(label_path);
  if (!infile.is_open()) {
    std::cerr << "Failed to open label file: " << label_path << std::endl;
    return label_map;  // empty
  }

  // Read entire file into a string
  std::string content((std::istreambuf_iterator<char>(infile)),
                      std::istreambuf_iterator<char>());
  infile.close();

  // Trim possible leading '{' and trailing '}'
  if (!content.empty() && content.front() == '{') content.erase(0, 1);
  if (!content.empty() && content.back() == '}') content.pop_back();

  // Regex to match: "<spaces><id><spaces>:<spaces>'<label>'"
  std::regex entry_regex(R"(\s*(\d+)\s*:\s*'([^']*)')");
  auto begin = std::sregex_iterator(content.begin(), content.end(), entry_regex);
  auto end   = std::sregex_iterator();

  // Parse all matches
  for (auto it = begin; it != end; ++it) {
    int id = std::stoi((*it)[1].str());     // group 1: numeric id
    std::string label = (*it)[2].str();     // group 2: label text
    label_map[id] = label;
  }

  return label_map;
}

/**
 * @brief Load labels line-by-line from a plain text file.
 *
 * Each non-empty line is considered one label. Trailing '\r' is stripped.
 *
 * @param[in] filename Path to the label text file.
 * @return std::vector<std::string> All labels in order of appearance.
 */
std::vector<std::string> load_linewise_labels(const std::string& filename)
{
    std::vector<std::string> labels;
    std::ifstream infile(filename);
    std::string line;

    if (!infile.is_open()) {
        std::cerr << "Failed to open label file: " << filename << std::endl;
        return labels; // empty
    }

    while (std::getline(infile, line)) {
        if (!line.empty()) {
            // Remove trailing '\r' if present (Windows line endings)
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            labels.push_back(line);
        }
    }
    return labels;
}

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
std::vector<std::string> load_id2token(const std::string& vocab_file)
{
    std::ifstream ifs(vocab_file);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open vocab file: " + vocab_file);
    }

    json j;
    ifs >> j;  // Parse JSON to object

    // Find maximum id to size the vector
    int max_id = -1;
    for (auto it = j.begin(); it != j.end(); ++it) {
        int id = it.value().get<int>();
        if (id > max_id) max_id = id;
    }

    // Allocate id->token vector
    std::vector<std::string> id2token(max_id + 1);

    // Reverse the mapping: token->id → id->token
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& token = it.key();
        int id = it.value().get<int>();
        if (id >= 0 && id <= max_id) {
            id2token[id] = token;
        }
    }

    return id2token;
}
