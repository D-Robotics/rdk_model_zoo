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
 * @file file_io.cpp
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
 * @brief Load classification labels from a text file.
 *
 * Each line in the file represents one label. The line index (0-based)
 * is treated as the corresponding class ID.
 *
 * @param[in] label_path Path to the label text file.
 * @return Vector of label strings. Returns an empty vector on failure.
 */
std::vector<std::string> load_linewise_labels(const std::string& label_path)
{
    std::vector<std::string> labels;

    std::ifstream infile(label_path);
    if (!infile.is_open()) {
        std::cerr << "Failed to open label file: " << label_path << std::endl;
        return labels;
    }

    std::string line;
    while (std::getline(infile, line)) {
        // 去掉可能的 CR（Windows 文本）
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // 可选：跳过空行
        if (line.empty()) {
            continue;
        }

        labels.emplace_back(line);
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
