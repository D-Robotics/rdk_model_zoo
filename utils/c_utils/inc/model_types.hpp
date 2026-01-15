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
 * @file model_types.hpp
 * @brief Define common data structures for representing model inference results.
 *
 * This file provides shared result data types used across inference,
 * postprocessing, and visualization modules.
 */

#pragma once

/**
 * @brief Top-k classification item.
 */
typedef struct Classification {
  int         id;           // Class id
  float       score;        // Logit or probability
  const char* class_name;   // Optional pointer to label string (may be null)

  Classification() : class_name(nullptr), id(0), score(0.0f) {}
  Classification(int id, float score, const char *class_name)
      : id(id), score(score), class_name(class_name) {}

  friend bool operator>(const Classification &lhs, const Classification &rhs) {
    return (lhs.score > rhs.score);
  }

  ~Classification() = default;
} Classification;

/**
 * @brief Generic detection with an axis-aligned bounding box.
 */
struct Detection {
    float bbox[4];           // x1, y1, x2, y2 (pixels, inclusive)
    float score;             // Confidence score (e.g., obj*cls)
    int   class_id;          // Argmax class id
};

/**
 * @brief A single keypoint.
 */
struct Keypoint {
    float x;                 // X coordinate (pixels)
    float y;                 // Y coordinate (pixels)
    float score;             // Raw score/logit (apply sigmoid if needed)
};
