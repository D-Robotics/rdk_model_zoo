#ifndef RDK_MODEL_ZOO_YOLOV5_DUMP_HPP_
#define RDK_MODEL_ZOO_YOLOV5_DUMP_HPP_
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Machine-comparable native evidence dump. This module is independent of the
// visualizer: rendering a picture is not evidence, so the adapters write the
// raw and transformed tensors, metadata, parameters and return code here with a
// content hash binding the run to its deployment and input files.

#include "yolov5_decode.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace yolov5 {

// One tensor payload plus the shape/dtype needed to read it back.
struct DumpTensor {
  std::string name;
  std::string dtype;  // "float32" or "int32"
  std::vector<long long> shape;
  std::vector<unsigned char> bytes;
};

// Metadata-only description of a tensor the model actually exposed.
struct DumpTensorInfo {
  std::string name;
  std::string dtype;
  std::vector<long long> shape;
  std::string quanti;  // "none", "scale", "shift" or "unknown"
  long long scale_len = 0;
};

struct DumpRecord {
  std::string dir;
  std::string utc;
  std::string target;
  std::string build_target;
  std::string asset_id;
  std::string model_path;
  std::string image_path;
  std::string cwd;
  std::vector<std::string> argv;
  int return_code = 0;
  std::string error;
  std::vector<std::string> notes;
  std::vector<DumpTensorInfo> inputs;
  std::vector<DumpTensorInfo> outputs;
  std::vector<std::pair<std::string, std::string>> options;
  std::vector<DumpTensor> raw_tensors;
  std::vector<DumpTensor> transformed_tensors;
  std::vector<Detection> detections;
};

// Writes <dir>/manifest.json and one raw little-endian file per dumped tensor.
// Returns false and fills *error on any filesystem failure.
bool write_dump(const DumpRecord& record, std::string* error);

// Lowercase hex SHA-256 of a file's contents, or an empty string if unreadable.
std::string sha256_file(const std::string& path);

// Lowercase hex SHA-256 of a byte range.
std::string sha256_hex(const void* data, std::size_t size);

// Current UTC timestamp in the "%Y-%m-%dT%H:%M:%SZ" form.
std::string utc_timestamp();

}  // namespace yolov5

#endif  // RDK_MODEL_ZOO_YOLOV5_DUMP_HPP_
