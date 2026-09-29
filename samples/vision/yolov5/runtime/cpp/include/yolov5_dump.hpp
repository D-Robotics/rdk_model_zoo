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
#include "yolov5_gate.hpp"

#include <cstddef>
#include <cstdint>
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

// Metadata-only description of a tensor the model actually exposed. The
// layout fields make a padded run diagnosable from the manifest alone:
// aligned_byte_size and stride[] are what the runtime reported, and aligned[]
// is the alignedShape the X5 SDK reports (the S SDK has no such field, so its
// entries stay unreported). -1 serializes as null. The quant arrays record the
// complete descriptor the runtime exposed so a board run can be replayed
// without the model file: a SCALE descriptor without readable values, or one
// longer than kMaxQuantValues entries, is an error instead of a silent
// truncation.
struct DumpTensorInfo {
  std::string name;
  std::string dtype;
  std::vector<long long> shape;
  std::string quanti;  // "none", "scale", "shift" or "unknown"
  long long scale_len = 0;
  long long aligned_byte_size = -1;
  long long stride[4] = {-1, -1, -1, -1};
  long long aligned[4] = {-1, -1, -1, -1};
  long long quantize_axis = -1;
  std::vector<double> scale_values;        // empty unless quanti == "scale"
  std::vector<long long> zero_point_values;  // empty unless a descriptor exists
};

// Sanity bound for descriptor copying: descriptors above this size are
// rejected outright (the gates accept at most one value per channel of real
// heads), never silently truncated.
constexpr long long kMaxQuantValues = 1LL << 20;

// Fills a DumpTensorInfo from a projected TensorMeta using the shared
// dtype/quanti names, including the reported layout fields. For a SCALE
// tensor the full scale (and zero-point, when declared) descriptor is copied;
// a missing buffer for a declared descriptor, or a descriptor longer than
// kMaxQuantValues, throws std::invalid_argument so the run fails loudly
// instead of recording an empty array as if it were complete.
DumpTensorInfo dump_tensor_info(const std::string& name, const TensorMeta& meta,
                                const float* scale_data = nullptr,
                                const std::int32_t* zero_point_data = nullptr);

struct DumpRecord {
  std::string dir;
  std::string utc;
  std::string target;
  std::string build_target;
  std::string asset_id;
  std::string model_path;
  std::string image_path;
  std::string cwd;
  // Executable that produced this run; its SHA-256 is recorded so the dump
  // binds to the deployed binary, not only to the model and image.
  std::string binary_path;
  std::vector<std::string> argv;
  int return_code = 0;
  std::string error;
  std::vector<std::string> notes;
  std::vector<DumpTensorInfo> inputs;
  std::vector<DumpTensorInfo> outputs;
  std::vector<std::pair<std::string, std::string>> options;
  // The input buffers actually submitted with this inference (deterministic
  // payload bytes; layout lives in the inputs metadata).
  std::vector<DumpTensor> input_tensors;
  std::vector<DumpTensor> raw_tensors;
  std::vector<DumpTensor> transformed_tensors;
  std::vector<Detection> detections;
  // The same detections mapped to final ORIGINAL-image coordinates with the
  // exact arithmetic the renderer applies (independent of the fixed source's
  // own mapping; never derived from it). Compared separately from the
  // model-space detections.
  std::vector<Detection> detections_original;
};

// Writes <dir>/manifest.json and one little-endian file per dumped tensor
// under category subdirectories (input/, raw/, transformed/): the two stages
// of one output never share a file, so a later write cannot overwrite the
// original bytes of an earlier one. Returns false and fills *error on any
// filesystem failure.
bool write_dump(const DumpRecord& record, std::string* error);

// Best-effort path of the currently running executable: /proc/self/exe where
// available, otherwise argv[0] resolved against the current directory. Empty
// when neither can be determined.
std::string current_binary_path(const std::string& argv0);

// Maps model-space letterbox detections to final ORIGINAL-image coordinates
// with exactly the arithmetic the renderer applies ((coord - pad) / scale,
// unclamped). SDK-free so the mapping is host-testable.
std::vector<Detection> map_to_original(const std::vector<Detection>& detections,
                                       int image_cols, int image_rows, int model_size);

// Lowercase hex SHA-256 of a file's contents, or an empty string if unreadable.
std::string sha256_file(const std::string& path);

// Lowercase hex SHA-256 of a byte range.
std::string sha256_hex(const void* data, std::size_t size);

// Current UTC timestamp in the "%Y-%m-%dT%H:%M:%SZ" form.
std::string utc_timestamp();

}  // namespace yolov5

#endif  // RDK_MODEL_ZOO_YOLOV5_DUMP_HPP_
