#ifndef RDK_MODEL_ZOO_YOLOV5_ADAPTER_HPP_
#define RDK_MODEL_ZOO_YOLOV5_ADAPTER_HPP_
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

namespace yolov5 {

struct RuntimeOptions {
  std::string model_path;
  std::string image_path;
  std::string output_path;
  std::string label_path;
  std::string target;
  // Publication fact resolved by the launcher; recorded in the evidence dump.
  std::string asset_id;
  // When non-empty the adapter writes a machine-comparable dump record here.
  std::string dump_dir;
  // Command line as received, recorded verbatim in the dump.
  std::vector<std::string> argv;
  float score_threshold = 0.25F;
  float nms_threshold = 0.45F;
  int priority = 0;
  int bpu_core = -1;
};

int run_native(const RuntimeOptions& options);

}  // namespace yolov5

#endif  // RDK_MODEL_ZOO_YOLOV5_ADAPTER_HPP_
