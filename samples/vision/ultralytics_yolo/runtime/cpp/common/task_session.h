// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// One loaded model context: packed model, model handle, probed input plan and
// owned NV12 input tensors. A benchmark stream owns one session plus its own
// output tensors; sessions are never shared across concurrent streams.
#ifndef YOLO_COMMON_TASK_SESSION_H_
#define YOLO_COMMON_TASK_SESSION_H_
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/dnn_io.h"
#include "common/dnn_resources.h"
namespace yolo {
class TaskSession {
 public:
  TaskSession() = default;
  TaskSession(const TaskSession&) = delete;
  TaskSession& operator=(const TaskSession&) = delete;

  void initialize(const std::string& model_path) {
    if (model_) throw std::logic_error("Task session is already initialized.");
    const char* file = model_path.c_str();
    check(hbDNNInitializeFromFiles(&packed_.handle, &file, 1),
          "Cannot load model " + model_path);
    const char** names = nullptr;
    int count = 0;
    check(hbDNNGetModelNameList(&names, &count, packed_.handle),
          "Cannot list model names");
    if (count != 1 || names == nullptr || names[0] == nullptr)
      throw std::runtime_error("Expected exactly one named model.");
    name_ = names[0];
    check(hbDNNGetModelHandle(&model_, packed_.handle, names[0]),
          "Cannot get model handle");
    std::string error;
    plan_ = probe_input_protocol(model_, &error);
    if (plan_.protocol == InputProtocol::kUnknown)
      throw std::runtime_error("Unsupported model input: " + error);
    if (!input_.allocate(model_, plan_))
      throw std::runtime_error("Cannot allocate model input tensors.");
  }

  hbDNNHandle_t model() const { return model_; }
  const std::string& name() const { return name_; }
  int input_h() const { return plan_.input_h; }
  int input_w() const { return plan_.input_w; }
  bool packed_input() const {
    return plan_.protocol == InputProtocol::kPackedNv12;
  }

  // `i420` holds input_h * input_w * 3 / 2 bytes from COLOR_BGR2YUV_I420.
  void upload(const uint8_t* i420) {
    if (!input_.upload(plan_, i420))
      throw std::runtime_error("Cannot upload the preprocessed frame.");
  }

  void infer(hbDNNTensor* outputs) {
    check(infer_sync(outputs, input_.tensors(), input_.input_count(), model_),
          "Inference failed");
  }

 private:
  static void check(int rc, const std::string& message) {
    if (rc) throw std::runtime_error(message + " (error " + std::to_string(rc) + ")");
  }
  // Declared first so the model is released after the input tensors.
  PackedModelOwner packed_;
  hbDNNHandle_t model_ = nullptr;
  InputPlan plan_;
  Nv12Input input_;
  std::string name_;
};
}  // namespace yolo
#endif
