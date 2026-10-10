// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "cif.hpp"
#include "platform_identity.h"
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace paraformer {

// --- SDK tensor contract and synchronous runner ------------------------------
enum class Stage { Encoder, Predictor, Decoder };
struct SdkModel {
  std::string path, target;
  Stage stage;
};
using SdkPreflight = std::function<void(const SdkModel &)>;
using RawTensor = std::variant<std::vector<float>, std::vector<int32_t>>;
using RawTensors = std::map<std::string, RawTensor>;
struct TensorMetadata {
  std::string name, role, dtype;
  std::vector<int> shape;
  std::vector<int64_t> strides;
  size_t allocation_bytes = 0;
};
struct SdkMetadata {
  std::string model_name;
  std::vector<TensorMetadata> inputs, outputs;
};
// Exactly one synchronous raw model call. The mandatory preflight callback must
// verify local identity and selected artifact before any SDK operation.
class SdkRunner {
public:
  SdkRunner(SdkModel, SdkPreflight);
  ~SdkRunner();
  SdkRunner(const SdkRunner &) = delete;
  SdkRunner &operator=(const SdkRunner &) = delete;
  const SdkMetadata &metadata() const;
  RawTensors infer(const RawTensors &inputs);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// --- Preflight: exact S100 publication per stage ------------------------------
struct ModelArtifact {
  SdkModel model;
  std::string asset_id, expected_sha256;
};
using ModelGroup = std::array<ModelArtifact, 3>;
inline constexpr const char *kVocabularySha256 =
    "2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127";
std::string expected_asset_id(Stage stage);
// Pure explicit-identity verifier for tests and callers with an actual identity
// snapshot. Production callers should use make_preflight to read local
// identity.
void verify_group(const ModelGroup &, const std::string &vocabulary,
                  const rdk::NativeIdentity &actual);
// Immediately verifies all three artifacts and vocabulary, before constructing
// any runner. Returned callback rechecks the group and exact selected model.
SdkPreflight make_preflight(ModelGroup, std::string vocabulary);

// --- Decode contract ----------------------------------------------------------
struct DecodedText {
  std::string text;
  std::vector<int> token_ids;
};
void validate_vocabulary(const std::vector<std::string> &vocabulary);
// Greedy valid-prefix decoding of logits[100*8404], not CTC. Ties choose the
// first ID, special <...> tokens are filtered and every @@ marker is removed.
DecodedText decode(const std::vector<float> &logits, int token_count,
                   const std::vector<std::string> &vocabulary);

// --- Pipeline model ------------------------------------------------------------
struct PredictorOutput {
  std::vector<float> weights;
  std::vector<float> hidden;
};
struct DecoderInput {
  const std::vector<float> &context;
  const std::vector<float> &acoustic;
  int32_t token_count;
  const std::array<float, 512> &bias;
};
struct Timings {
  double encoder_ms = 0;
  double predictor_ms = 0;
  double cif_ms = 0;
  std::optional<double> decoder_ms;
};
struct Prediction {
  std::string text;
  std::vector<int> token_ids;
  int32_t token_count = 0;
  bool decoder_executed = false;
  Timings timings;
};
using Encoder = std::function<std::vector<float>(const std::vector<float> &)>;
using Predictor = std::function<PredictorOutput(const std::vector<float> &)>;
using Decoder = std::function<std::vector<float>(const DecoderInput &)>;

// Three-stage model: encoder -> predictor -> CIF -> decoder, visible in
// predict. Two constructions exist: injectable stage callables (tests and
// alternative transports; each must synchronously return owned raw arrays) and
// the native construction owning all three stage runners and their observed
// metadata. SDK resources remain the runner's concern.
class Pipeline {
public:
  Pipeline(Encoder encoder, Predictor predictor, Decoder decoder,
           std::vector<std::string> vocabulary);
  Pipeline(const ModelGroup &models, const SdkPreflight &preflight,
           std::vector<std::string> vocabulary);
  // The native construction wires stage callables to runners owned by this
  // object; moving would leave them pointing at the moved-from instance.
  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;
  Pipeline(Pipeline &&) = delete;
  Pipeline &operator=(Pipeline &&) = delete;
  Prediction predict(const std::vector<float> &features,
                     int valid_frames) const;
  // Observed per-stage tensor metadata; only the native constructor owns any.
  const SdkMetadata &metadata(Stage stage) const;

private:
  std::array<std::unique_ptr<SdkRunner>, 3> native_; // native ctor only; the
                                                     // callables below borrow
                                                     // these runners.
  Encoder encoder_;
  Predictor predictor_;
  Decoder decoder_;
  std::vector<std::string> vocabulary_;
  const std::array<float, 512> bias_{};
};
} // namespace paraformer
