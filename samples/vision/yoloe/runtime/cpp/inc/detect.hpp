// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "platform_identity.h"
#include "yolo.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace yoloe {

// Canvas geometry of the 640x640 letterbox/stretch preprocessing and the
// mapping back to source pixels. E26 rounds resized sides to even values.
enum class Protocol { E11, E26 };
struct Geometry {
  int width, height, resized_w, resized_h, left, top, right, bottom,
      resize_type;
  Protocol protocol;
};
int round_even(double x);
Geometry make_geometry(int width, int height, Protocol protocol,
                      int resize_type = 1);
void validate_geometry(const Geometry &g);
std::array<float, 4> restore_box(const std::array<float, 4> &box,
                                 const Geometry &g);

struct Config {
  Protocol protocol = Protocol::E11;
  float score_threshold = 0.25f;
  std::optional<float> nms_threshold;
  int max_det = 300;
  bool single_label = true;
  bool do_morph = false;
  int resize_type = 1;
};
void validate_config(const Config &cfg);

// Compact NV12 planes handed to a backend; owned per call by the caller.
struct Nv12Input {
  std::vector<uint8_t> y, uv;
};
Nv12Input to_nv12(const cv::Mat &pixels);

// Owned model-canvas candidate. Geometry/mask restoration is a separate stage.
struct RawDetection {
  std::array<float, 4> box{};
  float score = 0;
  int label = 0;
  std::array<float, 32> coefficients{};
};

// Ten compact semantic FLOAT32 heads, ordered cls/box/coefficients at strides
// 8/16/32, then the 160x160x32 mask prototype.
using Heads = std::array<std::vector<float>, 10>;
// Backend owns model/tensor resources. Return independent compact semantic
// FLOAT32 heads; no borrowed SDK buffers may escape infer(). SDK metadata and
// hardware/artifact identity validation belong to the concrete backend.
class Runner {
public:
  virtual ~Runner() = default;
  virtual Protocol protocol() const = 0;
  virtual Heads infer(const Nv12Input &input) = 0;
};

// Native model identity policy: which target/variant pairs have a runnable
// float-output deployment.
struct SdkModel {
  std::string path, target, variant;
};
inline bool supported_native_model(const SdkModel &model) {
  const bool e11 = model.variant == "11s" || model.variant == "11m" ||
                   model.variant == "11l";
  const bool e26 = model.variant == "26n" || model.variant == "26s" ||
                   model.variant == "26m" || model.variant == "26l" ||
                   model.variant == "26x";
  return (model.target == "x5" && e11) ||
         (model.target == "s100" && (model.variant == "11s" || e26)) ||
         (model.target == "s100p" && e26);
}
// Required policy boundary: verify actual board identity, exact selected asset
// or custom float SHA-256 and vocabulary/conversion provenance. Runs before
// any SDK call. This low-level adapter does not supply a publication resolver.
using SdkPreflight = std::function<void(const SdkModel &)>;
constexpr const char *kVocabularySha256 =
    "1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3";
// Pure observation seam for host tests. Production factory always reads local
// sysfs/device-tree; there is no CLI/environment override for board identity.
void verify_preflight(const SdkModel &model, const std::string &expected_sha256,
                      const std::string &labels,
                      const rdk::NativeIdentity &actual);
// Observed/custom digest verifies bytes, not publisher origin or compiler
// provenance.
SdkPreflight make_preflight(std::string expected_sha256, std::string labels);

class YOLOE;
struct StageIdentity {};
class Prepared {
public:
  const Nv12Input &input() const { return input_; }
  const Geometry &geometry() const { return geometry_; }

private:
  friend class YOLOE;
  Prepared(Nv12Input input, Geometry geometry,
           std::shared_ptr<const StageIdentity> owner)
      : input_(std::move(input)), geometry_(geometry),
        owner_(std::move(owner)) {}
  Nv12Input input_;
  Geometry geometry_;
  std::shared_ptr<const StageIdentity> owner_;
};
class RawBatch {
public:
  const Heads &outputs() const { return outputs_; }
  const Geometry &geometry() const { return geometry_; }

private:
  friend class YOLOE;
  RawBatch(Heads outputs, Geometry geometry,
           std::shared_ptr<const StageIdentity> owner)
      : outputs_(std::move(outputs)), geometry_(geometry),
        owner_(std::move(owner)) {}
  Heads outputs_;
  Geometry geometry_;
  std::shared_ptr<const StageIdentity> owner_;
};
struct Instance {
  std::array<float, 4> box;
  float score;
  int label;
  cv::Mat mask;
};
using Result = std::vector<Instance>;

// Preprocessed 640x640 BGR canvas plus its geometry.
struct PreparedBGR {
  cv::Mat pixels;
  Geometry geometry;
};
PreparedBGR prepare_bgr(const cv::Mat &image, Protocol protocol,
                        int resize_type = 1);
struct RestoredMask {
  std::array<float, 4> box;
  cv::Mat mask;
};
// E26 combines the 160x160x32 prototype with per-detection coefficients into
// canvas logits, thresholds inside the letterboxed box and rescales the ROI to
// source pixels.
std::vector<RestoredMask>
restore_e26_masks(const std::vector<RawDetection> &candidates,
                  const std::vector<float> &proto, const Geometry &geometry);
// S E11 uses a cropped prototype binary mask, unlike E26 canvas logits.
std::vector<RestoredMask>
restore_e11_masks(const std::vector<RawDetection> &candidates,
                  const std::vector<float> &proto, const Geometry &geometry,
                  bool do_morph = false);

namespace detail {
float candidate_iou(const RawDetection &a, const RawDetection &b);
// Candidates have already passed finite DFL decoding. Ties are deterministic
// by input index.
std::vector<RawDetection> nms_e11(const std::vector<RawDetection> &candidates,
                                  float threshold);
} // namespace detail

// E11: 4585-class logits with DFL64 distances; score >= threshold, suppress
// IoU > NMS, per class.
std::vector<RawDetection>
decode_e11(const std::array<std::vector<float>, 10> &outputs,
           float score_threshold = 0.25f, float nms_threshold = 0.7f);
// E26 (raw-v1): direct LTRB distances, strict top-K over anchors (optionally
// expanded to multiple classes per anchor), no NMS.
std::vector<RawDetection>
decode_e26(const std::array<std::vector<float>, 10> &outputs,
           float threshold = 0.25f, int max_det = 300,
           bool single_label = true);

// Semantic order: (class, box, coefficients) at strides 8/16/32, prototype.
// Physical output order is deliberately not part of the contract. This only
// binds logical roles; use nhwc_float_plan/copy_float_output at the SDK
// boundary to require unquantized FLOAT32 and validate physical
// strides/allocation.
std::array<int, 10> bind_heads(const std::vector<yolo::OutputShape> &shapes,
                               int box_channels);

Result decode_result(const Heads &heads, const Geometry &geometry,
                     const Config &cfg);

// Board DNN adapter over the ultralytics_yolo shared backend; compiled only
// in SDK builds (YOLOE_HAS_SDK) and link-time replaced by host fixtures
// otherwise.
class SdkRunner final : public Runner {
public:
  SdkRunner(SdkModel model, SdkPreflight preflight);
  ~SdkRunner() override;
  SdkRunner(const SdkRunner &) = delete;
  SdkRunner &operator=(const SdkRunner &) = delete;
  Protocol protocol() const override;
  Heads infer(const Nv12Input &input) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// One task owns one backend. No file I/O, rendering, implicit download or
// hardware fallback. The injected backend must implement the declared protocol.
class YOLOE {
public:
  YOLOE(Config config, std::unique_ptr<Runner> runner);
  // Native path: builds the board SDK adapter itself behind the supplied
  // preflight gate; the config protocol must match the selected model.
  YOLOE(SdkModel model, Config config, SdkPreflight preflight)
      : YOLOE(config, std::make_unique<SdkRunner>(std::move(model),
                                                  std::move(preflight))) {}
  YOLOE(const YOLOE &) = delete;
  YOLOE &operator=(const YOLOE &) = delete;
  Prepared preprocess(const cv::Mat &image) const;
  RawBatch infer(const Prepared &input);
  Result postprocess(const RawBatch &raw) const;
  Result predict(const cv::Mat &image);

private:
  Config config_;
  std::unique_ptr<Runner> runner_;
  std::shared_ptr<const StageIdentity> identity_ =
      std::make_shared<StageIdentity>();
};
} // namespace yoloe
