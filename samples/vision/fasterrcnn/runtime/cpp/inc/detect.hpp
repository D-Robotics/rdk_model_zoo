#ifndef RDK_MODEL_ZOO_FASTERRCNN_DETECT_HPP_
#define RDK_MODEL_ZOO_FASTERRCNN_DETECT_HPP_
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Faster R-CNN MobileNetV3-Large-320-FPN two-stage detector, C++ port of the
// Python runtime contract (samples/vision/fasterrcnn/runtime/python/detect.py;
// every stage is pinned to the same TorchVision semantics by the shared test
// fixtures). This header owns the task's whole SDK-free vocabulary: the anchor
// table, the exact TorchVision geometry/NMS ops, the CPU RPN selection, the
// feat0-only RoIAlign (with the reference kernel's malformed-ROI clamp), the
// PartB input layouts (X5 float32 pooled batch, S packed ROI bank) and the
// decoding tail. The board SDK headers are deliberately absent; they stay inside
// detect.cpp behind compile-time target guards, so a host without an SDK
// compiles this core and its behaviour checks directly.
//
// The model does no file IO: the CLI loads images and passes caller-owned
// BGR pixels in, and every stage returns owned per-call values.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// The shared CPU ByteTrack core from the ByteTrack sample, reused without
// modification (tracking.hpp is SDK-free when no BYTETRACK_TARGET_* macro is
// defined, which this sample never defines: the person filter below lives at
// the detector boundary, exactly like the Python tracking.py).
#include "tracking.hpp"

namespace fasterrcnn {

// ------------------------------------------------------ static model facts ----
// Captured from the pinned TorchVision checkpoint (see model/frcnn_mbv3_320.json
// and the Python detect.py, which documents the provenance).

constexpr int kInputSize = 320;       // fixed deployment canvas
constexpr int kNumProposals = 150;    // static PartB batch == rpn post-nms top-n
constexpr int kNumClasses = 91;       // TorchVision slots incl. background 0
constexpr std::int64_t kPersonLabel = 1;  // original COCO category id for person

constexpr int kRpnPreNmsTopN = 150;   // per FPN level
constexpr int kRpnPostNmsTopN = 150;
constexpr float kRpnNmsThresh = 0.7F;
constexpr float kRpnScoreThresh = 0.05F;
constexpr float kRpnMinSize = 0.001F;

constexpr float kBoxScoreThresh = 0.05F;
constexpr float kBoxNmsThresh = 0.5F;
constexpr int kBoxDetectionsPerImg = 100;
constexpr float kBoxMinSize = 1.0e-2F;

constexpr float kPoolSpatialScale = 1.0F / 32.0F;  // feat0 stride; proven exact
constexpr int kPoolOutputSize = 7;
constexpr int kPoolSamplingRatio = 2;

constexpr int kAnchorCount = 3375;    // 1500 + 1500 + 375 across three levels

/// The paired-model contract is broken (wrong artifact, layout or metadata).
class ContractError : public std::runtime_error {
 public:
  explicit ContractError(const std::string& what) : std::runtime_error(what) {}
};

/// Streaming sha256 of a file, hex-encoded (board artifacts reach ~30 MB).
std::string file_sha256(const std::string& path);

// ---------------------------------------------------------------- anchors ----

/// The ``(3375, 4)`` float32 anchor table in TorchVision order (ratio index
/// outer, size index inner, levels concatenated in feature order; cell anchors
/// rounded, centers at ``(0.5 + j) * stride``). Row-major xyxy.
const std::vector<float>& anchor_table();

// --------------------------------------------------------------- geometry ----

/// Immutable per-call letterbox geometry (mirrors the Python LetterboxContext).
/// ``scale`` is the nominal aspect-preserving factor; the resize targets
/// ``round(dim * scale)`` per axis (half-to-even), so the achieved per-axis
/// factors differ from the nominal one by rounding. Padding is bottom/right.
struct LetterboxContext {
  int original_rows = 0, original_cols = 0;
  int net_rows = 0, net_cols = 0;
  int canvas = kInputSize;
  double scale = 0.0;

  double scale_y() const { return double(net_rows) / double(original_rows); }
  double scale_x() const { return double(net_cols) / double(original_cols); }

  /// Map one canvas xyxy box to original-image pixels (divide by the achieved
  /// per-axis factor, then clamp into the source size), in place.
  void to_original(float box[4]) const;
};

/// Compute the letterbox geometry for one source image size. Throws
/// std::invalid_argument for non-positive sizes or an empty letterbox target.
LetterboxContext letterbox_geometry(int rows, int cols, int canvas = kInputSize);

/// Decode RPN/tail deltas exactly like ``BoxCoder.decode_single``. ``codes``
/// holds ``N x 4`` (RPN) or ``N x 4*K`` (tail) float32 rows; ``dw``/``dh``
/// clamp at ``log(1000/16)`` before the exponential; output is float32.
void decode_deltas(const float* anchors, const float* codes, int rows, int code_cols,
                   const float weights[4], float* out);

/// Clamp xyxy boxes into ``[0, height] x [0, width]`` (continuous TorchVision
/// semantics, full extent — never ``size - 1``). ``n`` boxes, in place.
void clip_boxes_to_image(float* boxes, int n, float height, float width);

/// Greedy NMS with TorchVision semantics: suppress IoU strictly greater than
/// the threshold; visit in descending score order (stable for ties); areas use
/// continuous coordinates without the ``+1`` convention. Returns kept indices.
std::vector<int> nms(const float* boxes, const float* scores, int n, float iou_threshold);

/// Per-group NMS via the TorchVision coordinate-offset composition.
std::vector<int> batched_nms(const float* boxes, const float* scores,
                             const std::int64_t* idxs, int n, float iou_threshold);

// ---------------------------------------------------------- CPU stage 1: RPN ----

/// Selected proposals: the first ``n_valid`` rows of the zero-padded
/// ``150 x 4`` array are the live proposals; padding rows never reach the tail.
struct MiddleResult {
  std::vector<float> proposals;  // kNumProposals * 4, xyxy, zero-padded
  int n_valid = 0;
};

/// Select at most 150 proposals, mirroring ``filter_proposals``: per-level
/// top-150 on the raw objectness logits (monotone with the sigmoid), sigmoid
/// after selection, clip, small-box removal (0.001), ``score >= 0.05`` gate,
/// per-level NMS (0.7), global top-150. Throws std::invalid_argument when the
/// PartA outputs do not match the 3375 anchors or contain non-finite values.
MiddleResult rpn_select(const float* rpn_obj, const float* rpn_deltas,
                        const float* anchors);

// ------------------------------------------------------ CPU stage 2: RoIAlign ----

/// RoIAlign one ``[C,H,W]`` float32 map with ``rois [N,4]`` (unaligned
/// TorchVision kernel semantics). Each 7x7 output bin averages
/// ``sampling_ratio^2`` bilinear samples at ``((q + 0.5) * bin / ratio)``
/// offsets; samples outside ``[-1, H]`` (or ``W``) contribute zero, bilinear
/// corners clamp to the feature border, and accumulation runs in double before
/// the float32 store — the same arithmetic as the Python port. The kernel's
/// malformed-ROI clamp applies: any ROI narrower or shorter than one feature
/// pixel pools over a forced ``1x1`` feature-pixel window. ``out`` receives
/// ``N * C * output_size * output_size`` values.
void roi_align_feat0(const float* feat, int channels, int height, int width,
                     const float* rois, int roi_count, float* out,
                     float spatial_scale = kPoolSpatialScale,
                     int output_size = kPoolOutputSize,
                     int sampling_ratio = kPoolSamplingRatio);

// ------------------------------------------------------- PartB input layouts ----

/// PartB input layout per board family.
enum class PartbLayout {
  kPacked,  // S: float32 ROI bank [1,256,1050,7]
  kPooled,  // X5: float32 pooled batch [150,256,7,7]
};

/// Re-layout ``pooled`` (``150*256*7*7`` float32, C-contiguous) into the S
/// ROI-bank tensor (``1*256*1050*7`` float32) in ``bank``.
void pack_bank(const float* pooled, float* bank);

/// Transpose the S packed outputs ``cls_packed``/``reg_packed``
/// (``1*C*150*1``) into ``150``-row ``[150,C]`` arrays (``cls``, ``reg``).
void unpack_bank_outputs(const float* cls_packed, const float* reg_packed,
                         float* cls, float* reg);

// ---------------------------------------------------------- CPU stage 3: tail ----

/// One detection: original COCO category id, score, xyxy box.
struct Detection {
  std::int64_t label = 0;
  float score = 0.0F;
  float box[4] = {0.F, 0.F, 0.F, 0.F};
};

/// Owned result: parallel label/score/box arrays (float32 coordinates).
struct DetectionResult {
  std::vector<std::int64_t> labels;
  std::vector<float> scores;
  std::vector<float> boxes;  // n * 4 xyxy
  std::size_t size() const { return labels.size(); }
};

/// Decode PartB outputs into canvas-pixel detections. Exact
/// ``postprocess_detections`` order: decode (weights 10,10,5,5), softmax,
/// clip, drop background column 0, flatten roi-major/class-minor, strict
/// ``score > 0.05`` filter, small-box removal (1e-2), per-class NMS (0.5),
/// top 100. Only the first ``n_valid`` rows participate. Throws
/// std::invalid_argument for wrong PartB extents or non-finite values.
DetectionResult tail(const float* cls, const float* reg, const float* proposals,
                     int n_valid);

// --------------------------------------------------------- descriptor table ----
// The paired-model descriptor (model/frcnn_mbv3_320.json) is the source of
// truth; this compiled-in table is its exact transcription so the C++ runtime
// can resolve default pairs and enforce the recorded digests without a JSON
// parser. The two must be updated together.

struct DescriptorRow {
  const char* target;
  const char* march;
  const char* parta_file;
  const char* parta_sha256;
  const char* partb_file;
  const char* partb_sha256;
  PartbLayout partb_layout;
  double parta_latency_us, partb_latency_us;  // compiler estimates
};

/// Every published target row (x5, s100, s100p, s600, in that order).
const std::vector<DescriptorRow>& descriptor_rows();

/// Row for one target, or nullptr when the sample does not publish it.
const DescriptorRow* find_descriptor_row(const std::string& target);

/// Default artifact directory for one target (``<sample>/model/<target>``).
std::string descriptor_model_dir(const std::string& target);

// ------------------------------------------------------------------- model ----
// The two-graph detector. One translation unit (detect.cpp) owns the whole
// task: the SDK-free stages above unconditionally, and behind compile-time
// target guards the X5 HB-DNN / S UCP sessions for both graphs — no per-board
// task files. The constructor loads both artifacts and validates the live
// tensor metadata (input/output counts, shapes, dtypes) against the published
// contract. PartB always receives float32 and quantizes on chip inside the
// graph (X5 with its recorded calibration threshold). Stage data is owned
// per call; a later call can never rewrite what an earlier one returned.
// Serial use only (one stream per instance).
class FasterRCNN {
 public:
  /// Resolved pair (mirrors the Python PairedSelection). ``parta_sha256`` /
  /// ``partb_sha256`` are the descriptor-recorded digests the run enforces;
  /// empty strings mark a custom --parta-path/--partb-path pair, which must
  /// satisfy the live SDK metadata instead of the recorded digests.
  struct Config {
    std::string target;
    std::string parta_path;
    std::string partb_path;
    PartbLayout layout = PartbLayout::kPacked;
    std::string parta_sha256;
    std::string partb_sha256;
    std::string asset_id;
  };

  /// Caller-owned source pixels: 8-bit interleaved BGR, rows * cols * 3 bytes.
  struct Input {
    std::vector<std::uint8_t> bgr;
    int rows = 0, cols = 0;
  };

  /// Owned prepared context: the target's NV12 encoding of the letterboxed
  /// canvas plus the per-call geometry (X5 fills ``nv12``; S fills ``y``/``uv``).
  struct Prepared {
    LetterboxContext context;
    std::vector<std::uint8_t> nv12;   // X5: 320*320*3/2 packed bytes
    std::vector<std::uint8_t> y;      // S: 320*320 Y plane
    std::vector<std::uint8_t> uv;     // S: 320*160 interleaved UV plane
  };

  /// Owned PartA outputs (float32, compact): 3375 objectness logits, 3375*4
  /// RPN deltas and the single consumed feature map (256*10*10).
  struct PartaOutputs {
    std::vector<float> rpn_obj;
    std::vector<float> rpn_deltas;
    std::vector<float> feat0;
  };

  explicit FasterRCNN(const Config& config);
  ~FasterRCNN();
  FasterRCNN(const FasterRCNN&) = delete;
  FasterRCNN& operator=(const FasterRCNN&) = delete;

  /// Letterbox to the 320x320 zero-padded canvas (bilinear, achieved sizes via
  /// half-to-even rounding) and encode the target's NV12 input. Pure
  /// conversion: no SDK call, no instance mutation.
  Prepared preprocess(const Input& input) const;
  /// Run PartA and return its validated float32 outputs.
  PartaOutputs infer(const Prepared& prepared) const;
  /// CPU RPN selection on PartA outputs.
  MiddleResult middle(const PartaOutputs& outputs) const;
  /// RoIAlign the first ``n_valid`` proposals on ``feat0``; the remaining
  /// static-batch rows stay exactly zero.
  std::vector<float> pool(const std::vector<float>& proposals,
                          const std::vector<float>& feat0, int n_valid) const;
  /// Middle + pool + PartB + tail + original-image mapping.
  DetectionResult postprocess(const PartaOutputs& outputs,
                              const LetterboxContext& context) const;
  /// The visible chain: preprocess -> infer -> postprocess.
  DetectionResult predict(const Input& input);

  /// Proposals surviving the RPN of the last postprocess/predict (0 before
  /// the first call); the Python runtime reports the same count.
  int last_proposals() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// -------------------------------------------------------- person tracking ----
// Person tracking around the detector, mirroring the Python tracking.py: the
// shared ByteTrack core is reused without modification and the only adaptation
// lives at this detector boundary — original COCO ids mean person is label 1,
// filtered explicitly (with positive area) before the tracker. One ordered
// frame stream per instance; reset() discards history but keeps the tracker's
// id counter monotonic (no id reuse within one stream owner). Not thread safe.
class PersonTracker {
 public:
  /// ByteTrack source defaults as pinned by the Python sample (buffer 30).
  struct Config {
    bytetrack::TrackingConfig tracking{0.3, 30, 0.8, 30, false};
  };

  /// ``detector`` must outlive this instance.  ``config`` has no in-class
  /// default argument on purpose: the nested Config carries NSDMIs and a
  /// default-argument ``Config()`` inside this class is not parseable before
  /// the class is complete; callers pass their Config explicitly.
  PersonTracker(FasterRCNN& detector, const Config& config, unsigned threads = 1);

  /// Detector stages without touching tracker state.
  FasterRCNN::Prepared preprocess(const FasterRCNN::Input& input) const;
  FasterRCNN::PartaOutputs infer(const FasterRCNN::Prepared& prepared) const;

  /// Decode detections, drop non-person labels and zero-area boxes, then
  /// advance the tracker exactly once (including on empty frames).
  std::vector<bytetrack::Track> postprocess(const FasterRCNN::PartaOutputs& outputs,
                                            const LetterboxContext& context);

  /// preprocess -> infer -> postprocess.
  std::vector<bytetrack::Track> predict(const FasterRCNN::Input& input);

  /// Detector stages only, mapped to original pixels and person-filtered:
  /// the exact detection list postprocess feeds the tracker, WITHOUT
  /// advancing it (mirrors the Python sample's ``detect_persons``).
  std::vector<Detection> detect_persons(const FasterRCNN::Input& input) const;

  /// The wrapped detector (per-frame proposal counts are read through it,
  /// mirroring the Python task's public ``detector`` attribute).
  FasterRCNN& detector() const { return *detector_; }

  int frame_id() const;
  /// Discard stream history; detector and id monotonicity stay.
  void reset();

  /// Tracker-only wall time (microseconds) of the last postprocess call.
  double last_tracker_us() const;

 private:
  FasterRCNN* detector_;
  Config config_;
  std::unique_ptr<bytetrack::ByteTracker> tracker_;
  double last_tracker_us_ = 0.0;
};

}  // namespace fasterrcnn

#endif  // RDK_MODEL_ZOO_FASTERRCNN_DETECT_HPP_
