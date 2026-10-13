// HobotMot (IOU 2.0) multi-object tracker sample.
//
// Wraps the pinned upstream D-Robotics/mot wrappers (HobotMot / MOTMethod /
// IOU2) on the device CPU. Detections go in, per-detection track identities
// come out; no detector and no ROS node is involved. The upstream tracker is
// class-agnostic, so one upstream instance is kept per class id and identities
// are unique only within their class namespace.
#ifndef RDK_MODEL_ZOO_HOBOT_MOT_TRACK_HPP_
#define RDK_MODEL_ZOO_HOBOT_MOT_TRACK_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hobot_mot_sample {

/// Upstream output states that can appear on a per-detection result.
enum class TrackState : int32_t {
  kValid = 0,    // matched to a track; id and box come from the tracker
  kFiltered = 1,
  kInvisible = 2,
  kDisappeared = 3,
  kInvalid = 4,  // not matched in this frame; id is -1
};

/// One input detection in image pixel coordinates (x1, y1) to (x2, y2).
struct Detection {
  float x1 = 0.F;
  float y1 = 0.F;
  float x2 = 0.F;
  float y2 = 0.F;
  float score = 1.F;
  int32_t class_id = 0;
};

/// One output row, aligned with the input detection at the same index.
struct TrackResult {
  int32_t x1 = 0;
  int32_t y1 = 0;
  int32_t x2 = 0;
  int32_t y2 = 0;
  float score = 1.F;
  int32_t class_id = 0;
  int32_t track_id = -1;  // -1 when TrackState is not kValid
  TrackState state = TrackState::kInvalid;
};

/// A track identity removed by the tracker in the current frame.
struct DisappearedId {
  int32_t class_id = 0;
  int32_t track_id = -1;
};

/// Input (business data) plus the geometry of this call.
struct PreparedFrame {
  std::vector<Detection> detections;  // validated copy, input order kept
  int64_t timestamp_ms = 0;           // strictly increasing per tracker
  int32_t frame_width = 0;
  int32_t frame_height = 0;
};

/// RawOutputs: upstream output for every class instance fed this frame.
/// Rows are ordered like the detections of that class; the merged input order
/// is rebuilt by postprocess.
struct RawMotFrame {
  struct ClassOutput {
    int32_t class_id = 0;
    std::vector<TrackResult> rows;              // one per class detection
    std::vector<int32_t> detection_indexes;     // input index of each row
    std::vector<DisappearedId> gone;        // ids removed this frame
  };
  std::vector<ClassOutput> classes;
};

/// Result (business output) of one frame.
struct FrameTracks {
  int64_t timestamp_ms = 0;
  std::vector<TrackResult> tracks;        // same order as input detections
  std::vector<DisappearedId> removed_ids; // tracks deleted this frame
};

/// Raised for invalid configuration or input before upstream code runs.
class MotConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class MotInputError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Configuration and frame geometry defaults for one tracker instance.
struct TrackOptions {
  /// Path of a validated IOU_2.0 JSON config; validated by
  /// validate_config_file() before any upstream object is constructed.
  std::string config_path;
  int32_t frame_width = 960;
  int32_t frame_height = 544;
};

/// Validate a config file and return its content.
/// Throws MotConfigError unless the file parses as JSON and every field the
/// upstream IOU2Param actually reads carries the JSON type upstream requires
/// (upstream silently keeps defaults for mistyped fields, and constructs no
/// tracker for unknown tracker_type values, which crashes DoProcess).
std::string validate_config_file(const std::string &config_path);

/// Multi-object tracker over detection sequences.
/// One instance per video stream; calls are not thread-safe (upstream
/// serializes internally, but per-class routing here is not locked).
class HobotMotTracker {
 public:
  explicit HobotMotTracker(const TrackOptions &options);

  /// Validate detections and frame geometry. Throws MotInputError on
  /// non-finite or non-int32-representable coordinates, non-positive box
  /// extents, out-of-range scores or class ids, non-positive frame size or a
  /// timestamp that does not strictly increase.
  PreparedFrame preprocess(const std::vector<Detection> &detections,
                           int64_t timestamp_ms, int32_t frame_width,
                           int32_t frame_height) const;

  /// Run the upstream tracker for every class present in this frame (and for
  /// classes with live tracks when the frame carries none of them). Re-runs
  /// the full input gate, so staged calls cannot bypass preprocess
  /// validation or replay a stale timestamp; also rejects a config file that
  /// changed since construction before any native state is touched.
  RawMotFrame infer(const PreparedFrame &prepared);

  /// Merge per-class upstream output back into input order. Throws
  /// std::runtime_error when raw rows/indexes are inconsistent with the
  /// prepared frame.
  FrameTracks postprocess(const RawMotFrame &raw,
                          const PreparedFrame &prepared) const;

  /// preprocess -> infer -> postprocess.
  FrameTracks predict(const std::vector<Detection> &detections,
                      int64_t timestamp_ms, int32_t frame_width,
                      int32_t frame_height);

  /// Drop all state: class instances are destroyed and rebuilt with the same
  /// config, so track ids restart from 1 (upstream has no in-place reset and
  /// never reuses an id inside one instance).
  void reset();

 private:
  class UpstreamMot;
  void validate_frame(const PreparedFrame &prepared) const;
  void ensure_config_unchanged() const;
  TrackOptions options_;
  std::string config_content_;  // validated snapshot taken at construction
  int64_t last_timestamp_ms_ = -1;
  std::map<int32_t, std::shared_ptr<UpstreamMot>> per_class_;
};

}  // namespace hobot_mot_sample

#endif  // RDK_MODEL_ZOO_HOBOT_MOT_TRACK_HPP_
