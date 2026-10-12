// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
/// BYTETracker C++ runtime: detections -> tracks, faithful to the Python
/// tracker_backend semantics (double-precision Kalman, cython_bbox IoU
/// convention with the inclusive +1, score fusion, lap.lapjv assignment with
/// extend_cost/cost_limit, two-stage association, unconfirmed/lost/removed
/// lifecycle, duplicate removal and ID lifetime).
///
/// Two public models:
///  - bytetrack::ByteTracker  -- the CPU tracking algorithm. Accepts person
///    detections (XYXY + score, float coordinates exactly as a detector
///    emits them) and returns owned per-frame track snapshots. Stateful:
///    one ordered frame stream per instance; reset() clears history but
///    keeps the instance-local ID counter monotonic so IDs are never reused.
///  - bytetrack::ByteTrackModel -- board builds only (BYTETRACK_TARGET_X5 or
///    BYTETRACK_TARGET_S, i.e. x5/s100/s100p/s600). Named end-to-end model
///    composing the existing YOLOv5 C++ runtime (yolov5::Yolov5, unmodified)
///    with ByteTracker: preprocess -> infer -> postprocess -> predict,
///    mirroring the Python ByteTrackTask stages. Not thread safe: one video
///    stream per instance.
///
/// ID semantics vs the Python baseline: Python counts IDs on a process-global
/// BaseTrack counter; here the counter belongs to the tracker instance and
/// survives reset(). Within a single stream the sequences are identical
/// (monotonic, 1-based, no reuse); across independent instances/processes the
/// domains are independent -- compare per stream, not across instances.
#ifndef BYTETRACK_RUNTIME_CPP_TRACKING_HPP_
#define BYTETRACK_RUNTIME_CPP_TRACKING_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(BYTETRACK_TARGET_X5) || defined(BYTETRACK_TARGET_S)
// detect.hpp is the committed YOLOv5 C++ contract this model composes with.
// It is SDK-free (standard library only); the board SDK headers stay inside
// the .cpp translation units (detect.cpp / tracking.cpp board section).
#include "detect.hpp"
#endif

namespace bytetrack {

/// Source tracking defaults; frame_rate scales track_buffer/30. Thresholds
/// are double like the Python dataclass: the score *grouping* comparisons
/// deliberately round through float32 (numpy compares a float32 array to a
/// python float in float32), while the assignment cost limit and det_thresh
/// arithmetic stay double.
struct TrackingConfig {
  double track_thresh = .3;
  int track_buffer = 60;
  double match_thresh = .8;
  int frame_rate = 30;
  bool mot20 = false;
};

/// One person detection in original image pixels, XYXY + score. Coordinates
/// stay float like detector outputs: the Python baseline keeps the tlbr->tlwh
/// width/height subtraction in float32 before the Kalman filter converts to
/// double, and this port preserves that boundary.
struct Detection {
  float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
  float score = 0.f;
};

/// Owned snapshot of one output track: ID, XYXY double in original pixels,
/// score and frame index. Values are copies; later frames never mutate them.
struct Track {
  int track_id = 0;
  double tlbr[4] = {0., 0., 0., 0.};
  double score = 0.;
  int frame_id = 0;
};

/// Thrown for invalid stream inputs (NaN coordinates/scores, negative boxes).
class TrackingInputError : public std::invalid_argument {
 public:
  explicit TrackingInputError(const std::string& what) : std::invalid_argument(what) {}
};

/// Invalid configuration (out-of-range thresholds/buffer/frame rate).
class TrackingConfigError : public std::invalid_argument {
 public:
  explicit TrackingConfigError(const std::string& what) : std::invalid_argument(what) {}
};

/// The CPU BYTETracker. One ordered frame stream per instance; include empty
/// frames as empty vectors, exactly like the Python update path. The
/// constructor throws TrackingConfigError on invalid config. threads caps the
/// internal per-track Kalman / IoU parallelism (>=1, clamped to hardware);
/// results are identical for any thread count because parallel sections write
/// disjoint outputs and association stays sequential.
class ByteTracker {
 public:
  /// Validate config and build the tracker (buffer = frame_rate/30*buffer).
  explicit ByteTracker(const TrackingConfig& config = TrackingConfig{},
                       unsigned threads = 1);

  ByteTracker(const ByteTracker&) = delete;
  ByteTracker& operator=(const ByteTracker&) = delete;

  /// Advance one frame: `detections` are person boxes in original pixels
  /// (XYXY + score); img_h/img_w describe the source image the boxes live
  /// in (the Python baseline is fed the same shape twice, i.e. scale == 1).
  /// Throws TrackingInputError for non-positive image sizes, NaN
  /// coordinates/scores or zero/negative-area boxes (the Python person
  /// filter rejects zero/negative area before the tracker sees them).
  std::vector<Track> update(const std::vector<Detection>& detections, int img_h, int img_w);

  /// Current 1-based frame index (number of update calls).
  int frame_id() const;

  /// Track objects currently held (tracked + lost + removed, the persistent
  /// lifecycle lists). Monitoring/evidence hook for the lifetime tests: it
  /// must stay bounded by live+lost+cumulative-removed tracks, never by the
  /// total number of detections ever seen.
  std::size_t persistent_track_count() const;

  /// Parallel-region accounting since construction (not cleared by reset()).
  /// Honesty hook for benchmarking: a pool slot count > 1 does NOT mean the
  /// parallel paths ran — regions below the chunk thresholds (24 IoU rows /
  /// 64 predicted tracks per slot) execute inline on the caller thread, and
  /// `engaged == 0` means the run was effectively serial.
  struct ParallelUsage {
    std::size_t engaged = 0;          ///< regions executed on worker threads
    std::size_t inline_fallback = 0;  ///< regions below the chunk thresholds
    unsigned slots = 0;               ///< pool slots (0 = no pool, threads=1)
  };
  ParallelUsage parallel_usage() const;

  /// Discard stream history but keep the instance ID counter monotonic
  /// (no ID reuse across resets within one stream owner).
  void reset();

  ~ByteTracker();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;  // STrack/Kalman internals stay out of the header
};

#if defined(BYTETRACK_TARGET_X5) || defined(BYTETRACK_TARGET_S)
/// Board build: named end-to-end tracking model over the unified YOLOv5
/// detector. preprocess prepares one HWC BGR frame, infer runs the BPU
/// detector, postprocess decodes persons, filters COCO class 0 with positive
/// area and advances the tracker once; predict composes the three stages.
/// Mirrors python ByteTrackTask (source detector defaults .25/.45).
class ByteTrackModel {
 public:
  /// Model path plus detection/tracking options; see yolov5::Yolov5::Config.
  struct Options {
    std::string model_path;
    float score_thres = .25f;
    float nms_thres = .45f;
    int bpu_priority = 0;
    int bpu_core = -1;  // -1 = any core; 0..3 otherwise
    TrackingConfig tracking;
    unsigned threads = 1;
  };

  explicit ByteTrackModel(const Options& options);

  ByteTrackModel(const ByteTrackModel&) = delete;
  ByteTrackModel& operator=(const ByteTrackModel&) = delete;

  /// Source frame description shared by the three stages.
  struct Frame {
    std::vector<uint8_t> bgr;  // HWC, 3 channels, continuous
    int rows = 0, cols = 0;
  };

  /// Owned prepared context (preprocessed NV12 payload + source geometry).
  struct Prepared {
    yolov5::Yolov5::Prepared data;
  };
  /// Owned raw detector output (heads read back from this call's forward).
  struct Raw {
    yolov5::Yolov5::RawResult data;
  };

  /// Letterbox/tensor preparation (detector preprocess stage).
  Prepared preprocess(const Frame& frame) const;
  /// BPU inference (detector infer stage).
  Raw infer(const Prepared& prepared) const;
  /// Decode persons + one tracker update (detector decode + tracking stage).
  std::vector<Track> postprocess(const Raw& raw, const Prepared& prepared);
  /// Detector stages only (preprocess -> infer -> decode -> original-image
  /// mapping -> person filter): the exact detection list postprocess feeds
  /// the tracker, WITHOUT advancing it. Evidence hook for end-to-end
  /// tracing (dumping what the tracker sees) mirroring the Python sample's
  /// separation of detector stages from tracker.update.
  std::vector<Detection> detect(const Frame& frame);
  /// preprocess -> infer -> postprocess exactly once each.
  std::vector<Track> predict(const Frame& frame);

  /// Tracker stream frame index and reset, forwarded for records parity
  /// with the Python task (IDs stay monotonic across reset).
  int frame_id() const;
  void reset();

  ~ByteTrackModel();

 private:
  struct ModelImpl;
  std::unique_ptr<ModelImpl> model_impl_;
};
#endif  // board targets

}  // namespace bytetrack

#endif  // BYTETRACK_RUNTIME_CPP_TRACKING_HPP_
