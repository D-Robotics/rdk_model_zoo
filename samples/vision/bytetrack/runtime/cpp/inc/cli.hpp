// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
/// ByteTrack C++ CLI surface: option parsing, detection-replay file IO,
/// records writer and (board builds) video reading + annotated writing.
/// IO and parsing only -- the tracking flow lives in tracking.hpp so main
/// stays a thin parse -> construct model -> predict entry.
#ifndef BYTETRACK_RUNTIME_CPP_CLI_HPP_
#define BYTETRACK_RUNTIME_CPP_CLI_HPP_

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tracking.hpp"

namespace bytetrack {

/// Invalid command line (the message carries the usage text).
class CliError : public std::runtime_error {
 public:
  explicit CliError(const std::string& what) : std::runtime_error(what) {}
};

struct CliOptions {
  enum class Mode { Replay, Video };
  Mode mode = Mode::Replay;
  std::string detections_path;  // replay
  std::string input_path;       // video (board builds)
  std::string model_path;       // video (board builds)
  std::string output_path;      // optional annotated video out
  std::string records_path;     // optional per-frame records TSV
  TrackingConfig tracking;
  float score_thres = .25f;
  float nms_thres = .45f;
  int max_frames = 0;  // 0 processes every frame
  int bpu_core = -1;
  int priority = 0;
  unsigned threads = 1;
};

/// Parse argv ([0] = program). Option names mirror the Python CLI's
/// synonymous options (--input/--output/--model-path/--records/...).
/// Throws CliError with usage on invalid input; on host builds `video` is
/// rejected with an explicit message.
CliOptions parse_cli(int argc, char** argv);

/// In-memory detection replay: one frame stream of person detections.
struct ReplayFile {
  uint32_t n_frames = 0;
  uint32_t img_h = 0, img_w = 0;
  std::vector<std::vector<Detection>> frames;
};

/// Read the little-endian float32 replay format (magic "BTCPP1\0\0"):
/// u32 version, u32 n_frames, u32 img_h, u32 img_w, then per frame u32 n_det
/// and n_det * (x1, y1, x2, y2, score) float32. Throws CliError on any
/// structural problem (short file, bad magic, count beyond the file end).
ReplayFile read_replay_file(const std::string& path);

/// Records TSV: header then one row per output track, doubles at %.17g.
class RecordsWriter {
 public:
  explicit RecordsWriter(const std::string& path);  // throws std::runtime_error
  ~RecordsWriter();
  RecordsWriter(const RecordsWriter&) = delete;
  RecordsWriter& operator=(const RecordsWriter&) = delete;
  void row(int frame_id, int lane, const Track& track);
  /// Flush and close; false when any write failed (success must be observed,
  /// never assumed, before a run reports its output).
  bool close();

 private:
  std::FILE* file_ = nullptr;
};

#if defined(BYTETRACK_TARGET_X5) || defined(BYTETRACK_TARGET_S)
/// Sequential video reader (OpenCV capture behind an impl pointer): fills
/// one owned ByteTrackModel::Frame per read. Throws std::runtime_error when
/// the input cannot be opened.
class VideoReader {
 public:
  explicit VideoReader(const std::string& path);
  ~VideoReader();
  VideoReader(const VideoReader&) = delete;
  VideoReader& operator=(const VideoReader&) = delete;
  /// Read the next frame into `frame` (HWC BGR bytes owned by the frame).
  /// False at end of stream.
  bool read(ByteTrackModel::Frame& frame);
  /// Source frame rate, falling back to 30 when the container reports none.
  double fps() const;

 private:
  struct ReaderImpl;
  std::unique_ptr<ReaderImpl> reader_impl_;
};

/// Optional annotated-video writer. The output opens lazily on the first
/// write with that frame's size and mp4v encoding; failure to open throws
/// (a requested output that cannot be written must fail the run, not
/// silently produce a missing file). Draws the same source ID colors and
/// labels as the Python CLI renderer.
class VideoSink {
 public:
  // Defined out-of-line in cli.cpp: an inline defaulted constructor would
  // instantiate ~unique_ptr<SinkImpl> here, where SinkImpl is still
  // incomplete (every TU including this header would fail to compile).
  VideoSink();
  ~VideoSink();
  VideoSink(const VideoSink&) = delete;
  VideoSink& operator=(const VideoSink&) = delete;
  /// Render `tracks` on the frame and append it to the output video.
  void write(const std::string& path, double fps, const ByteTrackModel::Frame& frame,
             const std::vector<Track>& tracks);

 private:
  struct SinkImpl;
  std::unique_ptr<SinkImpl> sink_impl_;
};
#endif  // board targets

}  // namespace bytetrack

#endif  // BYTETRACK_RUNTIME_CPP_CLI_HPP_
