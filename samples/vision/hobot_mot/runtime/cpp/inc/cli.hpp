// CLI for the HobotMot replay runner: argument parsing, detection-sequence
// loading, rendering and result serialization. No tracker or SDK types here;
// see inc/track.hpp for the model contract.
#ifndef RDK_MODEL_ZOO_HOBOT_MOT_CLI_HPP_
#define RDK_MODEL_ZOO_HOBOT_MOT_CLI_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "track.hpp"

namespace hobot_mot_sample {

/// Parsed command line (names match the Python CLI of this sample).
struct Options {
  std::string input_path;    // detection JSONL; default test_data sequence
  std::string config_path;   // IOU_2.0 tracker config
  std::string output_path;   // optional JSON result file
  int32_t frame_width = 960; // used when a frame record has no image size
  int32_t frame_height = 544;
};

/// One parsed JSONL record: a frame of detections.
struct FrameRecord {
  int64_t frame_index = 0;
  int64_t timestamp_ms = 0;
  int32_t image_width = 0;   // 0 -> use Options defaults
  int32_t image_height = 0;
  std::vector<Detection> detections;
};

/// Parse argv. Throws std::runtime_error with a readable message on unknown
/// flags, missing values or non-numeric values (full-consumption checks).
Options parse_options(const std::vector<std::string> &args);

/// Load and structurally validate a detection JSONL file. Throws
/// std::runtime_error naming the file and line of the first bad record.
std::vector<FrameRecord> load_frame_records(const std::string &input_path);

/// Human-readable one-line-per-frame summary to stdout.
void render_frames(const std::vector<FrameTracks> &results,
                   const std::vector<FrameRecord> &records);

/// Write full results as JSON (frames -> tracks and disappeared ids).
/// Throws std::runtime_error on IO failure.
void save_results(const std::string &output_path,
                  const std::vector<FrameRecord> &records,
                  const std::vector<FrameTracks> &results);

}  // namespace hobot_mot_sample

#endif  // RDK_MODEL_ZOO_HOBOT_MOT_CLI_HPP_
