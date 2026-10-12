#ifndef RDK_MODEL_ZOO_FASTERRCNN_CLI_HPP_
#define RDK_MODEL_ZOO_FASTERRCNN_CLI_HPP_
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Faster R-CNN C++ CLI surface: option parsing, pair selection over the
// compiled-in descriptor table, digest verification, image/video IO,
// rendering and the JSON run reports. Option names, defaults, validation
// and report keys mirror the Python CLI (runtime/python/cli.py + main.py)
// so both runtimes report identical keys for the same run. IO and parsing only
// — the pipeline stages stay in detect.hpp and main.cpp stays a thin
// parse -> construct named model -> predict entry.
//
// Host builds (no board SDK) keep the model-free paths: help, --list-models
// and --dry-run. Image and video inference require a board target build
// (FRCNN_TARGET x5/s100/s100p/s600), exactly like the Python runtime
// requires the board SDK.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Same mapping detect.cpp applies: a board target macro selects the board
// build for this translation unit too (CMake defines FRCNN_TARGET_X5 /
// FRCNN_TARGET_S plus FRCNN_TARGET_NAME and the SOC_* string for cli.cpp and
// main.cpp exactly like for detect.cpp).
#if defined(FRCNN_TARGET_X5) || defined(FRCNN_TARGET_S)
#if !defined(FRCNN_BOARD)
#define FRCNN_BOARD 1
#endif
#endif

#include "detect.hpp"

namespace fasterrcnn {

/// Invalid command line, selection or label file (message carries the usage
/// text where applicable). Maps to the Python CLI's BindingError/ValueError
/// surface and exit code 2.
class CliError : public std::runtime_error {
 public:
  explicit CliError(const std::string& what) : std::runtime_error(what) {}
};

struct CliOptions {
  std::string target = "auto";     // auto|x5|s100|s100p|s600
  std::string parta_path;          // explicit pair half (both or neither)
  std::string partb_path;
  std::string input_path;          // image (or video with --video/--track)
  std::string video_path;          // non-empty enables tracking
  bool track = false;
  int reset_every = 0;             // 0 -> not given (never reset)
  bool reset_every_given = false;  // distinguishes "--reset-every 0" from none
  std::string records_path;        // per-frame track records TSV
  std::string save_path;           // annotated output path
  bool no_encode = false;          // skip the annotated writer (timing only)
  int threads = 0;                 // 0 -> leave OpenCV's default; else 1/2/4
  std::string label_file;
  bool list_models = false;
  bool dry_run = false;
};

/// Parse argv ([0] = program). Defaults mirror the Python parser: input is
/// the sample's test_data/bus.jpg, labels the 91-slot COCO file, save path
/// test_data/result.jpg. Throws CliError (usage attached) on invalid input;
/// on host builds image/video inference is rejected with an explicit reason.
CliOptions parse_cli(int argc, char** argv);

/// The running board's target id from the OS identity files, or "" when
/// nothing known is found — the same table and precedence as
/// utils/py_utils/platforms.py (docs/release/platforms.json transcribed).
std::string detect_board_target();

/// Resolved pair (mirrors cli.py resolve_selection + PairedSelection).
/// Recorded digests are empty strings for a custom --parta-path/--partb-path
/// pair, which must satisfy the live SDK metadata instead of the recorded
/// digests.
struct ResolvedSelection {
  std::string target, march;
  std::string parta_path, partb_path;
  std::string parta_sha256, partb_sha256;  // recorded digests; "" -> custom
  PartbLayout layout = PartbLayout::kPacked;
  std::string asset_id;
  bool custom_pair = false;
};

/// Resolve the target ("auto" reads the board identity) and its descriptor
/// row. Throws CliError for an unknown target, an unidentifiable board under
/// auto, or a half-specified custom pair.
ResolvedSelection resolve_selection(const CliOptions& options);

/// Observed artifact digests for the run report. ``*_recorded`` is false for
/// a custom pair (the JSON reports null, like the Python None).
struct HashReport {
  std::string parta_sha256, partb_sha256;
  bool parta_recorded = false, partb_recorded = false;
  bool parta_matches = false, partb_matches = false;
};

/// Hash both artifacts and enforce the descriptor digests for descriptor
/// pairs (a stale artifact under a published name fails instead of running
/// unverified bytes). Throws CliError / std::runtime_error on mismatch or
/// unreadable files — the Python CLI raises BindingError/OSError for the
/// same conditions.
HashReport verify_pair_hashes(const ResolvedSelection& selection);

/// Print the descriptor pairs with local file presence (target "" or "auto"
/// lists all; otherwise one row). Returns the process exit code (0).
int list_models(const std::string& target);

/// Print the resolved pair without loading any model. Exit code 0.
int dry_run(const ResolvedSelection& selection);

/// Load the 91-slot label file (line i names label i). Throws CliError on
/// any other line count — the label ids index the 91 TorchVision slots.
std::vector<std::string> load_labels(const std::string& path);

#if defined(FRCNN_BOARD)
// ---- board builds: image/video IO, rendering, the tracking loop ------------

/// Read one BGR image into the model's caller-owned input. Throws
/// std::runtime_error when the file is missing or not decodable.
FasterRCNN::Input load_image(const std::string& path);

/// One image-mode report payload: the detection arrays plus the written path
/// (the JSON writer mirrors the Python report keys from these values).
struct ImagePayload {
  std::vector<float> boxes;      // n * 4 xyxy, original-image pixels
  std::vector<float> scores;     // n
  std::vector<std::int64_t> labels;  // n original COCO ids
  std::string result_path;
};

/// Draw one result (Python render_detections palette: (0,215,255) boxes,
/// filled label chip, black text) and write it; parent directories are
/// created as needed. Throws std::runtime_error when the write fails.
ImagePayload save_detection_image(const FasterRCNN::Input& image,
                                  const DetectionResult& result,
                                  const std::string& label_file,
                                  const std::string& save_path);

/// Per-run measured video summary; the JSON writer emits exactly the Python
/// track_video keys from these values. All times in milliseconds. An empty
/// result_path reports null (writer skipped via --no-encode).
struct VideoSummary {
  std::string video, records_path, result_path;
  long frames = 0, tracks_total = 0, resets = 0;
  long reset_every = 0;           // 0 -> null (never reset)
  double predict_ms_mean = 0.0, predict_ms_min = 0.0, predict_ms_max = 0.0;
  double tracker_ms_mean = 0.0;
  double cpu_ms_total = 0.0, wall_ms_total = 0.0, proposals_mean = 0.0;
};

/// Person tracking over an ordered video, mirroring the Python
/// track_video: read every frame, predict, draw the tracks (same id colors),
/// write the annotated mp4 — parent directories are created and the writer
/// must open before any frame is processed, so a requested output that
/// cannot be written fails the run instead of silently missing. With
/// reset_every > 0 reset the tracker every N frames (ids stay monotonic);
/// with records_path write the per-frame TSV (frame, track_id, x1, y1, x2,
/// y2, score at %.17g). options.no_encode skips the writer for timing-only
/// runs and reports an empty result_path. Throws std::runtime_error when the
/// video cannot be opened or has no frames, CliError when the records file
/// or the output video cannot be written.
VideoSummary track_video(PersonTracker& tracker, const CliOptions& options);

/// Assemble the image-mode JSON report (same keys as the Python main.py).
std::string image_report_json(const std::string& asset_id, const HashReport& hashes,
                              double predict_ms, double cpu_ms, int proposals,
                              const ImagePayload& payload);

/// Assemble the video-mode JSON report (same keys as the Python main.py).
std::string video_report_json(const std::string& asset_id, const HashReport& hashes,
                              const VideoSummary& summary);
#endif  // FRCNN_BOARD

}  // namespace fasterrcnn

#endif  // RDK_MODEL_ZOO_FASTERRCNN_CLI_HPP_
