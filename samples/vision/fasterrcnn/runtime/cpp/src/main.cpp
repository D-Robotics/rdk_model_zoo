// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Thin entry: parse the CLI, run the model-free modes, resolve and verify
// the artifact pair, construct the named detector, call predict (per frame
// for video via cli::track_video, exactly like the Python main.py), and
// print the JSON run report. The pipeline stages live in detect.hpp; IO,
// rendering and report writing live in cli.hpp.

#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <string>

#include "cli.hpp"
#include "detect.hpp"

#if defined(FRCNN_BOARD)
// cv::setNumThreads below; the host build stays OpenCV-free.
#include <opencv2/core.hpp>
#endif

#if defined(FRCNN_BOARD)

namespace {

// Same clocks as the Python main.py: perf_counter (CLOCK_MONOTONIC) for wall
// time, process_time (user+system) for CPU time.
double wall_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

double cpu_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

}  // namespace

#endif  // FRCNN_BOARD

int main(int argc, char** argv) {
  try {
    const fasterrcnn::CliOptions options = fasterrcnn::parse_cli(argc, argv);
    if (options.list_models) return fasterrcnn::list_models(options.target);
    const fasterrcnn::ResolvedSelection selection = fasterrcnn::resolve_selection(options);
    if (options.dry_run) return fasterrcnn::dry_run(selection);

#if defined(FRCNN_BOARD)
    const fasterrcnn::HashReport hashes = fasterrcnn::verify_pair_hashes(selection);

    // The named model, constructed directly in the entry (Python:
    // FasterRCNNTask(selection)).
    fasterrcnn::FasterRCNN::Config config;
    config.target = selection.target;
    config.parta_path = selection.parta_path;
    config.partb_path = selection.partb_path;
    config.layout = selection.layout;
    config.parta_sha256 = selection.parta_sha256;
    config.partb_sha256 = selection.partb_sha256;
    config.asset_id = selection.asset_id;
    fasterrcnn::FasterRCNN detector(config);

    if (options.threads > 0) cv::setNumThreads(options.threads);

    if (options.track || !options.video_path.empty()) {
      fasterrcnn::PersonTracker tracker(detector, fasterrcnn::PersonTracker::Config{});
      const fasterrcnn::VideoSummary summary =
          fasterrcnn::track_video(tracker, options);
      std::printf("%s\n",
                  fasterrcnn::video_report_json(selection.asset_id, hashes, summary)
                      .c_str());
      return 0;
    }

    const fasterrcnn::FasterRCNN::Input image = fasterrcnn::load_image(options.input_path);
    const double cpu_started = cpu_ms();
    const double wall_started = wall_ms();
    const fasterrcnn::DetectionResult result = detector.predict(image);
    const double predict_ms = wall_ms() - wall_started;
    const double cpu_total = cpu_ms() - cpu_started;
    const fasterrcnn::ImagePayload payload = fasterrcnn::save_detection_image(
        image, result, options.label_file, options.save_path);
    std::printf("%s\n",
                fasterrcnn::image_report_json(
                    selection.asset_id, hashes, predict_ms, cpu_total,
                    detector.last_proposals(), payload)
                    .c_str());
    return 0;
#else
    throw std::runtime_error(
        "image and video inference require a board target build "
        "(FRCNN_TARGET x5, s100, s100p or s600); this host build supports "
        "--list-models and --dry-run only");
#endif
  } catch (const std::exception& exc) {
    std::fprintf(stderr, "error: %s\n", exc.what());
    return 2;
  }
}
