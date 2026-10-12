// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
/// Thin entry: parse the CLI, construct the named tracking model, call
/// predict per frame, hand the results to the CLI writers (records TSV and
/// optional annotated video). The tracking flow itself lives in
/// tracking.hpp; this file only composes.
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "tracking.hpp"

namespace {

int run_replay(const bytetrack::CliOptions& opt) {
  const bytetrack::ReplayFile rf = bytetrack::read_replay_file(opt.detections_path);
  const uint32_t n_frames =
      opt.max_frames > 0 && static_cast<uint32_t>(opt.max_frames) < rf.n_frames
          ? static_cast<uint32_t>(opt.max_frames)
          : rf.n_frames;
  bytetrack::ByteTracker tracker(opt.tracking, opt.threads);
  std::unique_ptr<bytetrack::RecordsWriter> records;
  if (!opt.records_path.empty()) records = std::make_unique<bytetrack::RecordsWriter>(opt.records_path);
  uint64_t total_tracks = 0;
  for (uint32_t fi = 0; fi < n_frames; ++fi) {
    const std::vector<bytetrack::Track> tracks =
        tracker.update(rf.frames[fi], static_cast<int>(rf.img_h), static_cast<int>(rf.img_w));
    total_tracks += tracks.size();
    if (records)
      for (const bytetrack::Track& t : tracks) records->row(tracker.frame_id(), 0, t);
  }
  bool records_ok = true;
  if (records) records_ok = records->close();
  if (!records_ok) {
    std::fprintf(stderr, "error: failed writing records file: %s\n", opt.records_path.c_str());
    return 1;
  }
  std::printf("replay: %u frames, %llu track rows, final frame_id %d\n", n_frames,
              static_cast<unsigned long long>(total_tracks), tracker.frame_id());
  return 0;
}

#if defined(BYTETRACK_TARGET_X5) || defined(BYTETRACK_TARGET_S)

int run_video(const bytetrack::CliOptions& opt) {
  // Video IO (reading, annotated writing, records) belongs to the CLI; the
  // entry only opens the input, constructs the named model and calls
  // predict. The model composes the YOLOv5 detector stages with one tracker
  // stream (multi-stream stress belongs to the benchmark harness, not the
  // delivery entry).
  bytetrack::VideoReader video(opt.input_path);

  bytetrack::ByteTrackModel::Options model_opt;
  model_opt.model_path = opt.model_path;
  model_opt.score_thres = opt.score_thres;
  model_opt.nms_thres = opt.nms_thres;
  model_opt.bpu_priority = opt.priority;
  model_opt.bpu_core = opt.bpu_core;
  model_opt.tracking = opt.tracking;
  model_opt.threads = opt.threads;
  bytetrack::ByteTrackModel model(model_opt);

  std::unique_ptr<bytetrack::RecordsWriter> records;
  if (!opt.records_path.empty()) records = std::make_unique<bytetrack::RecordsWriter>(opt.records_path);
  bytetrack::VideoSink sink;
  const double fps = video.fps();

  bytetrack::ByteTrackModel::Frame frame;
  uint64_t total_tracks = 0;
  int processed = 0;
  while (video.read(frame)) {
    if (opt.max_frames > 0 && processed >= opt.max_frames) break;
    const std::vector<bytetrack::Track> tracks = model.predict(frame);
    total_tracks += tracks.size();
    if (records)
      for (const bytetrack::Track& t : tracks) records->row(model.frame_id(), 0, t);
    if (!opt.output_path.empty())
      sink.write(opt.output_path, fps, frame, tracks);
    ++processed;
  }
  if (processed == 0) throw std::runtime_error("video has no readable frames: " + opt.input_path);
  bool records_ok = true;
  if (records) records_ok = records->close();
  if (!records_ok) {
    std::fprintf(stderr, "error: failed writing records file: %s\n", opt.records_path.c_str());
    return 1;
  }
  std::printf("video: %d frames, %llu track rows\n", processed,
              static_cast<unsigned long long>(total_tracks));
  return 0;
}
#endif  // board targets

}  // namespace

int main(int argc, char** argv) {
  try {
    const bytetrack::CliOptions opt = bytetrack::parse_cli(argc, argv);
#if defined(BYTETRACK_TARGET_X5) || defined(BYTETRACK_TARGET_S)
    if (opt.mode == bytetrack::CliOptions::Mode::Replay) return run_replay(opt);
    return run_video(opt);
#else
    return run_replay(opt);  // host build: parse_cli has rejected video mode
#endif
  } catch (const bytetrack::CliError& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
