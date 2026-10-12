// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
/// Evidence tool (board builds only): dump the person detections the
/// ByteTrack C++ composition ACTUALLY feeds its tracker for a video, in the
/// BTCPP1 replay format. Replaying this file through the tracker reproduces
/// the E2E records bit-for-bit iff the composition layer is self-consistent;
/// diffing it against the Python detector's frozen dump attributes any
/// remaining E2E difference to the detector path with per-frame evidence.
///
/// Uses ByteTrackModel::detect (the shared decode_persons path postprocess
/// feeds the tracker), so the dump is the tracker's true input by
/// construction. No timing here: this is an equivalence tool.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/videoio.hpp>

#include "tracking.hpp"

namespace {

[[noreturn]] void usage_exit(const char* msg) {
  if (msg) std::fprintf(stderr, "error: %s\n", msg);
  std::fprintf(stderr,
               "usage: bytetrack_detector_dump --input FILE --model-path FILE "
               "[--bpu-cores I] [--out FILE] [--frames N]\n");
  std::exit(msg ? 2 : 0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string video, model, out = "detector-dump.btcpp";
  int bpu_core = -1, max_frames = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_value = i + 1 < argc;
    if (a == "--input" && has_value) video = argv[++i];
    else if (a == "--model-path" && has_value) model = argv[++i];
    else if (a == "--out" && has_value) out = argv[++i];
    else if (a == "--bpu-cores" && has_value) bpu_core = std::atoi(argv[++i]);
    else if (a == "--frames" && has_value) max_frames = std::atoi(argv[++i]);
    else if (a == "--help") usage_exit(nullptr);
    else usage_exit(a.c_str());
  }
  if (video.empty() || model.empty()) usage_exit("--input and --model-path are required");

  cv::VideoCapture cap(video);
  if (!cap.isOpened()) usage_exit("cannot open video");
  const int cols = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int rows = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double fps = cap.get(cv::CAP_PROP_FPS);

  bytetrack::ByteTrackModel::Options opt;
  opt.model_path = model;
  opt.bpu_core = bpu_core;
  bytetrack::ByteTrackModel model_runner(opt);

  std::ofstream fh(out, std::ios::binary);
  if (!fh) usage_exit("cannot open output");
  const char magic[8] = {'B', 'T', 'C', 'P', 'P', '1', '\0', '\0'};
  fh.write(magic, 8);
  // version, n_frames placeholder (patched after the loop), img_h, img_w
  const uint32_t header[4] = {1u, 0u, static_cast<uint32_t>(rows),
                              static_cast<uint32_t>(cols)};
  fh.write(reinterpret_cast<const char*>(header), sizeof header);

  long long total = 0;
  uint32_t n_frames = 0;
  bytetrack::ByteTrackModel::Frame f;  // reused: only the current frame stays resident
  while (max_frames < 0 || static_cast<int>(n_frames) < max_frames) {
    cv::Mat mat;
    if (!cap.read(mat)) break;
    f.rows = mat.rows;
    f.cols = mat.cols;
    f.bgr.assign(mat.data, mat.data + mat.total() * mat.elemSize());
    const std::vector<bytetrack::Detection> persons = model_runner.detect(f);
    const uint32_t n = static_cast<uint32_t>(persons.size());
    fh.write(reinterpret_cast<const char*>(&n), sizeof n);
    for (const bytetrack::Detection& d : persons)
      fh.write(reinterpret_cast<const char*>(&d.x1), 5 * sizeof(float));
    total += n;
    ++n_frames;
  }
  // Patch the frame count in the header (offset 12 = magic 8 + version 4).
  fh.seekp(12);
  fh.write(reinterpret_cast<const char*>(&n_frames), sizeof n_frames);
  fh.close();
  if (!fh) usage_exit("failed writing the dump");
  std::printf("detector dump: %u frames, %lld person detections, video %dx%d@%.2f\n",
              n_frames, total, cols, rows, fps);
  return 0;
}
