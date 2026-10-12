// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "cli.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>

#if defined(BYTETRACK_TARGET_X5) || defined(BYTETRACK_TARGET_S)
#include <opencv2/opencv.hpp>
#endif

namespace bytetrack {
namespace {

const char kUsage[] =
    "usage: bytetrack <mode> [options]\n"
    "modes:\n"
    "  replay  detections file -> tracks (host and board builds)\n"
    "  video   video + model -> tracks (board builds only)\n"
    "common options:\n"
    "  --records FILE        per-frame records TSV (frame lane id x1 y1 x2 y2 score)\n"
    "  --threads N           tracker-internal parallelism (default 1)\n"
    "replay options:\n"
    "  --detections FILE     BTCPP1 float32 replay file (required)\n"
    "video options:\n"
    "  --input FILE          input video (required)\n"
    "  --model-path FILE     detector model file (required)\n"
    "  --output FILE         annotated output video\n"
    "  --score-thres F       detector confidence threshold (default 0.25)\n"
    "  --nms-thres F         detector NMS IoU threshold (default 0.45)\n"
    "  --track-thresh F      tracking activation threshold (default 0.3)\n"
    "  --track-buffer N      frames a lost track survives (default 60)\n"
    "  --match-thresh F      first-association cost limit (default 0.8)\n"
    "  --frame-rate N        source frame rate scaling the buffer (default 30)\n"
    "  --mot20               source optional matching mode (no score fusion)\n"
    "  --bpu-cores I         BPU core: -1 any (default), one fixed index\n"
    "  --priority I          BPU task priority (default 0)\n"
    "  --max-frames N        stop after N frames (0 = all)\n"
    "  --help                print this usage and exit\n";

[[noreturn]] void bad(const std::string& msg) { throw CliError(msg + "\n\n" + kUsage); }

const char* next_arg(int& i, int argc, char** argv, const std::string& name) {
  if (i + 1 >= argc) bad("missing value for " + name);
  return argv[++i];
}

/// Numeric option parsing: the text must be one finite number, integers must
/// be whole (no silent 5.5 -> 5 truncation), and NaN/Inf are rejected before
/// any cast (casting them is undefined behavior). numeric_limits::min() is
/// deliberately NOT used as a lower bound: for float it is the smallest
/// positive normal, which would reject the valid threshold 0. Semantic range
/// checks (e.g. thresholds in [0,1]) happen later, where the meaning lives.
template <typename T>
T parse_number(const char* text, const std::string& name) {
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(text, &end);
  if (end == text || *end != '\0' || errno == ERANGE)
    bad("invalid value for " + name + ": " + text);
  if (!std::isfinite(v)) bad("value must be finite for " + name + ": " + text);
  if (std::is_integral<T>::value && v != std::floor(v))
    bad("value must be a whole number for " + name + ": " + text);
  const double lo = std::is_integral<T>::value
                        ? static_cast<double>(std::numeric_limits<T>::min())
                        : -static_cast<double>(std::numeric_limits<T>::max());
  if (v < lo || v > static_cast<double>(std::numeric_limits<T>::max()))
    bad("value out of range for " + name + ": " + text);
  return static_cast<T>(v);
}

uint32_t read_u32(const unsigned char* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

float read_f32(const unsigned char* p) {
  static_assert(sizeof(float) == 4, "32-bit float expected");
  uint32_t bits = read_u32(p);
  float f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

/// True when two paths name the same file: every writer this CLI opens
/// truncates its target, so a requested output colliding with an input
/// would destroy that input mid-run. Existing paths are compared with
/// filesystem::equivalent (resolves symlinks and catches hardlinks);
/// not-yet-existing paths fall back to weakly_canonical normalization
/// (resolves the existing prefix, keeps the pending tail verbatim).
bool same_file(const std::string& a, const std::string& b) {
  if (a == b) return true;
  namespace fs = std::filesystem;
  std::error_code ec;
  if (fs::exists(fs::path(a), ec) && fs::exists(fs::path(b), ec)) {
    const bool equivalent = fs::equivalent(fs::path(a), fs::path(b), ec);
    if (!ec) return equivalent;
  }
  std::error_code ea, eb;
  // weakly_canonical keeps the not-yet-existing tail verbatim, so collapse
  // its `..`/`.` segments lexically (symlinks in the existing prefix are
  // already resolved) — the same normalization the Python CLI's resolve()
  // applies before comparing.
  const fs::path na = fs::weakly_canonical(a, ea).lexically_normal();
  const fs::path nb = fs::weakly_canonical(b, eb).lexically_normal();
  return !ea && !eb && na == nb;
}

}  // namespace

CliOptions parse_cli(int argc, char** argv) {
  CliOptions opt;
  if (argc < 2) bad("missing mode");
  const std::string mode = argv[1];
  // --help answers before any mode/option requirement is enforced.
  if (mode == "--help" || mode == "-h") {
    std::fputs(kUsage, stdout);
    std::exit(0);
  }
  if (mode == "replay")
    opt.mode = CliOptions::Mode::Replay;
  else if (mode == "video")
#if defined(BYTETRACK_TARGET_X5) || defined(BYTETRACK_TARGET_S)
    opt.mode = CliOptions::Mode::Video;
#else
    bad("video mode requires a board build (BYTETRACK_TARGET=x5|s100|s100p|s600)");
#endif
  else
    bad("unknown mode: " + mode);

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--detections")
      opt.detections_path = next_arg(i, argc, argv, a);
    else if (a == "--input")
      opt.input_path = next_arg(i, argc, argv, a);
    else if (a == "--model-path")
      opt.model_path = next_arg(i, argc, argv, a);
    else if (a == "--output")
      opt.output_path = next_arg(i, argc, argv, a);
    else if (a == "--records")
      opt.records_path = next_arg(i, argc, argv, a);
    else if (a == "--score-thres")
      opt.score_thres = parse_number<float>(next_arg(i, argc, argv, a), a);
    else if (a == "--nms-thres")
      opt.nms_thres = parse_number<float>(next_arg(i, argc, argv, a), a);
    else if (a == "--track-thresh")
      opt.tracking.track_thresh = parse_number<double>(next_arg(i, argc, argv, a), a);
    else if (a == "--track-buffer")
      opt.tracking.track_buffer = parse_number<int>(next_arg(i, argc, argv, a), a);
    else if (a == "--match-thresh")
      opt.tracking.match_thresh = parse_number<double>(next_arg(i, argc, argv, a), a);
    else if (a == "--frame-rate")
      opt.tracking.frame_rate = parse_number<int>(next_arg(i, argc, argv, a), a);
    else if (a == "--mot20")
      opt.tracking.mot20 = true;
    else if (a == "--bpu-cores")
      opt.bpu_core = parse_number<int>(next_arg(i, argc, argv, a), a);
    else if (a == "--priority")
      opt.priority = parse_number<int>(next_arg(i, argc, argv, a), a);
    else if (a == "--max-frames")
      opt.max_frames = parse_number<int>(next_arg(i, argc, argv, a), a);
    else if (a == "--threads")
      opt.threads = parse_number<unsigned>(next_arg(i, argc, argv, a), a);
    else
      bad("unknown option: " + a);
  }
  if (opt.threads == 0) bad("--threads must be >= 1");
  if (opt.mode == CliOptions::Mode::Replay) {
    if (opt.detections_path.empty()) bad("replay requires --detections FILE");
  } else {
    if (opt.input_path.empty()) bad("video requires --input FILE");
    if (opt.model_path.empty()) bad("video requires --model-path FILE");
  }
  // Ranges the option documentation states (mirrors the Python CLI): frame
  // caps are nonnegative, thresholds live in [0,1], the BPU task priority
  // fits the SDK's 0..255, and a core is either "any" (-1) or one fixed
  // index. The tracker config's own bounds are enforced by ByteTracker's
  // constructor (TrackingConfigError).
  if (opt.max_frames < 0) bad("--max-frames must be >= 0");
  if (opt.score_thres < 0.f || opt.score_thres > 1.f || opt.nms_thres < 0.f || opt.nms_thres > 1.f)
    bad("--score-thres and --nms-thres must be within [0,1]");
  if (opt.priority < 0 || opt.priority > 255) bad("--priority must be within 0..255");
  if (opt.bpu_core < -1 || opt.bpu_core > 3) bad("--bpu-cores must be -1 (any) or a core index 0..3");
  // Writers truncate what they open, so a requested output must not name an
  // input (or the other writer's file) before anything is opened.
  std::vector<std::string> inputs;
  if (opt.mode == CliOptions::Mode::Replay)
    inputs.push_back(opt.detections_path);
  else
    inputs = {opt.input_path, opt.model_path};
  std::vector<std::string> outputs;
  if (!opt.records_path.empty()) outputs.push_back(opt.records_path);
  if (!opt.output_path.empty()) outputs.push_back(opt.output_path);
  for (const std::string& out : outputs)
    for (const std::string& in : inputs)
      if (same_file(out, in))
        bad("output path would overwrite an input; use a different path: " + out);
  if (!opt.records_path.empty() && !opt.output_path.empty() &&
      same_file(opt.records_path, opt.output_path))
    bad("--records and --output must name different files: " + opt.records_path);
  return opt;
}

ReplayFile read_replay_file(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw CliError("cannot open detections file: " + path);
  struct CloseGuard {
    std::FILE* f;
    ~CloseGuard() { std::fclose(f); }
  } guard{f};

  unsigned char header[24];
  if (std::fread(header, 1, sizeof header, f) != sizeof header)
    throw CliError("detections file too short for header: " + path);
  if (std::memcmp(header, "BTCPP1\0\0", 8) != 0)
    throw CliError("detections file magic mismatch (expected BTCPP1): " + path);
  const uint32_t version = read_u32(header + 8);
  if (version != 1) throw CliError("unsupported detections file version " + std::to_string(version));
  ReplayFile rf;
  rf.n_frames = read_u32(header + 12);
  rf.img_h = read_u32(header + 16);
  rf.img_w = read_u32(header + 20);
  if (rf.img_h == 0 || rf.img_w == 0) throw CliError("detections file has zero image size");
  rf.frames.resize(rf.n_frames);
  unsigned char count[4];
  unsigned char det[20];
  for (uint32_t fi = 0; fi < rf.n_frames; ++fi) {
    if (std::fread(count, 1, 4, f) != 4) throw CliError("truncated frame count at frame " + std::to_string(fi));
    const uint32_t n = read_u32(count);
    // A count beyond 2^24 is certainly file corruption, not a real frame.
    if (n > (1u << 24)) throw CliError("implausible detection count at frame " + std::to_string(fi));
    rf.frames[fi].resize(n);
    for (uint32_t d = 0; d < n; ++d) {
      if (std::fread(det, 1, 20, f) != 20)
        throw CliError("truncated detections at frame " + std::to_string(fi));
      Detection& out = rf.frames[fi][d];
      out.x1 = read_f32(det + 0);
      out.y1 = read_f32(det + 4);
      out.x2 = read_f32(det + 8);
      out.y2 = read_f32(det + 12);
      out.score = read_f32(det + 16);
    }
  }
  return rf;
}

RecordsWriter::RecordsWriter(const std::string& path) {
  file_ = std::fopen(path.c_str(), "wb");
  if (!file_) throw std::runtime_error("cannot open records file: " + path);
  std::fprintf(file_, "frame\tlane\ttrack_id\tx1\ty1\tx2\ty2\tscore\n");
}

RecordsWriter::~RecordsWriter() {
  if (file_) std::fclose(file_);
}

void RecordsWriter::row(int frame_id, int lane, const Track& track) {
  std::fprintf(file_, "%d\t%d\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n", frame_id, lane,
               track.track_id, track.tlbr[0], track.tlbr[1], track.tlbr[2], track.tlbr[3],
               track.score);
}

bool RecordsWriter::close() {
  if (!file_) return false;
  const bool ok = std::fflush(file_) == 0 && std::ferror(file_) == 0;
  const int closed = std::fclose(file_);
  file_ = nullptr;
  return ok && closed == 0;
}

#if defined(BYTETRACK_TARGET_X5) || defined(BYTETRACK_TARGET_S)
namespace {

/// Draw track boxes/IDs on an owned copy of the HWC BGR frame, using the
/// source ID colors (same palette as the Python CLI renderer).
cv::Mat render_tracks(const cv::Mat& bgr, const std::vector<Track>& tracks) {
  cv::Mat result = bgr.clone();
  for (const Track& t : tracks) {
    const int x1 = static_cast<int>(t.tlbr[0]), y1 = static_cast<int>(t.tlbr[1]);
    const int x2 = static_cast<int>(t.tlbr[2]), y2 = static_cast<int>(t.tlbr[3]);
    const cv::Scalar color((37 * t.track_id) % 255, (17 * t.track_id) % 255,
                           (29 * t.track_id) % 255);
    cv::rectangle(result, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    cv::putText(result, "ID:" + std::to_string(t.track_id), cv::Point(x1, std::max(15, y1 - 5)),
                cv::FONT_HERSHEY_SIMPLEX, .5, color, 2);
  }
  return result;
}

}  // namespace

struct VideoReader::ReaderImpl {
  cv::VideoCapture capture;
};

VideoReader::VideoReader(const std::string& path)
    : reader_impl_(std::make_unique<ReaderImpl>()) {
  if (!reader_impl_->capture.open(path) || !reader_impl_->capture.isOpened())
    throw std::runtime_error("cannot open video: " + path);
}

VideoReader::~VideoReader() = default;

bool VideoReader::read(ByteTrackModel::Frame& frame) {
  cv::Mat mat;
  if (!reader_impl_->capture.read(mat) || mat.empty()) return false;
  frame.rows = mat.rows;
  frame.cols = mat.cols;
  frame.bgr.assign(mat.data, mat.data + mat.total() * mat.elemSize());
  return true;
}

double VideoReader::fps() const {
  const double fps = reader_impl_->capture.get(cv::CAP_PROP_FPS);
  return fps > 0 ? fps : 30.0;
}

struct VideoSink::SinkImpl {
  cv::VideoWriter writer;
};

VideoSink::VideoSink() = default;

VideoSink::~VideoSink() = default;

void VideoSink::write(const std::string& path, double fps, const ByteTrackModel::Frame& frame,
                      const std::vector<Track>& tracks) {
  if (!sink_impl_) sink_impl_ = std::make_unique<SinkImpl>();
  if (!sink_impl_->writer.isOpened()) {
    sink_impl_->writer.open(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                            cv::Size(frame.cols, frame.rows));
    if (!sink_impl_->writer.isOpened())
      throw std::runtime_error("cannot open output video for writing: " + path);
  }
  const cv::Mat image(frame.rows, frame.cols, CV_8UC3,
                      const_cast<uint8_t*>(frame.bgr.data()));
  sink_impl_->writer.write(render_tracks(image, tracks));
}
#endif

}  // namespace bytetrack
