// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Faster R-CNN CLI implementation: option parsing (names, defaults and
// validation mirror the Python argparse surface), board-identity resolution
// and pair selection over the compiled-in descriptor table, digest
// verification, and — on board builds — image/video IO, rendering and the
// tracking loop. The JSON reports use the same keys as the Python main.py so
// same-input board runs compare directly.

#include "cli.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <errno.h>
#include <filesystem>
#include <limits>
#include <memory>

#if defined(FRCNN_BOARD)
#include <opencv2/opencv.hpp>
#endif

#ifndef FRCNN_SAMPLE_DIR
#define FRCNN_SAMPLE_DIR "."
#endif

namespace fasterrcnn {
namespace {

const char kUsage[] =
    "usage: fasterrcnn [options]\n"
    "TorchVision Faster R-CNN MobileNetV3-Large-320-FPN on RDK boards\n"
    "options:\n"
    "  --target TARGET       board target: auto, x5, s100, s100p or s600\n"
    "                        (default auto: resolve the running board)\n"
    "  --parta-path FILE     explicit PartA artifact (.bin on X5, .hbm on S)\n"
    "  --partb-path FILE     explicit PartB artifact; pair with --parta-path\n"
    "  --input FILE, -i FILE  input image, or video with --video/--track\n"
    "  --video FILE          ordered person video; enables tracking\n"
    "  --track               track persons across --video frames\n"
    "  --reset-every N       reset the tracker every N frames\n"
    "                        (ids stay monotonic across resets)\n"
    "  --records FILE        write per-frame track records as TSV\n"
    "                        (frame, track_id, x1, y1, x2, y2, score)\n"
    "  --no-encode           skip the annotated video writer (timing-only\n"
    "                        runs); the report's result_path is null\n"
    "  --save-path FILE      annotated output image path\n"
    "  --threads N           CPU thread count for resize/convert (1, 2 or 4)\n"
    "  --label-file FILE     91-slot COCO label file (line i names label i)\n"
    "  --list-models         list descriptor pairs and local availability\n"
    "  --dry-run             print the resolved selection without loading models\n"
    "  --help                print this usage and exit\n";

[[noreturn]] void bad(const std::string& msg) { throw CliError(msg + "\n\n" + kUsage); }

const char* next_arg(int& i, int argc, char** argv, const std::string& name) {
  if (i + 1 >= argc) bad("missing value for " + name);
  return argv[++i];
}

/// Numeric option parsing (mirrors the bytetrack CLI's rules): one finite
/// number fully consumed; integers must be whole; NaN/Inf are rejected
/// before any cast. Semantic range checks live where the meaning lives.
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
  // numeric_limits<float>::min() is the smallest positive normal, which would
  // wrongly reject 0; floats bound at -max, integers at their true min.
  const double lo = std::is_integral<T>::value
                        ? static_cast<double>(std::numeric_limits<T>::min())
                        : -static_cast<double>(std::numeric_limits<T>::max());
  if (v < lo || v > static_cast<double>(std::numeric_limits<T>::max()))
    bad("value out of range for " + name + ": " + text);
  return static_cast<T>(v);
}

std::string to_lower(std::string text) {
  for (char& c : text)
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  return text;
}

std::string trim(const std::string& text) {
  std::size_t begin = 0, end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
  return text.substr(begin, end - begin);
}

/// Leading "~" / "~user" expansion like pathlib's expanduser (the Python CLI
/// applies it to every user-supplied path; only the plain "~" form is worth
/// mirroring here).
std::string expand_user(const std::string& path) {
  if (path.empty() || path[0] != '~') return path;
  if (path.size() == 1) {
    const char* home = std::getenv("HOME");
    return home ? home : path;
  }
  if (path[1] == '/') {
    const char* home = std::getenv("HOME");
    if (!home) return path;
    return std::string(home) + path.substr(1);
  }
  return path;  // ~otheruser: no portable expansion; pass through unchanged
}

std::string sample_path(const char* relative) {
  return std::string(FRCNN_SAMPLE_DIR) + "/" + relative;
}

// ---- tiny JSON emitter (keys and structure mirror the Python reports) ----

void json_string(std::string& out, const std::string& value) {
  out += '"';
  for (char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned char>(c));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  out += '"';
}

/// %.17g: full round-trip precision for every float in the reports, so the
/// board comparison tooling can parse and diff values without losing bits.
std::string g17(double v) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return std::string(buf);
}

// ------------------------------------------------------- board identity ----
// Exact transcription of docs/release/platforms.json (the registry the
// Python platforms.py reads); same matching precedence: soc_name first (with
// the s100p board_type refinement), then socinfo names, then device-tree
// models.

struct BoardIdentity {
  const char* id;
  const char* base_soc;                    // "" -> no refinement
  const char* const* board_types;          // nullptr -> none
  const char* const* soc_names;
  const char* const* socinfo_names;
  const char* const* device_tree_models;
};

const char* const kX5Soc[] = {"x5", nullptr};
const char* const kX5Socinfo[] = {"x5u", "x5h", "x5m", nullptr};
const char* const kX5DeviceTree[] = {"D-Robotics RDK X5 V1.0", nullptr};
const char* const kS100Soc[] = {"s100", nullptr};
const char* const kS100pBoardTypes[] = {"s100p", "rdk s100p", nullptr};
const char* const kS100pSoc[] = {"s100p", nullptr};
const char* const kS600Soc[] = {"s600", nullptr};

const BoardIdentity kBoardIdentities[] = {
    {"x5", "", nullptr, kX5Soc, kX5Socinfo, kX5DeviceTree},
    {"s100", "", nullptr, kS100Soc, nullptr, nullptr},
    {"s100p", "s100", kS100pBoardTypes, kS100pSoc, nullptr, nullptr},
    {"s600", "", nullptr, kS600Soc, nullptr, nullptr},
};

bool contains(const char* const* values, const std::string& text) {
  if (values == nullptr) return false;
  for (const char* const* v = values; *v != nullptr; ++v)
    if (text == *v) return true;
  return false;
}

/// Read one identity file, stripping the NUL/space padding the sysfs and
/// device-tree files carry (Python: read_text().strip('\x00 \t\r\n')).
std::string read_identity(const char* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) return "";
  std::string text;
  char buf[256];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  const std::string kPad = "\x00 \t\r\n";
  std::size_t begin = 0, end = text.size();
  while (begin < end && kPad.find(text[begin]) != std::string::npos) ++begin;
  while (end > begin && kPad.find(text[end - 1]) != std::string::npos) --end;
  return text.substr(begin, end - begin);
}

std::string match_target(const std::string& soc, const std::string& board) {
  for (const BoardIdentity& row : kBoardIdentities)
    if (std::string(row.base_soc) == soc && contains(row.board_types, board))
      return row.id;
  for (const BoardIdentity& row : kBoardIdentities)
    if (contains(row.soc_names, soc)) return row.id;
  return "";
}

std::string file_exists_string(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(std::filesystem::path(path), ec) ? "true" : "false";
}

const char* layout_name(PartbLayout layout) {
  return layout == PartbLayout::kPooled ? "pooled" : "packed";
}

}  // namespace

CliOptions parse_cli(int argc, char** argv) {
  CliOptions opt;
  // No arguments at all is the Python default parse: every option has a
  // default, so the flow continues to board resolution (auto) exactly like
  // the Python CLI.
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::fputs(kUsage, stdout);
      std::exit(0);
    } else if (a == "--target") {
      opt.target = next_arg(i, argc, argv, a);
    } else if (a == "--parta-path") {
      opt.parta_path = expand_user(next_arg(i, argc, argv, a));
    } else if (a == "--partb-path") {
      opt.partb_path = expand_user(next_arg(i, argc, argv, a));
    } else if (a == "--input" || a == "-i") {
      opt.input_path = expand_user(next_arg(i, argc, argv, a));
    } else if (a == "--video") {
      opt.video_path = expand_user(next_arg(i, argc, argv, a));
    } else if (a == "--track") {
      opt.track = true;
    } else if (a == "--reset-every") {
      opt.reset_every = parse_number<int>(next_arg(i, argc, argv, a), a);
      opt.reset_every_given = true;
    } else if (a == "--records") {
      opt.records_path = expand_user(next_arg(i, argc, argv, a));
    } else if (a == "--no-encode") {
      opt.no_encode = true;
    } else if (a == "--save-path") {
      opt.save_path = expand_user(next_arg(i, argc, argv, a));
    } else if (a == "--threads") {
      opt.threads = parse_number<int>(next_arg(i, argc, argv, a), a);
    } else if (a == "--label-file") {
      opt.label_file = expand_user(next_arg(i, argc, argv, a));
    } else if (a == "--list-models") {
      opt.list_models = true;
    } else if (a == "--dry-run") {
      opt.dry_run = true;
    } else {
      bad("unknown option: " + a);
    }
  }
  // Defaults mirror the Python parser (resolved here so every later stage
  // sees the same strings the Python CLI would).
  if (opt.input_path.empty()) opt.input_path = sample_path("test_data/bus.jpg");
  if (opt.label_file.empty()) opt.label_file = sample_path("model/coco91_labels.txt");
  if (opt.save_path.empty()) opt.save_path = sample_path("test_data/result.jpg");

  const std::string key = to_lower(trim(opt.target));
  if (key != "auto" && find_descriptor_row(key) == nullptr)
    bad("invalid --target '" + opt.target +
        "' (choose from auto, x5, s100, s100p, s600)");
  opt.target = key;
  if (opt.parta_path.empty() != opt.partb_path.empty())
    bad("--parta-path and --partb-path must be given together");
  if (opt.threads != 0 && opt.threads != 1 && opt.threads != 2 && opt.threads != 4)
    bad("--threads must be one of 1, 2, 4");
  // Mirrors the Python CLI: unset means never reset; an explicit value must
  // be a positive frame count (0 is rejected there too, by track_video).
  if (opt.reset_every_given && opt.reset_every < 1)
    bad("--reset-every must be a positive frame count");
#if !defined(FRCNN_BOARD)
  if (opt.track || !opt.video_path.empty())
    bad("image and video inference require a board target build "
        "(FRCNN_TARGET x5, s100, s100p or s600); this host build supports "
        "--list-models and --dry-run only");
#endif
  return opt;
}

std::string detect_board_target() {
  const std::string soc = read_identity("/sys/class/boardinfo/soc_name");
  if (!soc.empty())
    return match_target(to_lower(soc),
                        to_lower(read_identity("/sys/class/boardinfo/board_type")));
  const std::string socinfo = read_identity("/sys/class/socinfo/soc_name");
  if (!socinfo.empty()) {
    const std::string key = to_lower(socinfo);
    for (const BoardIdentity& row : kBoardIdentities)
      if (contains(row.socinfo_names, key)) return row.id;
    return "";
  }
  const std::string model = read_identity("/proc/device-tree/model");
  for (const BoardIdentity& row : kBoardIdentities)
    if (contains(row.device_tree_models, model)) return row.id;
  return "";
}

ResolvedSelection resolve_selection(const CliOptions& options) {
  std::string key = to_lower(trim(options.target));
  if (key.empty()) key = "auto";
  if (key == "auto") {
    key = detect_board_target();
    if (key.empty())
      throw CliError("cannot identify this board; pass an explicit --target "
                     "x5, s100, s100p or s600");
  }
  const DescriptorRow* row = find_descriptor_row(key);
  if (row == nullptr)
    throw CliError("Faster R-CNN publishes x5, s100, s100p and s600, not '" + key + "'.");
  const bool custom = !options.parta_path.empty();
  ResolvedSelection selection;
  selection.target = key;
  selection.march = row->march;
  selection.parta_path =
      custom ? options.parta_path : descriptor_model_dir(key) + "/" + row->parta_file;
  selection.partb_path =
      custom ? options.partb_path : descriptor_model_dir(key) + "/" + row->partb_file;
  selection.parta_sha256 = custom ? "" : row->parta_sha256;
  selection.partb_sha256 = custom ? "" : row->partb_sha256;
  selection.layout = row->partb_layout;
  selection.asset_id = custom ? key + ":fasterrcnn:custom-pair"
                              : key + ":fasterrcnn:" + row->parta_file + "+" + row->partb_file;
  selection.custom_pair = custom;
  return selection;
}

HashReport verify_pair_hashes(const ResolvedSelection& selection) {
  if (!std::filesystem::is_regular_file(std::filesystem::path(selection.parta_path)))
    throw CliError("PartA artifact not found: " + selection.parta_path +
                   " (compile it with the conversion guide, or pass --parta-path)");
  if (!std::filesystem::is_regular_file(std::filesystem::path(selection.partb_path)))
    throw CliError("PartB artifact not found: " + selection.partb_path +
                   " (compile it with the conversion guide, or pass --partb-path)");
  HashReport report;
  report.parta_sha256 = file_sha256(selection.parta_path);
  report.partb_sha256 = file_sha256(selection.partb_path);
  report.parta_recorded = !selection.parta_sha256.empty();
  report.partb_recorded = !selection.partb_sha256.empty();
  report.parta_matches = report.parta_recorded && report.parta_sha256 == selection.parta_sha256;
  report.partb_matches = report.partb_recorded && report.partb_sha256 == selection.partb_sha256;
  const std::pair<const char*, const char*> halves[2] = {
      {"PartA", selection.parta_path.c_str()}, {"PartB", selection.partb_path.c_str()}};
  const std::string observed[2] = {report.parta_sha256, report.partb_sha256};
  const std::string expected[2] = {selection.parta_sha256, selection.partb_sha256};
  for (int i = 0; i < 2; ++i) {
    if (!expected[i].empty() && observed[i] != expected[i])
      throw CliError(std::string(halves[i].first) + " artifact " + halves[i].second +
                     " sha256 " + observed[i] + " does not match the descriptor's " +
                     expected[i] + "; recompile it with the conversion guide or pass an "
                     "explicit --parta-path/--partb-path pair.");
  }
  return report;
}

int list_models(const std::string& target) {
  std::string key = to_lower(trim(target));
  if (key.empty()) key = "auto";
  std::string out = "[";
  bool first = true;
  for (const DescriptorRow& row : descriptor_rows()) {
    if (key != "auto" && key != row.target) continue;
    if (!first) out += ",";
    first = false;
    const std::string parta = descriptor_model_dir(row.target) + "/" + row.parta_file;
    const std::string partb = descriptor_model_dir(row.target) + "/" + row.partb_file;
    out += "\n  {\n    \"target\": ";
    json_string(out, row.target);
    out += ",\n    \"march\": ";
    json_string(out, row.march);
    out += ",\n    \"asset_id\": ";
    json_string(out, std::string(row.target) + ":fasterrcnn:" + row.parta_file + "+" +
                                row.partb_file);
    out += ",\n    \"parta\": ";
    json_string(out, parta);
    out += ",\n    \"parta_present\": " + file_exists_string(parta);
    out += ",\n    \"partb\": ";
    json_string(out, partb);
    out += ",\n    \"partb_present\": " + file_exists_string(partb);
    out += ",\n    \"partb_layout\": ";
    json_string(out, layout_name(row.partb_layout));
    out += ",\n    \"compiler_latency_us\": {\n      \"parta\": " + g17(row.parta_latency_us) +
           ",\n      \"partb\": " + g17(row.partb_latency_us) + "\n    }\n  }";
  }
  out += first ? "]" : "\n]";
  std::printf("%s\n", out.c_str());
  return 0;
}

int dry_run(const ResolvedSelection& selection) {
  std::string out = "{\n  \"target\": ";
  json_string(out, selection.target);
  out += ",\n  \"parta_path\": ";
  json_string(out, selection.parta_path);
  out += ",\n  \"parta_present\": " + file_exists_string(selection.parta_path);
  out += ",\n  \"partb_path\": ";
  json_string(out, selection.partb_path);
  out += ",\n  \"partb_present\": " + file_exists_string(selection.partb_path);
  out += ",\n  \"partb_layout\": ";
  json_string(out, layout_name(selection.layout));
  out += ",\n  \"asset_id\": ";
  json_string(out, selection.asset_id);
  out += "\n}";
  std::printf("%s\n", out.c_str());
  return 0;
}

std::vector<std::string> load_labels(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) throw CliError("cannot open label file: " + path);
  std::vector<std::string> lines;
  std::string current;
  char buf[4096];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) {
    for (std::size_t i = 0; i < n; ++i) {
      if (buf[i] == '\n') {
        if (!current.empty() && current.back() == '\r') current.pop_back();
        lines.push_back(current);
        current.clear();
      } else {
        current += buf[i];
      }
    }
  }
  std::fclose(f);
  if (!current.empty()) {
    if (!current.empty() && current.back() == '\r') current.pop_back();
    lines.push_back(current);
  }
  if (lines.size() != 91)
    throw CliError("label file must have 91 lines, " + path + " has " +
                   std::to_string(lines.size()) + ".");
  return lines;
}

#if defined(FRCNN_BOARD)
namespace {

/// A string value, or JSON null for the Python None (empty string marks it).
void json_string_or_null(std::string& out, const std::string& value) {
  if (value.empty())
    out += "null";
  else
    json_string(out, value);
}

// CLOCK_MONOTONIC matches time.perf_counter, CLOCK_PROCESS_CPUTIME_ID
// matches time.process_time (user+system of this process).
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

/// int(round(v)) with Python round semantics (ties to even).
int iround(double v) { return static_cast<int>(std::nearbyint(v)); }

cv::Mat render_detections(const cv::Mat& image, const DetectionResult& result,
                          const std::vector<std::string>& labels) {
  cv::Mat canvas = image.clone();
  for (std::size_t i = 0; i < result.size(); ++i) {
    const float* box = result.boxes.data() + i * 4;
    const int x1 = iround(box[0]), y1 = iround(box[1]);
    const int x2 = iround(box[2]), y2 = iround(box[3]);
    const cv::Scalar color(0, 215, 255);
    cv::rectangle(canvas, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    char text[512];
    std::snprintf(text, sizeof text, "%s %.2f", labels[std::size_t(result.labels[i])].c_str(),
                  result.scores[i]);
    int baseline = 0;
    const cv::Size size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(canvas, cv::Point(x1, y1 - size.height - 6),
                  cv::Point(x1 + size.width + 4, y1), color, -1);
    cv::putText(canvas, text, cv::Point(x1 + 2, y1 - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
  return canvas;
}

cv::Mat render_tracks(const cv::Mat& frame, const std::vector<bytetrack::Track>& tracks) {
  cv::Mat canvas = frame.clone();
  for (const bytetrack::Track& track : tracks) {
    const int x1 = iround(track.tlbr[0]), y1 = iround(track.tlbr[1]);
    const int x2 = iround(track.tlbr[2]), y2 = iround(track.tlbr[3]);
    const long long id = track.track_id;
    const cv::Scalar color((90 * id) % 255, 200, 255 - (60 * id) % 255);
    cv::rectangle(canvas, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    char text[64];
    std::snprintf(text, sizeof text, "id%lld %.2f", id, track.score);
    cv::putText(canvas, text, cv::Point(x1, y1 - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);
  }
  return canvas;
}

/// Path.with_suffix(".mp4") for the annotated video output.
std::string with_mp4_suffix(const std::string& path) {
  const std::size_t slash = path.find_last_of('/');
  const std::size_t dot = path.find_last_of('.');
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
    return path.substr(0, dot) + ".mp4";
  return path + ".mp4";
}

/// Per-frame records TSV with the Python header and %.17g values.
class RecordsWriter {
 public:
  explicit RecordsWriter(const std::string& path) {
    file_ = std::fopen(path.c_str(), "wb");
    if (file_ == nullptr)
      throw CliError("cannot open records file for writing: " + path);
    std::fprintf(file_, "frame\ttrack_id\tx1\ty1\tx2\ty2\tscore\n");
  }
  ~RecordsWriter() {
    if (file_ != nullptr) std::fclose(file_);
  }
  RecordsWriter(const RecordsWriter&) = delete;
  RecordsWriter& operator=(const RecordsWriter&) = delete;
  void row(long frame, const bytetrack::Track& track) {
    std::fprintf(file_, "%ld\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n", frame,
                 track.track_id, track.tlbr[0], track.tlbr[1], track.tlbr[2], track.tlbr[3],
                 track.score);
  }
  /// Flush and close; throws when any write failed (the Python run reports
  /// close failures through OSError -> exit 2).
  void close() {
    if (file_ == nullptr) return;
    const bool ok = std::fflush(file_) == 0 && std::ferror(file_) == 0;
    const int closed = std::fclose(file_);
    file_ = nullptr;
    if (!ok || closed != 0)
      throw CliError("failed writing the records file");
  }

 private:
  std::FILE* file_ = nullptr;
};

}  // namespace

FasterRCNN::Input load_image(const std::string& path) {
  const cv::Mat mat = cv::imread(path, cv::IMREAD_COLOR);
  if (mat.empty())
    throw std::runtime_error("input image not found or unreadable: " + path);
  FasterRCNN::Input image;
  image.rows = mat.rows;
  image.cols = mat.cols;
  image.bgr.assign(mat.data, mat.data + mat.total() * mat.elemSize());
  return image;
}

ImagePayload save_detection_image(const FasterRCNN::Input& image,
                                  const DetectionResult& result,
                                  const std::string& label_file,
                                  const std::string& save_path) {
  const std::vector<std::string> labels = load_labels(label_file);
  const cv::Mat source(image.rows, image.cols, CV_8UC3,
                       const_cast<std::uint8_t*>(image.bgr.data()));
  const cv::Mat canvas = render_detections(source, result, labels);
  namespace fs = std::filesystem;
  const fs::path out(save_path);
  if (out.has_parent_path()) {
    std::error_code ec;
    fs::create_directories(out.parent_path(), ec);
    if (ec && !fs::is_directory(out.parent_path()))
      throw std::runtime_error("failed to create output directory " +
                               out.parent_path().string() + ": " + ec.message());
  }
  if (!cv::imwrite(save_path, canvas))
    throw std::runtime_error("failed to write " + save_path);
  ImagePayload payload;
  payload.boxes = result.boxes;
  payload.scores = result.scores;
  payload.labels = result.labels;
  payload.result_path = save_path;
  return payload;
}

VideoSummary track_video(PersonTracker& tracker, const CliOptions& options) {
  const std::string video_path =
      options.video_path.empty() ? options.input_path : options.video_path;
  cv::VideoCapture capture(video_path);
  if (!capture.isOpened())
    throw std::runtime_error("video not found or unreadable: " + video_path);
  const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = capture.get(cv::CAP_PROP_FPS);
  if (!(fps > 0)) fps = 30.0;
  const std::string out_path = with_mp4_suffix(options.save_path);

  // The writer (and its parent directories) must be ready before any frame
  // is processed: a requested output that cannot be written fails the run
  // here, instead of "succeeding" with frames counted and no file produced.
  bool encode = !options.no_encode;
  cv::VideoWriter writer;
  if (encode) {
    namespace fs = std::filesystem;
    const fs::path out(out_path);
    if (out.has_parent_path()) {
      std::error_code ec;
      fs::create_directories(out.parent_path(), ec);
      if (ec && !fs::is_directory(out.parent_path()))
        throw std::runtime_error("failed to create output directory " +
                                 out.parent_path().string() + ": " + ec.message());
    }
    writer.open(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                cv::Size(width, height));
    if (!writer.isOpened())
      throw std::runtime_error("failed to open output video for writing: " + out_path);
  }

  VideoSummary summary;
  summary.video = video_path;
  if (encode) summary.result_path = out_path;  // empty -> null in the report
  summary.reset_every = options.reset_every;
  std::vector<double> predict_ms, tracker_ms;
  std::vector<double> proposals;
  std::unique_ptr<RecordsWriter> records;
  if (!options.records_path.empty())
    records = std::make_unique<RecordsWriter>(options.records_path);

  const double cpu_start = cpu_ms();
  const double wall_start = wall_ms();
  cv::Mat frame;
  while (capture.read(frame) && !frame.empty()) {
    FasterRCNN::Input input;
    input.rows = frame.rows;
    input.cols = frame.cols;
    input.bgr.assign(frame.data, frame.data + frame.total() * frame.elemSize());
    const double started = wall_ms();
    const std::vector<bytetrack::Track> tracks = tracker.predict(input);
    predict_ms.push_back(wall_ms() - started);
    tracker_ms.push_back(tracker.last_tracker_us() / 1e3);
    proposals.push_back(tracker.detector().last_proposals());
    summary.tracks_total += static_cast<long>(tracks.size());
    if (records)
      for (const bytetrack::Track& track : tracks)
        records->row(summary.frames, track);
    if (encode) writer.write(render_tracks(frame, tracks));
    ++summary.frames;
    if (options.reset_every > 0 && summary.frames % options.reset_every == 0) {
      tracker.reset();
      ++summary.resets;
    }
  }
  capture.release();
  if (encode) writer.release();
  if (records) records->close();
  if (summary.frames == 0)
    throw std::runtime_error("video has no readable frames: " + video_path);
  if (!options.records_path.empty()) summary.records_path = options.records_path;

  const double count = predict_ms.empty() ? 1.0 : double(predict_ms.size());
  double predict_sum = 0.0, tracker_sum = 0.0, proposals_sum = 0.0;
  for (double v : predict_ms) predict_sum += v;
  for (double v : tracker_ms) tracker_sum += v;
  for (double v : proposals) proposals_sum += v;
  summary.predict_ms_mean = predict_sum / count;
  summary.predict_ms_min = predict_ms.empty() ? 0.0 : *std::min_element(predict_ms.begin(), predict_ms.end());
  summary.predict_ms_max = predict_ms.empty() ? 0.0 : *std::max_element(predict_ms.begin(), predict_ms.end());
  summary.tracker_ms_mean = tracker_sum / count;
  summary.proposals_mean = proposals_sum / count;
  summary.cpu_ms_total = cpu_ms() - cpu_start;
  summary.wall_ms_total = wall_ms() - wall_start;
  return summary;
}

std::string image_report_json(const std::string& asset_id, const HashReport& hashes,
                              double predict_ms, double cpu_ms, int proposals,
                              const ImagePayload& payload) {
  std::string out = "{\"asset_id\": ";
  json_string(out, asset_id);
  out += ", \"parta_sha256\": ";
  json_string(out, hashes.parta_sha256);
  out += ", \"partb_sha256\": ";
  json_string(out, hashes.partb_sha256);
  out += ", \"parta_matches_descriptor\": ";
  out += hashes.parta_recorded ? (hashes.parta_matches ? "true" : "false") : "null";
  out += ", \"partb_matches_descriptor\": ";
  out += hashes.partb_recorded ? (hashes.partb_matches ? "true" : "false") : "null";
  out += ", \"predict_ms\": " + g17(predict_ms) + ", \"cpu_ms\": " + g17(cpu_ms) +
         ", \"proposals\": " + std::to_string(proposals) + ", \"boxes\": [";
  for (std::size_t i = 0; i < payload.boxes.size() / 4; ++i) {
    if (i) out += ", ";
    out += "[" + g17(payload.boxes[i * 4]) + ", " + g17(payload.boxes[i * 4 + 1]) + ", " +
           g17(payload.boxes[i * 4 + 2]) + ", " + g17(payload.boxes[i * 4 + 3]) + "]";
  }
  out += "], \"scores\": [";
  for (std::size_t i = 0; i < payload.scores.size(); ++i) {
    if (i) out += ", ";
    out += g17(payload.scores[i]);
  }
  out += "], \"labels\": [";
  for (std::size_t i = 0; i < payload.labels.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(payload.labels[i]);
  }
  out += "], \"result_path\": ";
  json_string(out, payload.result_path);
  out += "}";
  return out;
}

std::string video_report_json(const std::string& asset_id, const HashReport& hashes,
                              const VideoSummary& s) {
  std::string out = "{\"asset_id\": ";
  json_string(out, asset_id);
  out += ", \"parta_sha256\": ";
  json_string(out, hashes.parta_sha256);
  out += ", \"partb_sha256\": ";
  json_string(out, hashes.partb_sha256);
  out += ", \"parta_matches_descriptor\": ";
  out += hashes.parta_recorded ? (hashes.parta_matches ? "true" : "false") : "null";
  out += ", \"partb_matches_descriptor\": ";
  out += hashes.partb_recorded ? (hashes.partb_matches ? "true" : "false") : "null";
  out += ", \"video\": ";
  json_string(out, s.video);
  out += ", \"frames\": " + std::to_string(s.frames) +
         ", \"predict_ms_mean\": " + g17(s.predict_ms_mean) +
         ", \"predict_ms_min\": " + g17(s.predict_ms_min) +
         ", \"predict_ms_max\": " + g17(s.predict_ms_max) +
         ", \"tracker_ms_mean\": " + g17(s.tracker_ms_mean) +
         ", \"cpu_ms_total\": " + g17(s.cpu_ms_total) +
         ", \"wall_ms_total\": " + g17(s.wall_ms_total) +
         ", \"proposals_mean\": " + g17(s.proposals_mean) +
         ", \"tracks_total\": " + std::to_string(s.tracks_total) + ", \"reset_every\": ";
  out += s.reset_every > 0 ? std::to_string(s.reset_every) : std::string("null");
  out += ", \"resets\": " + std::to_string(s.resets) + ", \"records_path\": ";
  json_string_or_null(out, s.records_path);
  out += ", \"result_path\": ";
  json_string(out, s.result_path);
  out += "}";
  return out;
}

#endif  // FRCNN_BOARD

}  // namespace fasterrcnn
