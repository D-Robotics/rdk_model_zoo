#ifndef RDK_MODEL_ZOO_YOLOV5_CLI_HPP_
#define RDK_MODEL_ZOO_YOLOV5_CLI_HPP_
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// YOLOv5 CLI header. The CLI owns the command line, the source image file IO,
// the machine-comparable dump writer, the rendered output and the report
// orchestration; the model (detect.hpp) owns the inference stages and the run
// evidence this file serializes. Publication facts are resolved by
// launcher.py; the native binary never guesses a layout from a file name.

#include "detect.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cv {
class Mat;
}

namespace yolov5 {

struct RuntimeOptions {
  std::string model_path;
  std::string image_path;
  std::string output_path;
  std::string label_path;
  std::string target;
  // Publication fact resolved by the launcher; recorded in the evidence dump.
  std::string asset_id;
  // When non-empty the CLI writes a machine-comparable dump record here.
  std::string dump_dir;
  // Command line as received, recorded verbatim in the dump.
  std::vector<std::string> argv;
  float score_threshold = 0.25F;
  float nms_threshold = 0.45F;
  int priority = 0;
  int bpu_core = -1;
};

// Prefills options->argv verbatim, then parses the arguments. Returns true
// when help was requested (already printed); throws std::invalid_argument with
// the same messages as the original inline parser on bad input, and after the
// loop validates the required options and the scheduling ranges.
bool parse_options(int argc, char** argv, RuntimeOptions* options);

void print_help(const char* program);

// The CLI-owned source image. The CLI does the file IO: it reads
// options.image_path, keeps the loaded frame for rendering and exposes the
// pixels the model receives as its Input — the model never sees a path.
// Throws std::invalid_argument with the same per-backend message as the
// fixed sources when the file does not decode to an image.
class SourceImage {
 public:
  explicit SourceImage(const RuntimeOptions& options);
  ~SourceImage();
  SourceImage(SourceImage&&) noexcept;
  SourceImage& operator=(SourceImage&&) noexcept;
  SourceImage(const SourceImage&) = delete;
  SourceImage& operator=(const SourceImage&) = delete;

  int cols() const;
  int rows() const;
  // 8-bit interleaved BGR, rows()*cols()*3 bytes, for Yolov5::Input.
  const std::vector<unsigned char>& bgr() const;
  // The loaded frame; report() renders onto a shallow copy of it, matching
  // the fixed sources which drew on the image they had just read.
  cv::Mat& canvas() const;

 private:
  struct Frame;
  std::unique_ptr<Frame> frame_;
};

// Serializes a completed run: fills the CLI-owned launch facts (dump dir,
// timestamp, asset id, image path, argv, cwd, binary), appends the label file
// to the recorded parameters, writes the dump when requested and renders the
// output image onto the source frame. Returns 0; throws on a dump or render
// failure.
int report(const RuntimeOptions& options, const Yolov5::Prediction& run,
           SourceImage& source, int model_size);

// Best-effort manifest for a failed run so it stays traceable on the board;
// keeps the binary identity like a successful one.
int write_failure_record(const RuntimeOptions& options, const std::string& failure,
                         int status);

// Writes <dir>/manifest.json and one little-endian file per dumped tensor
// under category subdirectories (input/, raw/, transformed/): the two stages
// of one output never share a file, so a later write cannot overwrite the
// original bytes of an earlier one. Returns false and fills *error on any
// filesystem failure.
bool write_dump(const RunEvidence& record, std::string* error);

// Best-effort path of the currently running executable: /proc/self/exe where
// available, otherwise argv[0] resolved against the current directory. Empty
// when neither can be determined.
std::string current_binary_path(const std::string& argv0);

// Lowercase hex SHA-256 of a file's contents, or an empty string if unreadable.
std::string sha256_file(const std::string& path);

// Lowercase hex SHA-256 of a byte range.
std::string sha256_hex(const void* data, std::size_t size);

// Current UTC timestamp in the "%Y-%m-%dT%H:%M:%SZ" form.
std::string utc_timestamp();

void render_detections(cv::Mat& image, const std::vector<Detection>& detections,
                       int model_size, bool letterbox, const std::string& output_path,
                       const std::vector<std::string>& labels);

std::vector<std::string> load_labels(const std::string& label_path);

}  // namespace yolov5

#endif  // RDK_MODEL_ZOO_YOLOV5_CLI_HPP_
