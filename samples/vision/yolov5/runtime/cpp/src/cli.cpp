// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// YOLOv5 CLI: owns the command line, the machine-comparable evidence dump, the
// rendered output image and the report orchestration. The manifest bytes are
// the contract of this writer; the model hands over a completed RunEvidence
// and this file serializes it without reinterpretation.

#include "cli.hpp"

#include "../../../../../../utils/c_utils/sha256.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace yolov5 {
namespace {

// ------------------------------------------------------------------ JSON ----

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  for (const char raw : value) {
    const unsigned char ch = static_cast<unsigned char>(raw);
    switch (ch) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (ch < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
          out += buffer;
        } else {
          out += static_cast<char>(ch);
        }
    }
  }
  return out;
}

std::string quote(const std::string& value) { return "\"" + json_escape(value) + "\""; }

std::string number(long long value) { return std::to_string(value); }
std::string number(int value) { return std::to_string(value); }

std::string number(double value) {
  std::ostringstream out;
  out << std::setprecision(9) << value;
  return out.str();
}

std::string shape_json(const std::vector<long long>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) out += ", ";
    out += number(shape[i]);
  }
  return out + "]";
}

// A -1 entry means "not reported by this SDK" and serializes as null so a
// consumer cannot mistake it for a real stride or dimension.
std::string opt_number(long long value) {
  return value < 0 ? std::string("null") : number(value);
}

std::string opt_array_json(const long long (&values)[4]) {
  std::string out = "[";
  for (int i = 0; i < 4; ++i) {
    if (i) out += ", ";
    out += opt_number(values[i]);
  }
  return out + "]";
}

std::string double_array_json(const std::vector<double>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out += ", ";
    out += number(values[i]);
  }
  return out + "]";
}

std::string long_array_json(const std::vector<long long>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out += ", ";
    out += number(values[i]);
  }
  return out + "]";
}

std::string tensor_info_json(const DumpTensorInfo& info) {
  std::string block = "      {\n";
  block += "        \"name\": " + quote(info.name) + ",\n";
  block += "        \"dtype\": " + quote(info.dtype) + ",\n";
  block += "        \"shape\": " + shape_json(info.shape) + ",\n";
  block += "        \"quanti\": " + quote(info.quanti) + ",\n";
  block += "        \"scale_len\": " + number(info.scale_len) + ",\n";
  block += "        \"aligned_byte_size\": " + opt_number(info.aligned_byte_size) + ",\n";
  block += "        \"stride\": " + opt_array_json(info.stride) + ",\n";
  block += "        \"aligned\": " + opt_array_json(info.aligned) + ",\n";
  block += "        \"quantize_axis\": " + opt_number(info.quantize_axis) + ",\n";
  block += "        \"scale_values\": " + double_array_json(info.scale_values) + ",\n";
  block += "        \"zero_point_values\": " + long_array_json(info.zero_point_values) + "\n";
  block += "      }";
  return block;
}

std::string string_array_json(const std::vector<std::string>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out += ", ";
    out += quote(values[i]);
  }
  return out + "]";
}

// ------------------------------------------------------------ parsing ----

const char* require_value(int argc, char** argv, int* index, const char* option) {
  if (*index + 1 >= argc) throw std::invalid_argument(std::string(option) + " needs a value");
  return argv[++*index];
}

float parse_threshold(const char* text, const char* option) {
  const std::string value = text;
  std::size_t consumed = 0;
  const float parsed = std::stof(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0F || parsed > 1.0F)
    throw std::invalid_argument(std::string(option) + " must be a finite value in [0,1]");
  return parsed;
}

int parse_int(const char* text, const char* option) {
  const std::string value = text;
  std::size_t consumed = 0;
  const int parsed = std::stoi(value, &consumed);
  if (consumed != value.size()) throw std::invalid_argument(std::string(option) + " must be an integer");
  return parsed;
}

}  // namespace

std::string sha256_hex(const void* data, std::size_t size) {
  return rdk::sha256_hex(data, size);
}

std::string sha256_file(const std::string& path) {
  return rdk::sha256_file(path);
}

std::string utc_timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return buffer;
}

std::string current_binary_path(const std::string& argv0) {
#if defined(__linux__)
  std::error_code code;
  const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", code);
  if (!code && !self.empty() && self.is_absolute()) return self.string();
#endif
  if (argv0.empty()) return {};
  std::error_code resolve_error;
  const std::filesystem::path resolved = std::filesystem::absolute(argv0, resolve_error);
  if (!resolve_error) return resolved.string();
  return {};
}

bool write_dump(const RunEvidence& record, std::string* error) {
  const auto fail = [error](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (record.dir.empty()) return fail("dump directory is empty");

  std::error_code code;
  std::filesystem::create_directories(record.dir, code);
  if (code) return fail("cannot create dump directory: " + code.message());

  // Each stage writes into its own subdirectory: the raw and transformed
  // payloads of one output share neither a file nor bytes, so a later stage
  // can never overwrite the original evidence of an earlier one.
  const auto write_tensor = [&](const DumpTensor& tensor, const std::string& category,
                                std::size_t index) -> std::string {
    const std::string filename = category + "/" + std::to_string(index) + "-" +
                                 (tensor.name.empty() ? "tensor" : tensor.name) + ".bin";
    const std::filesystem::path path = std::filesystem::path(record.dir) / filename;
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path(), code);
      if (code) return {};
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return {};
    if (!tensor.bytes.empty())
      out.write(reinterpret_cast<const char*>(tensor.bytes.data()),
                static_cast<std::streamsize>(tensor.bytes.size()));
    if (!out) return {};
    return filename;
  };
  const auto hash_of = [](const DumpTensor& tensor) {
    return tensor.bytes.empty() ? std::string()
                                : sha256_hex(tensor.bytes.data(), tensor.bytes.size());
  };

  std::vector<std::string> input_files;
  std::vector<std::string> input_hashes;
  for (std::size_t i = 0; i < record.input_tensors.size(); ++i) {
    const std::string filename = write_tensor(record.input_tensors[i], "input", i);
    if (filename.empty()) return fail("cannot write input tensor file");
    input_files.push_back(filename);
    input_hashes.push_back(hash_of(record.input_tensors[i]));
  }
  std::vector<std::string> raw_files;
  std::vector<std::string> raw_hashes;
  for (std::size_t i = 0; i < record.raw_tensors.size(); ++i) {
    const std::string filename = write_tensor(record.raw_tensors[i], "raw", i);
    if (filename.empty()) return fail("cannot write raw tensor file");
    raw_files.push_back(filename);
    raw_hashes.push_back(hash_of(record.raw_tensors[i]));
  }
  std::vector<std::string> transformed_files;
  std::vector<std::string> transformed_hashes;
  for (std::size_t i = 0; i < record.transformed_tensors.size(); ++i) {
    const std::string filename = write_tensor(record.transformed_tensors[i], "transformed", i);
    if (filename.empty()) return fail("cannot write transformed tensor file");
    transformed_files.push_back(filename);
    transformed_hashes.push_back(hash_of(record.transformed_tensors[i]));
  }

  const auto tensor_list_json = [](const std::vector<DumpTensor>& tensors,
                                   const std::vector<std::string>& files,
                                   const std::vector<std::string>& hashes) {
    std::string list = "[\n";
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto& tensor = tensors[i];
      list += "    {\"name\": " + quote(tensor.name) +
              ", \"dtype\": " + quote(tensor.dtype) +
              ", \"shape\": " + shape_json(tensor.shape) +
              ", \"bytes\": " + number(static_cast<long long>(tensor.bytes.size())) +
              ", \"file\": " + quote(files[i]) +
              ", \"sha256\": " + quote(hashes[i]) + "}";
      list += (i + 1 == tensors.size()) ? "\n" : ",\n";
    }
    return list + "  ]";
  };

  std::string json = "{\n";
  json += "  \"schema\": \"rdk-model-zoo/yolov5-cpp-dump/v2\",\n";
  json += "  \"utc\": " + quote(record.utc) + ",\n";
  json += "  \"target\": " + quote(record.target) + ",\n";
  json += "  \"build_target\": " + quote(record.build_target) + ",\n";
  json += "  \"asset_id\": " + quote(record.asset_id) + ",\n";
  json += "  \"model_path\": " + quote(record.model_path) + ",\n";
  json += "  \"model_sha256\": " + quote(sha256_file(record.model_path)) + ",\n";
  json += "  \"binary_path\": " + quote(record.binary_path) + ",\n";
  json += "  \"binary_sha256\": " +
          (record.binary_path.empty()
               ? std::string("null")
               : quote(sha256_file(record.binary_path))) +
          ",\n";
  json += "  \"image_path\": " + quote(record.image_path) + ",\n";
  json += "  \"image_sha256\": " + quote(sha256_file(record.image_path)) + ",\n";
  json += "  \"cwd\": " + quote(record.cwd) + ",\n";
  json += "  \"argv\": " + string_array_json(record.argv) + ",\n";
  json += "  \"return_code\": " + number(record.return_code) + ",\n";
  json += "  \"error\": " + quote(record.error) + ",\n";
  json += "  \"notes\": " + string_array_json(record.notes) + ",\n";

  json += "  \"parameters\": {\n";
  for (std::size_t i = 0; i < record.options.size(); ++i) {
    json += "    " + quote(record.options[i].first) + ": " +
            quote(record.options[i].second);
    json += (i + 1 == record.options.size()) ? "\n" : ",\n";
  }
  json += "  },\n";

  json += "  \"inputs\": [\n";
  for (std::size_t i = 0; i < record.inputs.size(); ++i) {
    json += tensor_info_json(record.inputs[i]);
    json += (i + 1 == record.inputs.size()) ? "\n" : ",\n";
  }
  json += "  ],\n";

  json += "  \"outputs\": [\n";
  for (std::size_t i = 0; i < record.outputs.size(); ++i) {
    json += tensor_info_json(record.outputs[i]);
    json += (i + 1 == record.outputs.size()) ? "\n" : ",\n";
  }
  json += "  ],\n";

  json += "  \"input_tensors\": " +
          (record.input_tensors.empty()
               ? std::string("[]")
               : tensor_list_json(record.input_tensors, input_files, input_hashes)) +
          ",\n";
  json += "  \"raw_tensors\": " +
          (record.raw_tensors.empty()
               ? std::string("[]")
               : tensor_list_json(record.raw_tensors, raw_files, raw_hashes)) +
          ",\n";
  json += "  \"transformed_tensors\": " +
          (record.transformed_tensors.empty()
               ? std::string("[]")
               : tensor_list_json(record.transformed_tensors, transformed_files,
                                  transformed_hashes)) +
          ",\n";

  const auto detections_block = [](const std::vector<Detection>& list) {
    std::string block = "[\n";
    for (std::size_t i = 0; i < list.size(); ++i) {
      const auto& det = list[i];
      block += "    {\"x1\": " + number(static_cast<double>(det.x1)) +
               ", \"y1\": " + number(static_cast<double>(det.y1)) +
               ", \"x2\": " + number(static_cast<double>(det.x2)) +
               ", \"y2\": " + number(static_cast<double>(det.y2)) +
               ", \"score\": " + number(static_cast<double>(det.score)) +
               ", \"class_id\": " + number(det.class_id) + "}";
      block += (i + 1 == list.size()) ? "\n" : ",\n";
    }
    return block + "  ]";
  };
  json += "  \"detections\": " + detections_block(record.detections) + ",\n";
  json += "  \"detections_original\": " + detections_block(record.detections_original) +
          "\n}\n";

  std::ofstream manifest(std::filesystem::path(record.dir) / "manifest.json",
                         std::ios::binary | std::ios::trunc);
  if (!manifest) return fail("cannot open dump manifest for writing");
  manifest << json;
  if (!manifest) return fail("cannot write dump manifest");
  return true;
}

// ------------------------------------------------------------ arguments ----

void print_help(const char* program) {
  std::cout << "Usage: " << program << " --target <x5|s100|s600> --model-path <file>"
            << " --test-img <file> [options]\n"
            << "  --asset-id <published-id>  checked by launcher, recorded in dump\n"
            << "  --label-file <file>       labels for rendering\n"
            << "  --output <file>           rendered output (default result.jpg)\n"
            << "  --dump-dir <dir>          write machine-comparable raw/results evidence\n"
            << "  --score-thres <0..1>      default 0.25\n"
            << "  --nms-thres <0..1>        default 0.45\n"
            << "  --priority <0..255>       default 0 (S only; x5 rejects non-defaults)\n"
            << "  --bpu-core <-1|0..3>      default -1 = any core (S only; x5 rejects "
               "non-defaults)\n"
            << "  --help                    show this message\n";
}

bool parse_options(int argc, char** argv, RuntimeOptions* options) {
  for (int i = 0; i < argc; ++i) options->argv.push_back(argv[i]);
  bool help = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { help = true; continue; }
    if (arg == "--target") options->target = require_value(argc, argv, &i, "--target");
    else if (arg == "--model-path") options->model_path = require_value(argc, argv, &i, "--model-path");
    else if (arg == "--test-img") options->image_path = require_value(argc, argv, &i, "--test-img");
    else if (arg == "--label-file") options->label_path = require_value(argc, argv, &i, "--label-file");
    else if (arg == "--output") options->output_path = require_value(argc, argv, &i, "--output");
    else if (arg == "--dump-dir") options->dump_dir = require_value(argc, argv, &i, "--dump-dir");
    else if (arg == "--asset-id") options->asset_id = require_value(argc, argv, &i, "--asset-id");
    else if (arg == "--score-thres") options->score_threshold = parse_threshold(require_value(argc, argv, &i, "--score-thres"), "--score-thres");
    else if (arg == "--nms-thres") options->nms_threshold = parse_threshold(require_value(argc, argv, &i, "--nms-thres"), "--nms-thres");
    else if (arg == "--priority") options->priority = parse_int(require_value(argc, argv, &i, "--priority"), "--priority");
    else if (arg == "--bpu-core") options->bpu_core = parse_int(require_value(argc, argv, &i, "--bpu-core"), "--bpu-core");
    else throw std::invalid_argument("unknown option: " + arg);
  }
  if (help) { print_help(argv[0]); return true; }
  if (options->target.empty() || options->model_path.empty() || options->image_path.empty())
    throw std::invalid_argument("--target, --model-path and --test-img are required");
  if (options->priority < 0 || options->priority > 255 || options->bpu_core < -1 ||
      options->bpu_core > 3)
    throw std::invalid_argument(
        "invalid scheduling parameter (--bpu-core must be -1 or a core index 0..3)");
  return false;
}

// ---------------------------------------------------------- source image ----

// The loaded frame plus the pixel copy handed to the model. Kept behind the
// pimpl so cli.hpp (and main.cpp) stay free of OpenCV includes.
struct SourceImage::Frame {
  cv::Mat image;
  std::vector<unsigned char> bgr;
};

SourceImage::SourceImage(const RuntimeOptions& options) : frame_(new Frame) {
  frame_->image = cv::imread(options.image_path);
  if (frame_->image.empty())
    throw std::invalid_argument(options.target == "x5" ? "X5 image is empty"
                                                       : "S image is empty");
  // cv::imread returns a continuous 8UC3 BGR buffer; copy it out so the model
  // receives caller-owned pixels with no OpenCV header attached.
  frame_->bgr.assign(frame_->image.data,
                     frame_->image.data + frame_->image.total() * frame_->image.elemSize());
}

SourceImage::~SourceImage() = default;
SourceImage::SourceImage(SourceImage&&) noexcept = default;
SourceImage& SourceImage::operator=(SourceImage&&) noexcept = default;

int SourceImage::cols() const { return frame_->image.cols; }
int SourceImage::rows() const { return frame_->image.rows; }
const std::vector<unsigned char>& SourceImage::bgr() const { return frame_->bgr; }
cv::Mat& SourceImage::canvas() const { return frame_->image; }

// ------------------------------------------------------------ rendering ----

void render_detections(cv::Mat& image, const std::vector<Detection>& detections,
                       int model_size, bool letterbox, const std::string& output_path,
                       const std::vector<std::string>& labels) {
  if (image.empty() || model_size <= 0 || output_path.empty())
    throw std::invalid_argument("Cannot render YOLOv5 detections");
  const double scale = letterbox
      ? std::min(static_cast<double>(model_size) / image.cols,
                 static_cast<double>(model_size) / image.rows)
      : 1.0;
  const double pad_x = letterbox ? (model_size - image.cols * scale) / 2.0 : 0.0;
  const double pad_y = letterbox ? (model_size - image.rows * scale) / 2.0 : 0.0;
  for (const auto& detection : detections) {
    const int x1 = std::max(0, std::min(image.cols - 1,
        static_cast<int>((detection.x1 - pad_x) / scale)));
    const int y1 = std::max(0, std::min(image.rows - 1,
        static_cast<int>((detection.y1 - pad_y) / scale)));
    const int x2 = std::max(0, std::min(image.cols - 1,
        static_cast<int>((detection.x2 - pad_x) / scale)));
    const int y2 = std::max(0, std::min(image.rows - 1,
        static_cast<int>((detection.y2 - pad_y) / scale)));
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
    const std::string name = detection.class_id >= 0 &&
            detection.class_id < static_cast<int>(labels.size())
        ? labels[static_cast<std::size_t>(detection.class_id)]
        : std::to_string(detection.class_id);
    cv::putText(image, name + ":" +
                std::to_string(detection.score), cv::Point(x1, std::max(12, y1 - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  }
  if (!cv::imwrite(output_path, image)) throw std::runtime_error("Cannot save YOLOv5 output image");
}

std::vector<std::string> load_labels(const std::string& label_path) {
  if (label_path.empty()) return {};
  std::ifstream input(label_path);
  if (!input) throw std::runtime_error("Cannot open YOLOv5 label file: " + label_path);
  std::vector<std::string> labels;
  for (std::string line; std::getline(input, line);) labels.push_back(line);
  return labels;
}

// --------------------------------------------------------------- report ----

int report(const RuntimeOptions& options, const Yolov5::Prediction& run,
           SourceImage& source, int model_size) {
  RunEvidence record = run.evidence;
  record.dir = options.dump_dir;
  record.utc = utc_timestamp();
  record.asset_id = options.asset_id;
  record.image_path = options.image_path;
  record.argv = options.argv;
  record.cwd = std::filesystem::current_path().string();
  record.binary_path =
      current_binary_path(options.argv.empty() ? "" : options.argv.front());
  record.options.push_back({"label_file", options.label_path});
  record.return_code = 0;
  if (!options.dump_dir.empty()) {
    std::string error;
    if (!write_dump(record, &error)) {
      throw std::runtime_error((options.target == "x5" ? "cannot write YOLOv5 X5 dump: "
                                                       : "cannot write YOLOv5 S dump: ") +
                               error);
    }
  }
  // The renderer draws on the image it is given; drawing on a shallow copy of
  // the CLI's frame matches the fixed source, which rendered onto the image it
  // had just read.
  cv::Mat canvas = source.canvas();
  render_detections(canvas, record.detections, model_size, true, options.output_path,
                    load_labels(options.label_path));
  return 0;
}

int write_failure_record(const RuntimeOptions& options, const std::string& failure,
                         int status) {
  // A failed run still has to be traceable on the board, and the failure
  // record keeps the binary identity like a successful one.
  RunEvidence record;
  record.dir = options.dump_dir;
  record.utc = utc_timestamp();
  record.target = options.target;
  record.asset_id = options.asset_id;
  record.model_path = options.model_path;
  record.image_path = options.image_path;
  record.argv = options.argv;
  record.cwd = std::filesystem::current_path().string();
  record.binary_path =
      current_binary_path(options.argv.empty() ? "" : options.argv.front());
  record.return_code = status;
  record.error = failure;
  record.notes = {"Failure record; no tensor payload was produced."};
  std::string ignored;
  write_dump(record, &ignored);
  return 0;
}

}  // namespace yolov5
