// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli_io.h"
#include "preflight.h"
#include "sha256.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
namespace yoloe {
namespace fs = std::filesystem;
namespace {
std::string bytes(const std::string &path) {
  if (!fs::is_regular_file(path))
    throw std::invalid_argument("Input is not a regular file: " + path);
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::invalid_argument("Cannot open input: " + path);
  std::string data(std::istreambuf_iterator<char>(file), {});
  if (file.bad() || data.empty())
    throw std::invalid_argument("Empty or unreadable input: " + path);
  return data;
}
std::string quote(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    switch (c) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 32)
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(c) << std::dec;
      else
        out << static_cast<char>(c);
    }
  }
  out << '"';
  return out.str();
}
void write_image(const fs::path &path, const cv::Mat &image) {
  if (!cv::imwrite(path.string(), image))
    throw std::runtime_error("Cannot save image: " + path.string());
}
void validate_result(const Instance &item, const cv::Mat &image,
                     size_t classes) {
  if (item.label < 0 || static_cast<size_t>(item.label) >= classes ||
      !std::isfinite(item.score) || item.score < 0 || item.score > 1)
    throw std::invalid_argument("Invalid instance class/score");
  for (int i = 0; i < 4; ++i)
    if (!std::isfinite(item.box[i]) || item.box[i] < 0 ||
        double(item.box[i]) > (i % 2 ? image.rows : image.cols))
      throw std::invalid_argument("Invalid restored box");
  if (item.box[2] < item.box[0] || item.box[3] < item.box[1] ||
      item.mask.type() != CV_8UC1 ||
      item.mask.cols != int(item.box[2]) - int(item.box[0]) ||
      item.mask.rows != int(item.box[3]) - int(item.box[1]))
    throw std::invalid_argument("ROI mask geometry differs from box");
  for (int y = 0; y < item.mask.rows; ++y)
    for (int x = 0; x < item.mask.cols; ++x)
      if (item.mask.at<uint8_t>(y, x) > 1)
        throw std::invalid_argument("ROI masks must contain only 0/1");
}
} // namespace
CliInputs load_cli_inputs(const CliOptions &options) {
  auto encoded = bytes(options.image_path), labels = bytes(options.label_path);
  CliInputs result;
  result.image_sha256 = rdk::sha256_hex(encoded.data(), encoded.size());
  result.vocabulary_sha256 = rdk::sha256_hex(labels.data(), labels.size());
  if (result.vocabulary_sha256 != kVocabularySha256)
    throw std::invalid_argument("Vocabulary checksum mismatch");
  std::istringstream lines(labels);
  std::string line;
  while (std::getline(lines, line))
    result.labels.push_back(line);
  if (result.labels.size() != 4585)
    throw std::invalid_argument("Expected 4585 fixed vocabulary entries");
  result.image = cv::imdecode(
      std::vector<uint8_t>(encoded.begin(), encoded.end()), cv::IMREAD_COLOR);
  if (result.image.empty())
    throw std::invalid_argument("Cannot decode image: " + options.image_path);
  return result;
}
void create_output_directory(const std::string &name) {
  fs::path path(name);
  if (fs::exists(path) || fs::is_symlink(path))
    throw std::invalid_argument("Output directory must be new: " + name);
  if (!path.parent_path().empty())
    fs::create_directories(path.parent_path());
  if (!fs::create_directory(path))
    throw std::runtime_error("Cannot create output directory: " + name);
}
void save_cli_outputs(const CliOptions &options, const CliInputs &inputs,
                      const Result &result) {
  fs::path output(options.output);
  if (inputs.image.empty() || inputs.labels.size() != 4585 ||
      !fs::is_directory(output) || !fs::is_empty(output))
    throw std::invalid_argument(
        "Expected valid inputs and an empty new output directory");
  for (const auto &item : result)
    validate_result(item, inputs.image, inputs.labels.size());
  fs::create_directory(output / "masks");
  auto canvas = inputs.image.clone();
#if defined(YOLOE_HOST_FIXTURE)
  const char *backend = "host-fixture";
#else
  const char *backend = "native-sdk";
#endif
  std::ostringstream report;
  report << std::setprecision(std::numeric_limits<float>::max_digits10);
  report << "{\n  \"schema\": \"rdk-model-zoo/yoloe-native-run/v1\",\n  "
            "\"execution_backend\": "
         << quote(backend) << ",\n  \"target\": " << quote(options.model.target)
         << ",\n  \"variant\": " << quote(options.model.variant)
         << ",\n  \"model_path\": "
         << quote(fs::absolute(options.model.path).string())
         << ",\n  \"model_sha256\": " << quote(options.model_sha256)
         << ",\n  \"image_sha256\": " << quote(inputs.image_sha256)
         << ",\n  \"vocabulary_sha256\": " << quote(inputs.vocabulary_sha256)
         << ",\n  \"image_shape\": [" << inputs.image.rows << ", "
         << inputs.image.cols
         << ", 3],\n  \"mask_layout\": \"roi\",\n  \"mask_encoding\": "
            "\"png-uint8-0-or-255\",\n  \"image_saved\": \"annotated.png\",\n  "
            "\"config\": {\"score_threshold\": "
         << options.config.score_threshold << ", \"nms_threshold\": ";
  if (options.config.protocol == Protocol::E11)
    report << options.config.nms_threshold.value_or(0.7f);
  else
    report << "null";
  report << ", \"resize_type\": " << options.config.resize_type
         << ", \"do_morph\": " << (options.config.do_morph ? "true" : "false")
         << ", \"max_det\": " << options.config.max_det
         << ", \"single_label\": "
         << (options.config.single_label ? "true" : "false")
         << ", \"contours\": " << (options.contours ? "true" : "false")
         << "},\n  \"argv\": [";
  for (size_t i = 0; i < options.argv.size(); ++i) {
    if (i)
      report << ", ";
    report << quote(options.argv[i]);
  }
  report << "],\n  \"count\": " << result.size() << ",\n  \"instances\": [\n";
  for (size_t i = 0; i < result.size(); ++i) {
    const auto &item = result[i];
    int x = int(item.box[0]), y = int(item.box[1]);
    cv::Scalar color((37 * item.label + 50) % 256, (67 * item.label + 80) % 256,
                     (97 * item.label + 110) % 256);
    std::string mask_name;
    if (!item.mask.empty()) {
      std::ostringstream name;
      name << "masks/" << std::setw(6) << std::setfill('0') << i << ".png";
      mask_name = name.str();
      write_image(output / mask_name, item.mask * 255);
      auto view = canvas(cv::Rect(x, y, item.mask.cols, item.mask.rows));
      cv::Mat mixed;
      cv::addWeighted(view, 0.6, cv::Mat(view.size(), view.type(), color), 0.4,
                      0, mixed);
      mixed.copyTo(view, item.mask);
      if (options.contours) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(item.mask.clone(), contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(view, contours, -1, color, 1);
      }
    }
    cv::rectangle(canvas, cv::Point(x, y),
                  cv::Point(int(item.box[2]), int(item.box[3])), color, 2);
    cv::putText(canvas,
                inputs.labels[item.label] + " " +
                    cv::format("%.3f", item.score),
                cv::Point(x, std::max(15, y - 5)), cv::FONT_HERSHEY_SIMPLEX,
                0.5, color, 1);
    if (i)
      report << ",\n";
    report << "    {\"class_id\": " << item.label
           << ", \"label\": " << quote(inputs.labels[item.label])
           << ", \"score\": " << item.score << ", \"box\": [" << item.box[0]
           << ", " << item.box[1] << ", " << item.box[2] << ", " << item.box[3]
           << "], \"mask_shape\": [" << item.mask.rows << ", " << item.mask.cols
           << "], \"mask\": " << (mask_name.empty() ? "null" : quote(mask_name))
           << "}";
  }
  report << "\n  ]\n}\n";
  write_image(output / "annotated.png", canvas);
  auto temporary = output / "report.json.tmp";
  std::ofstream file(temporary, std::ios::binary);
  file << report.str();
  file.close();
  if (!file)
    throw std::runtime_error("Cannot write result report");
  fs::rename(temporary, output / "report.json");
}
} // namespace yoloe
