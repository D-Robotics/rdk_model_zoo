/**
 * @file main.cpp
 * @brief Run source-indexed HIMLoco inputs through an RDK X5 BIN model.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/utsname.h>

#include "himloco.hpp"

namespace fs = std::filesystem;

namespace {

constexpr std::uintmax_t kInputBytes =
    static_cast<std::uintmax_t>(himloco::kInputElements) * sizeof(float);
constexpr std::uintmax_t kOutputBytes =
    static_cast<std::uintmax_t>(himloco::kOutputElements) * sizeof(float);

struct Options {
  fs::path model_path;
  fs::path input_path;
  fs::path output_dir;
  fs::path report_path;
  int warmup = 10;
  int priority = -1;
};

struct InputRecord {
  std::int64_t source_index = 0;
  fs::path path;
};

struct OutputRecord {
  std::int64_t source_index = 0;
  fs::path input_path;
  fs::path output_path;
  double latency_ms = 0.0;
};

void PrintUsage(const char* program) {
  std::cout
      << "Usage: " << program << " --model-path MODEL.bin --input-path PATH "
      << "--output-dir DIR --report REPORT.json [--warmup N] "
      << "[--priority N]\n\n"
      << "PATH may be one numerically named float32 .bin file or a directory "
      << "of them. Each input must contain exactly 270 float32 values.\n";
}

int ParseInt(const std::string& text, const std::string& option) {
  std::size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(text, &parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument(option + " requires an integer, got: " + text);
  }
  if (parsed != text.size()) {
    throw std::invalid_argument(option + " requires an integer, got: " + text);
  }
  return value;
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help" || option == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + option);
    }
    const std::string value = argv[++index];
    if (option == "--model-path") {
      options.model_path = value;
    } else if (option == "--input-path") {
      options.input_path = value;
    } else if (option == "--output-dir") {
      options.output_dir = value;
    } else if (option == "--report") {
      options.report_path = value;
    } else if (option == "--warmup") {
      options.warmup = ParseInt(value, option);
    } else if (option == "--priority") {
      options.priority = ParseInt(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }

  if (options.model_path.empty() || options.input_path.empty() ||
      options.output_dir.empty() || options.report_path.empty()) {
    throw std::invalid_argument(
        "--model-path, --input-path, --output-dir, and --report are required");
  }
  if (options.warmup < 0) {
    throw std::invalid_argument("--warmup must be non-negative");
  }
  if (options.priority < -1 || options.priority > 255) {
    throw std::invalid_argument("--priority must be -1 or in [0,255]");
  }
  return options;
}

std::int64_t ParseSourceIndex(const fs::path& path) {
  const std::string stem = path.stem().string();
  std::size_t parsed = 0;
  std::int64_t source_index = 0;
  try {
    source_index = std::stoll(stem, &parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument(
        "input dump stem must be a rollout source index: " +
        path.filename().string());
  }
  if (parsed != stem.size() || source_index < 0) {
    throw std::invalid_argument(
        "input dump stem must be a non-negative rollout source index: " +
        path.filename().string());
  }
  return source_index;
}

std::vector<InputRecord> DiscoverInputs(const fs::path& input_path) {
  std::vector<fs::path> paths;
  if (fs::is_regular_file(input_path)) {
    paths.push_back(input_path);
  } else if (fs::is_directory(input_path)) {
    for (const fs::directory_entry& entry : fs::directory_iterator(input_path)) {
      if (entry.is_regular_file() && entry.path().extension() == ".bin") {
        paths.push_back(entry.path());
      }
    }
  } else {
    throw std::runtime_error("input path does not exist: " + input_path.string());
  }
  if (paths.empty()) {
    throw std::runtime_error("no .bin input dumps found in " +
                             input_path.string());
  }

  std::map<std::int64_t, fs::path> sorted;
  for (const fs::path& path : paths) {
    if (path.extension() != ".bin") {
      throw std::invalid_argument("input file must use .bin: " + path.string());
    }
    const std::int64_t source_index = ParseSourceIndex(path);
    if (fs::file_size(path) != kInputBytes) {
      throw std::runtime_error(path.string() + " must contain exactly " +
                               std::to_string(kInputBytes) + " bytes");
    }
    if (!sorted.emplace(source_index, path).second) {
      throw std::runtime_error("duplicate source index: " +
                               std::to_string(source_index));
    }
  }

  std::vector<InputRecord> records;
  records.reserve(sorted.size());
  for (const auto& item : sorted) {
    records.push_back({item.first, fs::absolute(item.second)});
  }
  return records;
}

std::vector<float> LoadInput(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open input: " + path.string());
  }
  std::vector<float> observation(himloco::kInputElements);
  stream.read(reinterpret_cast<char*>(observation.data()),
              static_cast<std::streamsize>(kInputBytes));
  if (!stream || stream.gcount() != static_cast<std::streamsize>(kInputBytes)) {
    throw std::runtime_error("failed to read complete input: " + path.string());
  }
  if (!std::all_of(observation.begin(), observation.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error("input contains NaN/Inf: " + path.string());
  }
  return observation;
}

fs::path OutputPath(const fs::path& directory, std::int64_t source_index) {
  std::ostringstream filename;
  filename << std::setfill('0') << std::setw(6) << source_index << ".bin";
  return directory / filename.str();
}

void WriteOutput(const fs::path& path, const std::vector<float>& actions) {
  if (actions.size() != static_cast<std::size_t>(himloco::kOutputElements)) {
    throw std::runtime_error("internal error: unexpected action count");
  }
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open output: " + path.string());
  }
  stream.write(reinterpret_cast<const char*>(actions.data()),
               static_cast<std::streamsize>(kOutputBytes));
  if (!stream) {
    throw std::runtime_error("failed to write output: " + path.string());
  }
}

std::string JsonString(const std::string& value) {
  std::ostringstream escaped;
  escaped << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        escaped << "\\\"";
        break;
      case '\\':
        escaped << "\\\\";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (character < 0x20U) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<int>(character) << std::dec;
        } else {
          escaped << static_cast<char>(character);
        }
    }
  }
  escaped << '"';
  return escaped.str();
}

std::string ShapeJson(const std::vector<int>& shape) {
  std::ostringstream stream;
  stream << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      stream << ',';
    }
    stream << shape[index];
  }
  stream << ']';
  return stream.str();
}

std::string ReadTextFile(const fs::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    return "unreported";
  }
  std::ostringstream content;
  content << stream.rdbuf();
  std::string result = content.str();
  while (!result.empty() &&
         (result.back() == '\n' || result.back() == '\r')) {
    result.pop_back();
  }
  return result.empty() ? "unreported" : result;
}

std::string MachineName() {
  struct utsname information {};
  if (uname(&information) != 0) {
    return "unreported";
  }
  return information.machine;
}

double Percentile(const std::vector<double>& sorted, double percentile) {
  if (sorted.empty()) {
    throw std::invalid_argument("cannot summarize empty latency values");
  }
  const double position =
      percentile * static_cast<double>(sorted.size() - 1U);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

void WriteReport(const Options& options, const himloco::HimLoco& model,
                 const std::vector<OutputRecord>& records) {
  std::vector<double> latencies;
  latencies.reserve(records.size());
  for (const OutputRecord& record : records) {
    latencies.push_back(record.latency_ms);
  }
  std::sort(latencies.begin(), latencies.end());
  const double mean =
      std::accumulate(latencies.begin(), latencies.end(), 0.0) /
      static_cast<double>(latencies.size());

  const fs::path report_parent = options.report_path.parent_path();
  if (!report_parent.empty()) {
    fs::create_directories(report_parent);
  }
  std::ofstream report(options.report_path);
  if (!report) {
    throw std::runtime_error("failed to open report: " +
                             options.report_path.string());
  }

  const himloco::TensorMetadata& input = model.input_metadata();
  const himloco::TensorMetadata& output = model.output_metadata();
  report << std::fixed << std::setprecision(6)
         << "{\n"
         << "  \"schema_version\": \"1.0\",\n"
         << "  \"platform\": \"RDK X5\",\n"
         << "  \"model\": " << JsonString(fs::absolute(options.model_path).string())
         << ",\n"
         << "  \"environment\": {\n"
         << "    \"board_os_version\": "
         << JsonString(ReadTextFile("/etc/version")) << ",\n"
         << "    \"machine\": " << JsonString(MachineName()) << ",\n"
         << "    \"dnn_runtime\": " << JsonString(model.runtime_version()) << "\n"
         << "  },\n"
         << "  \"runtime\": {\n"
         << "    \"model_name\": " << JsonString(model.model_name()) << ",\n"
         << "    \"input\": {\"name\": " << JsonString(input.name)
         << ", \"dtype\": \"F32\", \"valid_shape\": "
         << ShapeJson(input.valid_shape) << ", \"aligned_shape\": "
         << ShapeJson(input.aligned_shape) << ", \"aligned_byte_size\": "
         << input.aligned_byte_size << "},\n"
         << "    \"output\": {\"name\": " << JsonString(output.name)
         << ", \"dtype\": \"F32\", \"valid_shape\": "
         << ShapeJson(output.valid_shape) << ", \"aligned_shape\": "
         << ShapeJson(output.aligned_shape) << ", \"aligned_byte_size\": "
         << output.aligned_byte_size << "}\n"
         << "  },\n"
         << "  \"input_path\": " << JsonString(fs::absolute(options.input_path).string())
         << ",\n"
         << "  \"output_directory\": "
         << JsonString(fs::absolute(options.output_dir).string()) << ",\n"
         << "  \"sample_count\": " << records.size() << ",\n"
         << "  \"warmup_runs\": " << options.warmup << ",\n"
         << "  \"scheduling\": {\"priority\": ";
  if (model.priority() < 0) {
    report << "null, \"bpu_core\": \"Runtime default\"},\n";
  } else {
    report << model.priority()
           << ", \"bpu_core\": \"Runtime default\"},\n";
  }
  report << "  \"timing_scope\": \"hbDNNInfer plus hbDNNWaitTaskDone; input/output "
            "file I/O and cache maintenance excluded\",\n"
         << "  \"latency_ms\": {\n"
         << "    \"minimum\": " << latencies.front() << ",\n"
         << "    \"mean\": " << mean << ",\n"
         << "    \"p50\": " << Percentile(latencies, 0.50) << ",\n"
         << "    \"p95\": " << Percentile(latencies, 0.95) << ",\n"
         << "    \"maximum\": " << latencies.back() << "\n"
         << "  },\n"
         << "  \"sequential_throughput_fps\": " << 1000.0 / mean << ",\n"
         << "  \"records\": [\n";
  for (std::size_t index = 0; index < records.size(); ++index) {
    const OutputRecord& record = records[index];
    report << "    {\"source_index\": " << record.source_index
           << ", \"input_file\": " << JsonString(record.input_path.string())
           << ", \"output_file\": " << JsonString(record.output_path.string())
           << ", \"latency_ms\": " << record.latency_ms << '}';
    report << (index + 1U == records.size() ? "\n" : ",\n");
  }
  report << "  ]\n}\n";
  if (!report) {
    throw std::runtime_error("failed to write report: " +
                             options.report_path.string());
  }
}

void ValidateOutputTargets(const Options& options) {
  if (!fs::is_regular_file(options.model_path)) {
    throw std::runtime_error("model does not exist: " +
                             options.model_path.string());
  }
  if (options.model_path.extension() != ".bin") {
    throw std::invalid_argument("RDK X5 Runtime requires a .bin model");
  }
  if (fs::exists(options.output_dir)) {
    if (!fs::is_directory(options.output_dir)) {
      throw std::runtime_error("output path is not a directory: " +
                               options.output_dir.string());
    }
    if (!fs::is_empty(options.output_dir)) {
      throw std::runtime_error("output directory is not empty: " +
                               options.output_dir.string());
    }
  }
  if (fs::exists(options.report_path)) {
    throw std::runtime_error("report already exists: " +
                             options.report_path.string());
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    PrintUsage(argv[0]);
    return 2;
  }
  try {
    const Options options = ParseOptions(argc, argv);
    ValidateOutputTargets(options);
    const std::vector<InputRecord> inputs = DiscoverInputs(options.input_path);
    himloco::HimLoco model(options.model_path.string(), options.priority);

    const std::vector<float> warmup_input = LoadInput(inputs.front().path);
    std::cout << "[INFO] model=" << model.model_name()
              << " dnn_runtime=" << model.runtime_version()
              << " samples=" << inputs.size() << " warmup=" << options.warmup
              << std::endl;
    for (int index = 0; index < options.warmup; ++index) {
      model.Infer(warmup_input);
    }
    if (options.warmup > 0) {
      std::cout << "[INFO] warmup complete" << std::endl;
    }

    fs::create_directories(options.output_dir);
    std::vector<OutputRecord> outputs;
    outputs.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      const InputRecord& input = inputs[index];
      const himloco::InferenceResult result = model.Infer(LoadInput(input.path));
      const fs::path output_path = OutputPath(options.output_dir, input.source_index);
      WriteOutput(output_path, result.actions);
      outputs.push_back({input.source_index, input.path, fs::absolute(output_path),
                         result.latency_ms});
      std::cout << "[INFO] sample=" << index + 1U << '/' << inputs.size()
                << " source_index=" << input.source_index
                << " latency_ms=" << std::fixed << std::setprecision(6)
                << result.latency_ms << std::endl;
    }

    WriteReport(options, model, outputs);
    std::vector<double> latencies;
    latencies.reserve(outputs.size());
    for (const OutputRecord& output : outputs) {
      latencies.push_back(output.latency_ms);
    }
    const double mean =
        std::accumulate(latencies.begin(), latencies.end(), 0.0) /
        static_cast<double>(latencies.size());
    std::cout << "[INFO] complete mean_latency_ms=" << std::fixed
              << std::setprecision(6) << mean
              << " sequential_throughput_fps=" << 1000.0 / mean
              << " report=" << fs::absolute(options.report_path) << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
