// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "asr.h"
#include "cli_io.h"
#include "preflight.h"
#include "sha256.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
namespace fs = std::filesystem;
using Json = nlohmann::json;
namespace {
void save(const fs::path &path, const Json &report) {
  const fs::path temporary = path.string() + ".tmp";
  std::ofstream out(temporary, std::ios::binary);
  if (!out)
    throw std::runtime_error("Cannot create report: " + temporary.string());
  out << report.dump(2) << '\n';
  out.close();
  if (!out)
    throw std::runtime_error("Cannot write report: " + temporary.string());
  fs::rename(temporary, path);
}
std::string file_digest(const std::string &path) {
  if (!fs::is_regular_file(path) || fs::file_size(path) == 0)
    throw std::invalid_argument("Missing or empty regular input: " + path);
  const auto digest = rdk::sha256_file(path);
  if (digest.size() != 64)
    throw std::runtime_error("Cannot hash input: " + path);
  return digest;
}
} // namespace
int main(int argc, char **argv) {
  fs::path output;
  Json report;
  try {
    const auto options = asr::parse_cli(argc, argv);
    if (options.help) {
      std::cout << asr::cli_help();
      return 0;
    }
    auto gate = asr::make_preflight(options.model_sha256, options.vocabulary);
    gate(options.model);
    const auto vocabulary = asr::load_vocabulary(options.vocabulary);
    const auto audio_sha = file_digest(options.audio);
    if (fs::exists(options.output) || fs::is_symlink(options.output))
      throw std::invalid_argument("Output directory must be new");
    const fs::path destination = options.output;
    if (!destination.parent_path().empty())
      fs::create_directories(destination.parent_path());
    if (!fs::create_directory(destination))
      throw std::runtime_error("Cannot create new output directory");
    output = destination;
#ifdef ASR_HOST_FIXTURE
    const char *backend = "host-fixture";
#else
    const char *backend = "native-sdk";
#endif
    report = {{"schema", "rdk-model-zoo/asr-native-run/v1"},
              {"status", "running"},
              {"execution_backend", backend},
              {"target", options.model.target},
              {"asset_id", options.asset_id},
              {"model_sha256", options.model_sha256},
              {"audio_sha256", audio_sha},
              {"vocabulary_sha256", asr::kVocabularySha256},
              {"decode_mode", options.decode_mode},
              {"frontend", "libsamplerate-sinc-best; independent windows; "
                           "var+1e-5 before padding"},
              {"config", {{"audio_maxlen", 30000}, {"new_rate", 16000}}},
              {"chunks", Json::array()}};
    asr::SdkRunner runner(options.model, gate);
    const auto &meta = runner.metadata();
    report["metadata"] = {{"model_name", meta.model_name},
                          {"input_shape", {1, 30000}},
                          {"output_shape", {1, meta.steps, 3503}},
                          {"dtype", "float32"},
                          {"input_strides", meta.input_strides},
                          {"output_strides", meta.output_strides},
                          {"input_bytes", meta.input_bytes},
                          {"output_bytes", meta.output_bytes}};
    asr::ASR task(
        [&runner](const std::vector<float> &values) {
          return runner.infer(values);
        },
        meta.steps, vocabulary,
        options.decode_mode == "ctc" ? asr::DecodeMode::Ctc
                                     : asr::DecodeMode::Legacy);
    asr::AudioReader reader(options.audio);
    asr::AudioChunk chunk;
    std::string text;
    while (reader.next(chunk)) {
      auto prepared = task.pre_process(chunk);
      const auto transcript = task.post_process(task.forward(prepared));
      text += transcript;
      report["chunks"].push_back(
          {{"index", chunk.index},
           {"source_start", chunk.source_start},
           {"source_frames", chunk.samples.size() / size_t(chunk.channels)},
           {"source_rate", chunk.sample_rate},
           {"valid_target_samples", prepared.valid_samples},
           {"text", transcript}});
    }
    if (report["chunks"].empty())
      throw std::invalid_argument("No audio chunks processed");
    if (file_digest(options.audio) != audio_sha ||
        file_digest(options.model.path) != options.model_sha256 ||
        file_digest(options.vocabulary) != asr::kVocabularySha256)
      throw std::runtime_error(
          "Input/model/vocabulary changed during inference");
    report["status"] = "completed";
    report["text"] = text;
    save(output / "result.json", report);
    std::cout << text << "\nReport: " << (output / "result.json").string()
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    if (!output.empty())
      try {
        report["status"] = "failed";
        report["error"] = e.what();
        save(output / "failed.json", report);
      } catch (const std::exception &report_error) {
        std::cerr << "Could not save failure report: " << report_error.what()
                  << '\n';
      }
    std::cerr << "error: " << e.what() << '\n';
    return 2;
  }
}
