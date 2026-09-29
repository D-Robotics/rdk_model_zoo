// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli_io.h"
#include "feature_io.h"
#include "pipeline.h"
#include "sha256.h"
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
namespace fs = std::filesystem;
using Json = nlohmann::json;
namespace {
std::string utc() {
  const auto now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::ostringstream out;
  out << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}
void save(const fs::path &path, const Json &report) {
  const auto temporary = path.string() + ".tmp";
  std::ofstream file(temporary, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot create report");
  file << report.dump(2) << '\n';
  file.close();
  if (!file)
    throw std::runtime_error("Cannot write report");
  fs::rename(temporary, path);
}
Json tensors(const std::vector<paraformer::TensorMetadata> &metadata) {
  Json out = Json::array();
  for (const auto &m : metadata)
    out.push_back({{"name", m.name},
                   {"role", m.role},
                   {"dtype", m.dtype},
                   {"shape", m.shape},
                   {"strides", m.strides},
                   {"allocation_bytes", m.allocation_bytes}});
  return out;
}
std::string digest(const std::string &path) {
  auto value = rdk::sha256_file(path);
  if (value.size() != 64)
    throw std::invalid_argument("Cannot hash file: " + path);
  return value;
}
} // namespace
int main(int argc, char **argv) {
  fs::path output;
  Json report;
  std::string current;
  try {
    const auto options = paraformer::parse_cli(argc, argv);
#ifdef PARAFORMER_HOST_FIXTURE
    const std::string backend = "host-fixture";
#else
    const std::string backend = "native-sdk";
#endif
    if (options.help) {
      std::cout << paraformer::cli_help() << "Backend: " << backend << '\n';
      return 0;
    }
    auto gate = paraformer::make_preflight(options.models, options.vocabulary);
    const auto vocabulary = paraformer::load_vocabulary(options.vocabulary);
    const auto manifest_sha = digest(options.manifest);
    const auto items =
        paraformer::load_prepared_manifest(options.manifest, options.max_utts);
    if (digest(options.manifest) != manifest_sha)
      throw std::runtime_error("Manifest changed while reading");
    if (fs::exists(options.output) || fs::is_symlink(options.output))
      throw std::invalid_argument("Output directory must be new");
    const fs::path destination = options.output;
    if (!destination.parent_path().empty())
      fs::create_directories(destination.parent_path());
    if (!fs::create_directory(destination))
      throw std::runtime_error("Cannot create new output directory");
    output = destination;
    report = {{"schema", "rdk-model-zoo/paraformer-native-run/v1"},
              {"status", "running"},
              {"execution_backend", backend},
              {"target", "s100"},
              {"started_utc", utc()},
              {"manifest_path", fs::absolute(options.manifest).string()},
              {"manifest_sha256", manifest_sha},
              {"vocabulary_sha256", paraformer::kVocabularySha256},
              {"inference_attempted", false},
              {"inference_executed", false},
              {"models", Json::array()},
              {"records", Json::array()}};
    std::array<std::unique_ptr<paraformer::SdkRunner>, 3> runners;
    const std::array<std::string, 3> stages{"encoder", "predictor", "decoder"};
    for (size_t i = 0; i < 3; ++i) {
      const auto &artifact = options.models[i];
      report["models"].push_back(
          {{"stage", stages[i]},
           {"asset_id", artifact.asset_id},
           {"path", fs::absolute(artifact.model.path).string()},
           {"sha256", artifact.expected_sha256}});
    }
    for (size_t i = 0; i < 3; ++i) {
      const auto &artifact = options.models[i];
      runners[i] =
          std::make_unique<paraformer::SdkRunner>(artifact.model, gate);
      const auto &meta = runners[i]->metadata();
      report["models"][i]["metadata"] = {{"model_name", meta.model_name},
                                         {"inputs", tensors(meta.inputs)},
                                         {"outputs", tensors(meta.outputs)}};
    }
    paraformer::Pipeline pipeline(
        [&](const std::vector<float> &features) {
          auto out = runners[0]->infer({{"features", features}});
          return std::move(std::get<std::vector<float>>(out.at("context")));
        },
        [&](const std::vector<float> &context) {
          auto out = runners[1]->infer({{"context", context}});
          return paraformer::PredictorOutput{
              std::move(std::get<std::vector<float>>(out.at("alphas"))),
              std::move(std::get<std::vector<float>>(out.at("hidden")))};
        },
        [&](const paraformer::DecoderInput &input) {
          auto out = runners[2]->infer(
              {{"context", input.context},
               {"acoustic", input.acoustic},
               {"count", std::vector<int32_t>{input.token_count}},
               {"bias",
                std::vector<float>(input.bias.begin(), input.bias.end())}});
          return std::move(std::get<std::vector<float>>(out.at("logits")));
        },
        vocabulary);
    for (const auto &item : items) {
      current = item.utt_id;
      const auto features = paraformer::load_features(item);
      report["inference_attempted"] = true;
      if (report["records"].empty())
        report["inference_executed"] = nullptr;
      const auto result = pipeline.predict(features, item.valid_frames);
      report["inference_executed"] = true;
      Json timings = {{"encoder_ms", result.timings.encoder_ms},
                      {"predictor_ms", result.timings.predictor_ms},
                      {"cif_ms", result.timings.cif_ms},
                      {"decoder_ms", result.timings.decoder_ms
                                         ? Json(*result.timings.decoder_ms)
                                         : Json(nullptr)}};
      Json record = {{"utt_id", item.utt_id},
                     {"feature_path", item.path},
                     {"feature_sha256", item.sha256},
                     {"valid_frames", item.valid_frames},
                     {"original_frames", item.original_frames},
                     {"truncated", item.truncated},
                     {"source_record", Json::parse(item.original_record_json)},
                     {"text", result.text},
                     {"token_ids", result.token_ids},
                     {"token_count", result.token_count},
                     {"decoder_executed", result.decoder_executed},
                     {"timings", timings}};
      if (item.reference_text)
        record["reference_text"] = *item.reference_text;
      report["records"].push_back(std::move(record));
      std::cout << item.utt_id << ": " << result.text << '\n';
    }
    current.clear();
    gate(options.models[0].model);
    if (digest(options.manifest) != manifest_sha)
      throw std::runtime_error("Manifest changed during inference");
    for (const auto &item : items)
      if (digest(item.path) != item.sha256)
        throw std::runtime_error("Feature changed during inference: " +
                                 item.utt_id);
    report["status"] = "completed";
    report["finished_utc"] = utc();
    save(output / "result.json", report);
    std::cout << "Report: " << (output / "result.json").string() << '\n';
    return 0;
  } catch (const std::exception &error) {
    if (!output.empty())
      try {
        report["status"] = "failed";
        report["finished_utc"] = utc();
        report["error"] = error.what();
        report["current_utt_id"] = current;
        save(output / "failed.json", report);
      } catch (const std::exception &failure) {
        std::cerr << "Could not save failure report: " << failure.what()
                  << '\n';
      }
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
