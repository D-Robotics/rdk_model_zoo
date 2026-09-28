// Host feature reader probe; does not load any SDK or execute a model.
#include "feature_io.h"
#include "sha256.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc, char **argv) {
  try {
    if (argc != 2 && argc != 3)
      throw std::invalid_argument("Expected prepared manifest");
    nlohmann::json output = nlohmann::json::array();
    for (const auto &item : paraformer::load_prepared_manifest(
             argv[1], argc == 3 ? std::stoull(argv[2]) : 0)) {
      const auto features = paraformer::load_features(item);
      output.push_back(
          {{"utt_id", item.utt_id},
           {"frames", item.valid_frames},
           {"original_frames", item.original_frames},
           {"truncated", item.truncated},
           {"size", features.size()},
           {"values_sha256",
            rdk::sha256_hex(features.data(), features.size() * sizeof(float))},
           {"source_record",
            nlohmann::json::parse(item.original_record_json)}});
    }
    std::cout << output.dump() << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
