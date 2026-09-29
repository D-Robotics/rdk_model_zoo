// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "preflight.h"
#include "sha256.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <unistd.h>
using namespace paraformer;
// Test-only identity reader; production library links the real shared reader.
rdk::NativeIdentity local_identity{"s100", "", "", ""};
namespace rdk {
NativeIdentity read_native_identity() { return local_identity; }
} // namespace rdk
template <class F> void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (const std::exception &) {
    caught = true;
  }
  assert(caught);
}
int main(int argc, char **argv) {
  assert(argc == 2);
  char dir[] = "/tmp/paraformer-preflight-XXXXXX";
  assert(mkdtemp(dir));
  ModelGroup group;
  const std::array<Stage, 3> stages{Stage::Encoder, Stage::Predictor,
                                    Stage::Decoder};
  for (size_t i = 0; i < 3; ++i) {
    auto path = (std::filesystem::path(dir) / std::to_string(i)).string();
    {
      std::ofstream f(path);
      f << "fixture-" << i;
    }
    group[i] = {{path, "s100", stages[i]},
                expected_asset_id(stages[i]),
                rdk::sha256_file(path)};
  }
  assert(expected_asset_id(Stage::Encoder) ==
         "s:paraformer:s100/paraformer_large_encoder_400x560_s100.hbm");
  const rdk::NativeIdentity identity{"s100", "", "", ""};
  verify_group(group, argv[1], identity);
  auto gate = make_preflight(group, argv[1]);
  for (const auto &a : group)
    gate(a.model);
  auto wrong = group[0].model;
  wrong.path = group[1].model.path;
  rejects([&] { gate(wrong); });
  local_identity.board = "RDK S100P";
  rejects([&] { gate(group[0].model); });
  local_identity.board.clear();
  auto reversed = group;
  std::swap(reversed[0], reversed[2]);
  verify_group(reversed, argv[1], identity);
  for (auto id :
       {rdk::NativeIdentity{}, rdk::NativeIdentity{"s600", "", "", ""},
        rdk::NativeIdentity{"s100", "RDK S100P", "", ""}})
    rejects([&] { verify_group(group, argv[1], id); });
  for (int failure = 0; failure < 9; ++failure) {
    auto bad = group;
    switch (failure) {
    case 0:
      bad[2].model.stage = Stage::Encoder;
      break;
    case 1:
      bad[2].asset_id = group[0].asset_id;
      break;
    case 2:
      bad[1].model.target = "s100p";
      break;
    case 3:
      bad[2].expected_sha256 = std::string(64, '0');
      break;
    case 4:
      bad[0].expected_sha256 = "unknown";
      break;
    case 5:
      bad[2].model.path = group[0].model.path;
      bad[2].expected_sha256 = group[0].expected_sha256;
      break;
    case 6:
      bad[2].model.path = dir;
      break;
    case 7:
      bad[2].model.path += "missing";
      break;
    case 8:
      bad[2].model.stage = static_cast<Stage>(99);
      break;
    }
    rejects([&] { verify_group(bad, argv[1], identity); });
  }
  rejects([&] { verify_group(group, group[0].model.path, identity); });
  const auto alias = (std::filesystem::path(dir) / "hardlink").string();
  std::filesystem::create_hard_link(group[0].model.path, alias);
  auto aliased = group;
  aliased[2].model.path = alias;
  aliased[2].expected_sha256 = group[0].expected_sha256;
  rejects([&] { verify_group(aliased, argv[1], identity); });
  {
    std::ofstream f(group[2].model.path);
    f << "changed";
  }
  rejects([&] { gate(group[0].model); });
  rejects([&] { verify_group(group, argv[1], identity); });
  {
    std::ofstream f(group[2].model.path);
  }
  group[2].expected_sha256 = rdk::sha256_file(group[2].model.path);
  rejects([&] { verify_group(group, argv[1], identity); });
  std::filesystem::remove_all(dir);
}
