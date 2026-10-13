// Board-only reference path for parity verification: replays the same
// detection JSONL through the genuine upstream HobotMot API with none of the
// sample wrapper code (no track.cpp, no C ABI), one upstream instance per
// class because upstream is class-agnostic, classes with a live instance but
// no detection still fed so their tracks age. The output JSON matches the
// schema written by runtime/python and runtime/cpp, so running all three on
// the same input and config proves the sample's validation and marshalling
// layers do not alter upstream tracking semantics.
//
// Usage:
//   parity_reference --input detections.jsonl --config iou2.json \
//                    --output reference.json [--frame-width 960]
//                    [--frame-height 544]
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hobot_mot.h"
#include "json/json.h"

namespace {

using hobot_mot::DataState;
using hobot_mot::MotBox;

struct RefDetection {
  float x1 = 0.F;
  float y1 = 0.F;
  float x2 = 0.F;
  float y2 = 0.F;
  float score = 0.F;
  int32_t class_id = 0;
};

struct RefRecord {
  int64_t frame_index = 0;
  int64_t timestamp_ms = 0;
  int32_t width = 0;   // 0 -> CLI default
  int32_t height = 0;
  std::vector<RefDetection> detections;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

struct Args {
  std::string input;
  std::string config;
  std::string output;
  int32_t frame_width = 960;
  int32_t frame_height = 544;
};

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const bool has_value = i + 1 < argc;
    if (flag == "--input" && has_value) {
      args.input = argv[++i];
    } else if (flag == "--config" && has_value) {
      args.config = argv[++i];
    } else if (flag == "--output" && has_value) {
      args.output = argv[++i];
    } else if (flag == "--frame-width" && has_value) {
      args.frame_width = std::atoi(argv[++i]);
    } else if (flag == "--frame-height" && has_value) {
      args.frame_height = std::atoi(argv[++i]);
    } else {
      fail("usage: parity_reference --input FILE --config FILE --output FILE "
           "[--frame-width N] [--frame-height N]");
    }
  }
  if (args.input.empty() || args.config.empty() || args.output.empty()) {
    fail("--input, --config and --output are required");
  }
  return args;
}

std::vector<RefRecord> load_records(const std::string &path) {
  std::ifstream stream(path);
  if (!stream.good()) {
    fail("cannot open input file: " + path);
  }
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  std::string errors;
  std::vector<RefRecord> records;
  std::string line;
  size_t line_no = 0;
  while (std::getline(stream, line)) {
    ++line_no;
    if (line.empty()) {
      continue;
    }
    Json::Value root;
    std::istringstream line_stream(line);
    if (!Json::parseFromStream(builder, line_stream, &root, &errors)) {
      fail("line " + std::to_string(line_no) + ": not valid JSON");
    }
    RefRecord record;
    record.frame_index = root["frame_index"].asInt64();
    record.timestamp_ms = root["timestamp_ms"].asInt64();
    if (root.isMember("image_width") && root["image_width"].isInt()) {
      record.width = root["image_width"].asInt();
      record.height = root["image_height"].asInt();
    }
    for (const auto &node : root["detections"]) {
      RefDetection det;
      const Json::Value &bbox = node["bbox"];
      det.x1 = bbox[0].asFloat();
      det.y1 = bbox[1].asFloat();
      det.x2 = bbox[2].asFloat();
      det.y2 = bbox[3].asFloat();
      det.score = node["score"].asFloat();
      det.class_id = node["class_id"].asInt();
      record.detections.push_back(det);
    }
    records.push_back(record);
  }
  if (records.empty()) {
    fail("input file has no frame records: " + path);
  }
  return records;
}

const char *state_name(DataState state) {
  switch (state) {
    case DataState::VALID:
      return "valid";
    case DataState::FILTERED:
      return "filtered";
    case DataState::INVISIBLE:
      return "invisible";
    case DataState::DISAPPEARED:
      return "disappeared";
    case DataState::INVALID:
      return "invalid";
    default:
      return "other";
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);
    const std::vector<RefRecord> records = load_records(args.input);

    // One genuine upstream instance per class; identities are per-instance.
    std::map<int32_t, std::unique_ptr<HobotMot>> per_class;
    Json::Value frames(Json::arrayValue);
    for (const auto &record : records) {
      const int32_t width = record.width > 0 ? record.width : args.frame_width;
      const int32_t height =
          record.height > 0 ? record.height : args.frame_height;
      std::map<int32_t, std::vector<size_t>> by_class;
      for (size_t i = 0; i < record.detections.size(); ++i) {
        by_class[record.detections[i].class_id].push_back(i);
      }
      for (const auto &entry : per_class) {
        by_class.try_emplace(entry.first);  // age tracks of absent classes
      }

      Json::Value frame(Json::objectValue);
      frame["frame_index"] = record.frame_index;
      frame["timestamp_ms"] = record.timestamp_ms;
      Json::Value tracks(Json::arrayValue);
      tracks.resize(record.detections.size());
      Json::Value gone(Json::arrayValue);
      for (const auto &[class_id, indexes] : by_class) {
        auto &instance = per_class[class_id];
        if (!instance) {
          instance = std::make_unique<HobotMot>(args.config);
        }
        std::vector<MotBox> in;
        in.reserve(indexes.size());
        for (const size_t i : indexes) {
          const RefDetection &det = record.detections[i];
          MotBox box;
          box.x1 = det.x1;
          box.y1 = det.y1;
          box.x2 = det.x2;
          box.y2 = det.y2;
          box.score = det.score;
          box.state_ = DataState::VALID;
          in.push_back(box);
        }
        std::vector<MotBox> out;
        std::vector<std::shared_ptr<MotTrackId>> disappeared;
        const int ret = instance->DoProcess(
            in, out, disappeared, static_cast<time_t>(record.timestamp_ms),
            static_cast<uint32_t>(width), static_cast<uint32_t>(height));
        if (ret != 0) {
          fail("upstream DoProcess failed with code " + std::to_string(ret));
        }
        if (out.size() != in.size()) {
          fail("upstream returned " + std::to_string(out.size()) +
               " rows for " + std::to_string(in.size()) + " inputs");
        }
        for (size_t i = 0; i < out.size(); ++i) {
          Json::Value row(Json::objectValue);
          row["class_id"] = class_id;
          row["track_id"] = out[i].id;
          Json::Value bbox(Json::arrayValue);
          bbox.append(out[i].x1);
          bbox.append(out[i].y1);
          bbox.append(out[i].x2);
          bbox.append(out[i].y2);
          row["bbox"] = bbox;
          row["score"] = out[i].score;
          row["state"] = state_name(out[i].state_);
          tracks[static_cast<Json::ArrayIndex>(indexes[i])] = row;
        }
        for (const auto &id : disappeared) {
          Json::Value row(Json::objectValue);
          row["class_id"] = class_id;
          row["track_id"] = id->value;
          gone.append(row);
        }
      }
      frame["tracks"] = tracks;
      frame["disappeared"] = gone;
      frames.append(frame);
    }

    std::ofstream out_stream(args.output);
    if (!out_stream.good()) {
      fail("cannot write output file: " + args.output);
    }
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "  ";
    out_stream << Json::writeString(writer, frames) << "\n";
    out_stream.close();
    if (!out_stream.good()) {
      fail("failed writing output file: " + args.output);
    }
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << "Error: " << exc.what() << "\n";
    return 2;
  }
}
