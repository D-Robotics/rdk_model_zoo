// CLI implementation: argument parsing, JSONL loading, rendering and JSON
// result output for the HobotMot replay runner.
#include "cli.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "json/json.h"

namespace hobot_mot_sample {
namespace {

[[noreturn]] void cli_error(const std::string &message) {
  throw std::runtime_error(message);
}

int32_t parse_int(const std::string &flag, const std::string &text) {
  try {
    size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != text.size() ||
        value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
      cli_error("value of " + flag + " must be a 32-bit integer, got '" +
                text + "'");
    }
    return static_cast<int32_t>(value);
  } catch (const std::invalid_argument &) {
    cli_error("value of " + flag + " must be an integer, got '" + text + "'");
  } catch (const std::out_of_range &) {
    cli_error("value of " + flag + " is out of int32 range: '" + text + "'");
  }
}

Detection parse_detection(const Json::Value &node, size_t line) {
  Detection det;
  if (!node.isObject() || !node.isMember("bbox") ||
      !node["bbox"].isArray() || node["bbox"].size() != 4 ||
      !node.isMember("score") || !node["score"].isNumeric() ||
      !node.isMember("class_id") || !node["class_id"].isInt()) {
    cli_error("line " + std::to_string(line) +
              ": each detection needs numeric bbox[4], score and integer "
              "class_id");
  }
  const Json::Value &bbox = node["bbox"];
  det.x1 = bbox[0].asFloat();
  det.y1 = bbox[1].asFloat();
  det.y2 = bbox[3].asFloat();
  det.x2 = bbox[2].asFloat();
  det.score = node["score"].asFloat();
  det.class_id = node["class_id"].asInt();
  return det;
}

// Names must match runtime/python/track.py STATE_NAMES so the JSON written by
// both runtimes is directly comparable (see tests/compare_results.py).
const char *state_name(TrackState state) {
  switch (state) {
    case TrackState::kValid:
      return "valid";
    case TrackState::kFiltered:
      return "filtered";
    case TrackState::kInvisible:
      return "invisible";
    case TrackState::kDisappeared:
      return "disappeared";
    case TrackState::kInvalid:
      return "invalid";
    default:
      return "other";
  }
}

}  // namespace

Options parse_options(const std::vector<std::string> &args) {
  Options options;
  options.input_path = HOBOT_MOT_SAMPLE_ROOT "/test_data/detections.jsonl";
  options.config_path = HOBOT_MOT_SAMPLE_ROOT "/model/configs/iou2.json";
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &arg = args[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= args.size()) {
        cli_error("missing value for " + arg);
      }
      return args[++i];
    };
    if (arg == "--input") {
      options.input_path = value();
    } else if (arg == "--config") {
      options.config_path = value();
    } else if (arg == "--output") {
      options.output_path = value();
    } else if (arg == "--frame-width") {
      options.frame_width = parse_int(arg, value());
    } else if (arg == "--frame-height") {
      options.frame_height = parse_int(arg, value());
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "HobotMot (IOU 2.0) tracker replay\n"
          << "  --input PATH        detection JSONL (default sample test_data)\n"
          << "  --config PATH       IOU_2.0 tracker config JSON\n"
          << "  --output PATH       optional JSON result file\n"
          << "  --frame-width N     default frame width when the record has "
             "none (960)\n"
          << "  --frame-height N    default frame height when the record has "
             "none (544)\n";
      std::exit(0);
    } else {
      cli_error("unknown argument '" + arg +
                "' (supported: --input --config --output --frame-width "
                "--frame-height)");
    }
  }
  return options;
}

std::vector<FrameRecord> load_frame_records(const std::string &input_path) {
  std::ifstream stream(input_path);
  if (!stream.good()) {
    cli_error("cannot open input file: " + input_path);
  }
  std::vector<FrameRecord> records;
  std::string line;
  size_t line_no = 0;
  int64_t last_ts = -1;
  int64_t last_index = -1;
  while (std::getline(stream, line)) {
    ++line_no;
    if (line.empty()) {
      continue;
    }
    Json::Value root;
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    std::string errors;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(line.data(), line.data() + line.size(), &root,
                       &errors)) {
      cli_error("line " + std::to_string(line_no) +
                " is not valid JSON: " + errors);
    }
    if (!root.isObject() || !root.isMember("frame_index") ||
        !root["frame_index"].isInt64() || !root.isMember("timestamp_ms") ||
        !root["timestamp_ms"].isInt64() || !root.isMember("detections") ||
        !root["detections"].isArray()) {
      cli_error("line " + std::to_string(line_no) +
                " needs integer frame_index, integer timestamp_ms and "
                "detections[]");
    }
    FrameRecord record;
    record.frame_index = root["frame_index"].asInt64();
    record.timestamp_ms = root["timestamp_ms"].asInt64();
    if (record.timestamp_ms <= last_ts) {
      cli_error("line " + std::to_string(line_no) +
                ": timestamp_ms must strictly increase");
    }
    last_ts = record.timestamp_ms;
    if (record.frame_index <= last_index) {
      cli_error("line " + std::to_string(line_no) +
                ": frame_index must strictly increase");
    }
    last_index = record.frame_index;
    if (root.isMember("image_width") || root.isMember("image_height")) {
      if (!root.isMember("image_width") || !root.isMember("image_height") ||
          !root["image_width"].isInt() || !root["image_height"].isInt() ||
          root["image_width"].asInt() <= 0 ||
          root["image_height"].asInt() <= 0) {
        cli_error("line " + std::to_string(line_no) +
                  ": image_width and image_height must be positive integers "
                  "when present");
      }
      record.image_width = root["image_width"].asInt();
      record.image_height = root["image_height"].asInt();
    }
    for (const auto &node : root["detections"]) {
      record.detections.push_back(parse_detection(node, line_no));
    }
    records.push_back(record);
  }
  if (records.empty()) {
    cli_error("input file has no frame records: " + input_path);
  }
  return records;
}

void render_frames(const std::vector<FrameTracks> &results,
                   const std::vector<FrameRecord> &records) {
  for (size_t i = 0; i < results.size() && i < records.size(); ++i) {
    std::cout << "frame " << records[i].frame_index << " ts "
              << results[i].timestamp_ms << ": ";
    for (const auto &track : results[i].tracks) {
      std::cout << "[cls " << track.class_id << " id " << track.track_id
                << " (" << track.x1 << "," << track.y1 << "," << track.x2
                << "," << track.y2 << ") score " << track.score << " "
                << state_name(track.state) << "] ";
    }
    for (const auto &gone : results[i].removed_ids) {
      std::cout << "[removed cls " << gone.class_id << " id "
                << gone.track_id << "] ";
    }
    std::cout << "\n";
  }
}

void save_results(const std::string &output_path,
                  const std::vector<FrameRecord> &records,
                  const std::vector<FrameTracks> &results) {
  Json::Value root(Json::arrayValue);
  for (size_t i = 0; i < results.size() && i < records.size(); ++i) {
    Json::Value frame(Json::objectValue);
    frame["frame_index"] = records[i].frame_index;
    frame["timestamp_ms"] = results[i].timestamp_ms;
    Json::Value tracks(Json::arrayValue);
    for (const auto &track : results[i].tracks) {
      Json::Value row(Json::objectValue);
      row["class_id"] = track.class_id;
      row["track_id"] = track.track_id;
      Json::Value bbox(Json::arrayValue);
      bbox.append(track.x1);
      bbox.append(track.y1);
      bbox.append(track.x2);
      bbox.append(track.y2);
      row["bbox"] = bbox;
      row["score"] = track.score;
      row["state"] = state_name(track.state);
      tracks.append(row);
    }
    frame["tracks"] = tracks;
    Json::Value gone(Json::arrayValue);
    for (const auto &removed : results[i].removed_ids) {
      Json::Value row(Json::objectValue);
      row["class_id"] = removed.class_id;
      row["track_id"] = removed.track_id;
      gone.append(row);
    }
    frame["disappeared"] = gone;
    root.append(frame);
  }
  std::ofstream out(output_path);
  if (!out.good()) {
    cli_error("cannot write output file: " + output_path);
  }
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "  ";
  out << Json::writeString(writer, root) << "\n";
  out.close();
  if (!out.good()) {
    cli_error("failed writing output file: " + output_path);
  }
}

}  // namespace hobot_mot_sample
