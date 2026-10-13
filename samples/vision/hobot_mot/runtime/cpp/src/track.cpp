// HobotMot tracker model: config validation, per-class upstream routing and
// the preprocess -> infer -> postprocess chain. See inc/track.hpp for the
// interface contract and inc/mot_c_api.cpp for the C boundary.
#include "track.hpp"

#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "hobot_mot.h"
#include "json/json.h"

namespace hobot_mot_sample {
namespace {

using hobot_mot::MotBox;
using hobot_mot::MotTrackId;

// Field contract of upstream IOU2Param (include/data_type/mot_data_type.hpp):
// JSON type each macro expansion actually reads, and the range enforced here.
struct FieldSpec {
  const char *name;
  char kind;  // 'b' bool, 'i' int, 'u' unsigned int, 'd' double, 's' string
  double min;
  double max;   // NaN means no upper bound
  bool has_max;
};

// Integer bounds mirror the upstream C++ members: jsoncpp isInt/isUInt
// silently reject out-of-range integers (keeping the default), so values
// beyond int32/uint32 are rejected here instead.
const double kInt32Max = 2147483647.0;
const double kUint32Max = 4294967295.0;
// Coordinates must stay inside this magnitude so the int conversion and the
// float32 transport of the C ABI are both exact and defined.
const double kCoordMax = 1073741824.0;  // 2^30

const FieldSpec kFields[] = {
    {"match_type", 's', 0, 0, false},
    {"use_kalman_filter", 'b', 0, 0, false},
    {"missing_time_thres", 'i', 0, kInt32Max, true},
    {"vanish_frame_count", 'u', 0, kUint32Max, true},
    {"time_gap", 'u', 1, kUint32Max, true},
    {"remove_obsolete_track", 'i', 0, kInt32Max, true},
    {"iou_thres", 'd', 0, 1, true},
    {"euclidean_thres", 'd', 0, 0, false},
    {"use_location_gain", 'i', 0, kInt32Max, true},
    {"max_trajectory_number", 'i', 1, kInt32Max, true},
    {"min_score", 'd', 0, 1, true},
    {"ignore_overlap_thres", 'd', 0, 1, true},
};

bool json_has_kind(const Json::Value &value, char kind) {
  switch (kind) {
    case 'b':
      return value.isBool();
    case 'i':
      return value.isInt();
    case 'u':
      return value.isUInt();
    case 'd':
      return value.isNumeric();
    default:
      return value.isString();
  }
}

const char *kind_name(char kind) {
  switch (kind) {
    case 'b':
      return "bool";
    case 'i':
      return "int";
    case 'u':
      return "unsigned int";
    case 'd':
      return "number";
    default:
      return "string";
  }
}

void validate_field(const FieldSpec &spec, const Json::Value &value) {
  // Upstream silently keeps the default when the JSON type does not match
  // (for example "use_kalman_filter": 1 is ignored because only isBool()
  // reads it). Reject here so the config a user wrote is the config that runs.
  if (!json_has_kind(value, spec.kind)) {
    std::ostringstream message;
    message << "config field '" << spec.name << "' must be JSON "
            << kind_name(spec.kind) << ", got " << value.type()
            << "; upstream would silently ignore this value";
    throw MotConfigError(message.str());
  }
  if (spec.kind == 'd' || spec.kind == 'i' || spec.kind == 'u') {
    const double number = value.asDouble();
    if (!std::isfinite(number) || number < spec.min ||
        (spec.has_max && number > spec.max)) {
      std::ostringstream message;
      message << "config field '" << spec.name << "' = " << number
              << " out of range [" << spec.min << ", "
              << (spec.has_max ? std::to_string(spec.max) : "inf") << "]";
      throw MotConfigError(message.str());
    }
  } else if (spec.kind == 's' && std::string(spec.name) == "match_type") {
    const std::string text = value.asString();
    if (text != "IOU" && text != "Euclidean") {
      throw MotConfigError(
          "config field 'match_type' must be \"IOU\" or \"Euclidean\", got \"" +
          text + "\"");
    }
  }
}

// Resolve an explicit frame dimension: 0 means "unspecified" (the Python
// runtime passes None the same way), anything negative is rejected instead
// of silently falling back to the default.
int32_t resolve_dimension(int32_t explicit_value, int32_t default_value,
                          const char *name) {
  if (explicit_value == 0) {
    return default_value;
  }
  if (explicit_value < 0) {
    throw MotInputError(std::string(name) +
                        " must be positive when given, got " +
                        std::to_string(explicit_value));
  }
  return explicit_value;
}

}  // namespace

std::string validate_config_file(const std::string &config_path) {
  std::ifstream stream(config_path);
  if (!stream.good()) {
    throw MotConfigError("cannot open tracker config: " + config_path);
  }
  std::stringstream buffer;
  buffer << stream.rdbuf();
  const std::string content = buffer.str();

  Json::Value root;
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  std::string errors;
  if (!Json::parseFromStream(builder, buffer, &root, &errors)) {
    throw MotConfigError("tracker config is not valid JSON: " + errors);
  }
  if (!root.isObject()) {
    throw MotConfigError("tracker config must be a JSON object");
  }
  // The upstream build only compiles the IOU2 path; any other tracker_type
  // (or a missing one, in a config file that exists) leaves the method
  // pointer null and crashes on the first DoProcess.
  if (!root.isMember("tracker_type") || !root["tracker_type"].isString() ||
      root["tracker_type"].asString() != "IOU_2.0") {
    throw MotConfigError(
        "config field 'tracker_type' must be the string \"IOU_2.0\" "
        "(the only tracker compiled in the pinned upstream)");
  }
  for (const auto &spec : kFields) {
    if (root.isMember(spec.name)) {
      validate_field(spec, root[spec.name]);
    }
  }
  return content;
}

// One upstream HobotMot instance plus the id namespace of its class.
class HobotMotTracker::UpstreamMot {
 public:
  explicit UpstreamMot(const std::string &config_path)
      : mot_(std::make_unique<HobotMot>(config_path)) {}

  int process(const std::vector<Detection> &detections, int64_t timestamp_ms,
              uint32_t width, uint32_t height, std::vector<TrackResult> &rows,
              std::vector<DisappearedId> &gone) {
    std::vector<MotBox> in;
    in.reserve(detections.size());
    for (const auto &det : detections) {
      MotBox box;
      box.x1 = det.x1;
      box.y1 = det.y1;
      box.x2 = det.x2;
      box.y2 = det.y2;
      box.score = det.score;
      box.state_ = hobot_mot::DataState::VALID;
      in.push_back(box);
    }
    std::vector<MotBox> out;
    std::vector<std::shared_ptr<MotTrackId>> disappeared;
    const int ret = mot_->DoProcess(in, out, disappeared,
                                    static_cast<time_t>(timestamp_ms), width,
                                    height);
    if (ret != 0) {
      return ret;
    }
    // Upstream returns one box per input, in input order: matched entries
    // carry id and tracker coordinates, the rest stay id -1 / INVALID.
    if (out.size() != in.size()) {
      return -1;
    }
    rows.clear();
    rows.reserve(out.size());
    for (const auto &box : out) {
      TrackResult row;
      row.x1 = box.x1;
      row.y1 = box.y1;
      row.x2 = box.x2;
      row.y2 = box.y2;
      row.score = box.score;
      row.track_id = box.id;
      row.state = static_cast<TrackState>(static_cast<int>(box.state_));
      rows.push_back(row);
    }
    gone.clear();
    for (const auto &id : disappeared) {
      DisappearedId removed;
      removed.track_id = static_cast<int32_t>(id->value);
      gone.push_back(removed);
    }
    return 0;
  }

 private:
  std::unique_ptr<HobotMot> mot_;
};

HobotMotTracker::HobotMotTracker(const TrackOptions &options)
    : options_(options) {
  if (options_.frame_width <= 0 || options_.frame_height <= 0) {
    throw MotConfigError("frame_width and frame_height must be positive");
  }
  // Snapshot the validated content: per-class upstream instances are built
  // lazily, and a config file edited to an invalid state after construction
  // would crash upstream (it ignores Init failures). ensure_config_unchanged
  // rejects any later edit before native state is touched.
  config_content_ = validate_config_file(options_.config_path);
}

void HobotMotTracker::ensure_config_unchanged() const {
  if (validate_config_file(options_.config_path) != config_content_) {
    throw MotConfigError(
        "tracker config file changed after construction: " +
        options_.config_path);
  }
}

// Shared input gate for preprocess and the public infer. Everything that
// crosses into upstream code as int is checked to stay representable, and
// coordinates stay inside +/-2^30 so lround and the float32 transport of
// the C ABI are exact.
void HobotMotTracker::validate_frame(const PreparedFrame &prepared) const {
  if (prepared.frame_width <= 0 || prepared.frame_height <= 0) {
    throw MotInputError("frame size must be positive, got " +
                        std::to_string(prepared.frame_width) + "x" +
                        std::to_string(prepared.frame_height));
  }
  if (prepared.timestamp_ms < 0) {
    throw MotInputError("timestamp_ms must be >= 0, got " +
                        std::to_string(prepared.timestamp_ms));
  }
  if (prepared.timestamp_ms <= last_timestamp_ms_) {
    throw MotInputError(
        "timestamp_ms must strictly increase: previous " +
        std::to_string(last_timestamp_ms_) + ", got " +
        std::to_string(prepared.timestamp_ms));
  }
  for (size_t i = 0; i < prepared.detections.size(); ++i) {
    const Detection &det = prepared.detections[i];
    const float coords[4] = {det.x1, det.y1, det.x2, det.y2};
    for (const float value : coords) {
      if (!std::isfinite(value) || std::fabs(value) > kCoordMax) {
        throw MotInputError("detection " + std::to_string(i) +
                            ": coordinates must be finite and within +-" +
                            std::to_string(static_cast<int64_t>(kCoordMax)));
      }
    }
    if (det.x2 <= det.x1 || det.y2 <= det.y1) {
      throw MotInputError("detection " + std::to_string(i) +
                          ": box extents must be positive (x2 > x1 and "
                          "y2 > y1)");
    }
    if (!std::isfinite(det.score) || det.score < 0.F || det.score > 1.F) {
      throw MotInputError("detection " + std::to_string(i) +
                          ": score must be within [0, 1]");
    }
  }
}

PreparedFrame HobotMotTracker::preprocess(
    const std::vector<Detection> &detections, int64_t timestamp_ms,
    int32_t frame_width, int32_t frame_height) const {
  const int32_t width = resolve_dimension(frame_width, options_.frame_width,
                                          "frame_width");
  const int32_t height = resolve_dimension(frame_height, options_.frame_height,
                                           "frame_height");
  PreparedFrame prepared;
  prepared.timestamp_ms = timestamp_ms;
  prepared.frame_width = width;
  prepared.frame_height = height;
  prepared.detections = detections;
  validate_frame(prepared);
  return prepared;
}

RawMotFrame HobotMotTracker::infer(const PreparedFrame &prepared) {
  // Re-run the gate so staged callers cannot bypass preprocess validation
  // or replay a stale timestamp.
  validate_frame(prepared);

  // Group detections by class, remembering each row's input index. Classes
  // that have a live instance but no detection this frame are still fed an
  // empty frame so their tracks age and can disappear.
  std::map<int32_t, std::vector<size_t>> by_class;
  for (size_t i = 0; i < prepared.detections.size(); ++i) {
    by_class[prepared.detections[i].class_id].push_back(i);
  }
  for (const auto &entry : per_class_) {
    by_class.try_emplace(entry.first);
  }
  bool needs_new_instance = false;
  for (const auto &[class_id, indexes] : by_class) {
    (void)indexes;
    if (per_class_.find(class_id) == per_class_.end()) {
      needs_new_instance = true;
    }
  }
  if (needs_new_instance) {
    ensure_config_unchanged();
  }

  RawMotFrame raw;
  for (const auto &[class_id, indexes] : by_class) {
    auto &instance = per_class_[class_id];
    if (!instance) {
      instance = std::make_shared<UpstreamMot>(options_.config_path);
    }
    std::vector<Detection> class_detections;
    class_detections.reserve(indexes.size());
    for (const size_t i : indexes) {
      class_detections.push_back(prepared.detections[i]);
    }
    RawMotFrame::ClassOutput output;
    output.class_id = class_id;
    const int ret = instance->process(class_detections, prepared.timestamp_ms,
                                      static_cast<uint32_t>(prepared.frame_width),
                                      static_cast<uint32_t>(prepared.frame_height),
                                      output.rows, output.gone);
    if (ret != 0) {
      throw std::runtime_error("upstream HobotMot DoProcess failed with code " +
                               std::to_string(ret));
    }
    output.detection_indexes.assign(indexes.begin(), indexes.end());
    raw.classes.push_back(std::move(output));
  }
  last_timestamp_ms_ = prepared.timestamp_ms;
  return raw;
}

FrameTracks HobotMotTracker::postprocess(const RawMotFrame &raw,
                                         const PreparedFrame &prepared) const {
  // Raw rows and indexes come back from this class's own infer, but the
  // staged API is public: verify sizes, bounds and uniqueness before any
  // write into the result row space.
  std::vector<bool> filled(prepared.detections.size(), false);
  for (const auto &output : raw.classes) {
    if (output.rows.size() != output.detection_indexes.size()) {
      throw std::runtime_error("raw output rows and indexes disagree");
    }
    for (const int32_t index : output.detection_indexes) {
      if (index < 0 ||
          static_cast<size_t>(index) >= prepared.detections.size()) {
        throw std::runtime_error("raw output index out of range: " +
                                 std::to_string(index));
      }
      if (filled[index]) {
        throw std::runtime_error("raw output index used twice: " +
                                 std::to_string(index));
      }
      filled[index] = true;
    }
  }
  FrameTracks result;
  result.timestamp_ms = prepared.timestamp_ms;
  for (const auto &output : raw.classes) {
    for (const auto &gone : output.gone) {
      DisappearedId removed;
      removed.class_id = output.class_id;
      removed.track_id = gone.track_id;
      result.removed_ids.push_back(removed);
    }
  }
  result.tracks.resize(prepared.detections.size());
  for (const auto &output : raw.classes) {
    for (size_t i = 0; i < output.rows.size(); ++i) {
      const int32_t index = output.detection_indexes[i];
      TrackResult row = output.rows[i];
      row.class_id = output.class_id;
      result.tracks[index] = row;
    }
  }
  for (size_t i = 0; i < result.tracks.size(); ++i) {
    if (!filled[i]) {
      throw std::runtime_error("internal error: unfilled result row");
    }
  }
  return result;
}

FrameTracks HobotMotTracker::predict(const std::vector<Detection> &detections,
                                     int64_t timestamp_ms, int32_t frame_width,
                                     int32_t frame_height) {
  const PreparedFrame prepared =
      preprocess(detections, timestamp_ms, frame_width, frame_height);
  const RawMotFrame raw = infer(prepared);
  // infer advances the timestamp gate, so predict and manually staged
  // preprocess -> infer -> postprocess behave identically.
  return postprocess(raw, prepared);
}

void HobotMotTracker::reset() {
  per_class_.clear();
  last_timestamp_ms_ = -1;
}

}  // namespace hobot_mot_sample
