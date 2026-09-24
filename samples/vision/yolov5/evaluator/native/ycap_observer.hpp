// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Read-only observation hooks injected into temporary instrumented copies of
// the FIXED YOLOv5 C++ sources (X5 main.cc, S yolov5.cpp/main.cpp) by
// samples/vision/yolov5/evaluator/native/instrument.py. Self-contained and
// SDK-free: every entry point takes plain scalars or raw pointers, and the
// injected glue extracts fields from SDK/OpenCV objects at the call sites.
// Hooks never write model memory and never alter the source's own
// preprocessing, inference, decoding or NMS; they only read buffers the
// source already produced at stage boundaries.
//
// Activation: capture runs only when YOLOV5_CAPTURE_DIR names a directory
// that is EMPTY (or does not yet exist). A non-empty directory is refused —
// the previous run's capture.json must never be mistaken for this run's —
// and a `capture-error.txt` is left in it explaining the refusal. begin()
// also drops a `capture-in-progress.json` marker that is only removed by a
// successful finish(), so an early death or crash leaves a detectable
// unfinished state. Every payload write records its byte count and SHA-256
// at write time; model/image hashes, full argv, cwd and start/end UTC are
// recorded by the observer itself, never recomputed later from mutable
// paths. Process-level evidence (the true subprocess argv — including
// arguments gflags strips from argc/argv — separate stdout/stderr, the
// real exit code and board identity) is owned by the external Python
// runner (run_capture.py) and never impersonated here. Floats
// serialize with roundtrip precision (17 significant digits). Any hook-level
// failure (unwritable file, short write, illegal layout) sets an error that
// finish() persists as a FAILED capture — the comparison refuses it.
#ifndef RDK_MODEL_ZOO_YOLOV5_OBSERVER_HPP_
#define RDK_MODEL_ZOO_YOLOV5_OBSERVER_HPP_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace ycap {

// ---------------------------------------------------------------- SHA-256 ---
// Dependency-free SHA-256 so hashes are computed at capture time on a board
// image with no crypto library.
namespace detail {

struct Sha256 {
  std::uint32_t state[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                            0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::uint64_t total = 0;
  std::uint8_t buffer[64] = {};
  std::size_t buffered = 0;

  static std::uint32_t rotr(std::uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
  }

  void block(const std::uint8_t* p) {
    static const std::uint32_t k[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
        0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (static_cast<std::uint32_t>(p[i * 4]) << 24) |
             (static_cast<std::uint32_t>(p[i * 4 + 1]) << 16) |
             (static_cast<std::uint32_t>(p[i * 4 + 2]) << 8) |
             static_cast<std::uint32_t>(p[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = h + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + maj;
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
  }

  void update(const std::uint8_t* data, std::size_t size) {
    total += size;
    while (size > 0) {
      const std::size_t take = std::min(size, sizeof(buffer) - buffered);
      std::memcpy(buffer + buffered, data, take);
      buffered += take;
      data += take;
      size -= take;
      if (buffered == sizeof(buffer)) {
        block(buffer);
        buffered = 0;
      }
    }
  }

  std::string hex() {
    const std::uint64_t bits = total * 8;
    std::uint8_t pad = 0x80;
    update(&pad, 1);
    pad = 0;
    while (buffered != 56) update(&pad, 1);
    std::uint8_t tail[8];
    for (int i = 0; i < 8; ++i) tail[i] = static_cast<std::uint8_t>(bits >> (56 - i * 8));
    update(tail, 8);  // buffered == 0 here, so no further finalization runs
    std::ostringstream out;
    for (std::uint32_t word : state)
      for (int shift = 28; shift >= 0; shift -= 4) {
        const int nibble = (word >> shift) & 0xF;
        out << static_cast<char>(nibble < 10 ? '0' + nibble : 'a' + nibble - 10);
      }
    return out.str();
  }
};

}  // namespace detail

inline std::string sha256_hex(const void* data, std::size_t size) {
  detail::Sha256 hasher;
  hasher.update(static_cast<const std::uint8_t*>(data), size);
  return hasher.hex();
}

inline std::string sha256_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  detail::Sha256 hasher;
  std::vector<char> buffer(1 << 16);
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = input.gcount();
    if (got > 0)
      hasher.update(reinterpret_cast<const std::uint8_t*>(buffer.data()),
                    static_cast<std::size_t>(got));
  }
  return hasher.hex();
}

// ----------------------------------------------------------------- state ---

struct TensorRecord {
  std::string name;
  std::string dtype;
  long long shape[4] = {0, 0, 0, 0};
  long long stride[4] = {0, 0, 0, 0};
  long long aligned_byte_size = 0;
  std::string quanti;
  long long scale_len = 0;
  long long zero_point_len = 0;
  long long quantize_axis = -1;
  std::vector<double> scale_values;
  std::vector<long long> zero_point_values;
  std::string payload_file;
  long long payload_bytes = 0;
  std::string payload_sha256;
};

struct DetectionRecord {
  double x1 = 0, y1 = 0, x2 = 0, y2 = 0, score = 0;
  long long class_id = -1;
};

enum class State { disabled, active, failed };

struct Capture {
  std::string schema = "rdk-model-zoo/yolov5-cpp-capture/v2";
  std::string utc_start;
  std::string utc_finish;
  std::string cwd;
  std::string argv;
  std::string model_path, image_path, label_path;
  std::string model_sha256, image_sha256;
  double score_threshold = 0;
  double nms_threshold = 0;
  std::string soc_name, board_soc;
  std::vector<TensorRecord> inputs;
  std::vector<TensorRecord> outputs;
  std::vector<DetectionRecord> detections;          // model-space xyxy
  std::vector<DetectionRecord> detections_original;  // final image coords
  std::string error;   // first hook-level failure; fails the capture
  std::string warning; // non-fatal notes (kept for forward compatibility)
  int return_code = -1;
  bool finished = false;
};

inline State& state() {
  static State value = State::disabled;
  return value;
}

inline Capture& capture() {
  static Capture instance;
  return instance;
}

inline const std::string& capture_dir() {
  static const std::string dir = [] {
    const char* from_env = std::getenv("YOLOV5_CAPTURE_DIR");
    return from_env == nullptr ? std::string() : std::string(from_env);
  }();
  return dir;
}

inline void fail(const std::string& message) {
  if (capture().error.empty()) capture().error = message;
  state() = State::failed;
}

inline void warn(const std::string& message) {
  if (capture().warning.empty()) capture().warning = message;
}

inline std::string utc_now() {
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

inline void begin(const char* argv0) {
  const std::string dir = capture_dir();
  if (dir.empty()) return;  // capture not requested: every hook is a no-op
  Capture& cap = capture();
  cap.utc_start = utc_now();
  cap.cwd = std::filesystem::current_path().string();
  cap.argv = argv0 == nullptr ? "" : argv0;
  {
    std::ifstream soc("/sys/class/socinfo/soc_name");
    std::getline(soc, cap.soc_name);
  }
  {
    std::ifstream board("/sys/class/boardinfo/soc_name");
    std::getline(board, cap.board_soc);
  }
  state() = State::active;
  std::error_code code;
  if (std::filesystem::exists(dir, code)) {
    std::error_code list_error;
    std::size_t entries = 0;
    for (auto& entry : std::filesystem::directory_iterator(dir, list_error)) {
      (void)entry;
      ++entries;
      if (list_error) break;
    }
    if (entries != 0) {
      // Refuse: an old capture in this directory must never be mistaken for
      // this run's evidence.
      std::ofstream refusal(std::filesystem::path(dir) / "capture-error.txt",
                            std::ios::trunc);
      if (refusal)
        refusal << "refused: capture directory is not empty (" << entries
                << " entries) at " << cap.utc_start << "\n";
      state() = State::failed;
      cap.error = "capture directory is not empty";
      return;
    }
  }
  std::filesystem::create_directories(dir, code);
  {
    std::ofstream marker(std::filesystem::path(dir) / "capture-in-progress.json",
                         std::ios::trunc);
    if (!marker) {
      fail("cannot write the in-progress marker into " + dir);
      return;
    }
    marker << "{\"utc_start\": \"" << cap.utc_start << "\", \"argv\": "
           << "\"" << cap.argv << "\"}\n";
  }
}

inline void note_path(const char* kind, const char* value) {
  if (state() != State::active) return;
  Capture& cap = capture();
  const std::string text = value == nullptr ? "" : value;
  if (std::string(kind) == "model") {
    cap.model_path = text;
    cap.model_sha256 = sha256_file(text);
    if (cap.model_sha256.empty()) fail("cannot hash model file: " + text);
  } else if (std::string(kind) == "image") {
    cap.image_path = text;
    cap.image_sha256 = sha256_file(text);
    if (cap.image_sha256.empty()) fail("cannot hash image file: " + text);
  } else if (std::string(kind) == "labels") {
    cap.label_path = text;
  }
}

inline void note_thresholds(double score, double nms) {
  if (state() != State::active) return;
  capture().score_threshold = score;
  capture().nms_threshold = nms;
}

inline void tensor_meta(const char* name, const char* dtype, long long d0, long long d1,
                        long long d2, long long d3, long long s0, long long s1,
                        long long s2, long long s3, long long aligned_bytes,
                        const char* quanti, long long scale_len, long long zero_len,
                        long long axis, int inputs) {
  if (state() != State::active) return;
  TensorRecord record;
  record.name = name == nullptr ? "" : name;
  record.dtype = dtype == nullptr ? "" : dtype;
  record.shape[0] = d0; record.shape[1] = d1; record.shape[2] = d2; record.shape[3] = d3;
  record.stride[0] = s0; record.stride[1] = s1; record.stride[2] = s2; record.stride[3] = s3;
  record.aligned_byte_size = aligned_bytes;
  record.quanti = quanti == nullptr ? "" : quanti;
  record.scale_len = scale_len;
  record.zero_point_len = zero_len;
  record.quantize_axis = axis;
  if (record.name.empty() || record.dtype.empty() || d0 < 0 || d1 < 0 || d2 < 0 ||
      d3 < 0 || s0 < 0 || s1 < 0 || s2 < 0 || s3 < 0 || aligned_bytes < 0) {
    fail("illegal tensor layout for " + record.name);
    return;
  }
  (inputs ? capture().inputs : capture().outputs).push_back(std::move(record));
}

inline void tensor_scale(const char* name, const float* values, long long count) {
  if (state() != State::active) return;
  if (name == nullptr || values == nullptr || count < 0) return;
  for (auto& record : capture().outputs)
    if (record.name == name)
      for (long long i = 0; i < count; ++i) record.scale_values.push_back(values[i]);
}

inline void tensor_zero_i8(const char* name, const std::int8_t* values, long long count) {
  if (state() != State::active) return;
  if (name == nullptr || values == nullptr || count < 0) return;
  for (auto& record : capture().outputs)
    if (record.name == name)
      for (long long i = 0; i < count; ++i) record.zero_point_values.push_back(values[i]);
}

inline void tensor_zero_i32(const char* name, const std::int32_t* values, long long count) {
  if (state() != State::active) return;
  if (name == nullptr || values == nullptr || count < 0) return;
  for (auto& record : capture().outputs)
    if (record.name == name)
      for (long long i = 0; i < count; ++i) record.zero_point_values.push_back(values[i]);
}

// Writes one payload file and records its length and SHA-256 at write time.
// A short or failed write fails the capture instead of being silently lost.
inline void payload(const char* name, const void* data, long long bytes) {
  if (state() != State::active) return;
  if (name == nullptr || data == nullptr || bytes <= 0) {
    fail(std::string("illegal payload for ") + (name ? name : "<null>"));
    return;
  }
  const std::string file = std::string("payload-") + name + ".bin";
  const std::filesystem::path path = std::filesystem::path(capture_dir()) / file;
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      fail("cannot open payload file " + path.string());
      return;
    }
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    out.flush();
    if (!out) {
      fail("short write on payload file " + path.string());
      return;
    }
  }
  const std::string digest = sha256_hex(data, static_cast<std::size_t>(bytes));
  TensorRecord* target = nullptr;
  for (auto& record : capture().inputs)
    if (record.name == name) target = &record;
  for (auto& record : capture().outputs)
    if (record.name == name) target = &record;
  if (target == nullptr) {
    fail("payload without matching tensor metadata: " + std::string(name));
    return;
  }
  if (!target->payload_file.empty()) {
    fail("duplicate payload for tensor " + std::string(name));
    return;
  }
  target->payload_file = file;
  target->payload_bytes = bytes;
  target->payload_sha256 = digest;
}

inline void detection(double x1, double y1, double x2, double y2, double score,
                      long long class_id) {
  if (state() != State::active) return;
  capture().detections.push_back({x1, y1, x2, y2, score, class_id});
}

inline void detection_original(double x1, double y1, double x2, double y2, double score,
                               long long class_id) {
  if (state() != State::active) return;
  capture().detections_original.push_back({x1, y1, x2, y2, score, class_id});
}

// ----------------------------------------------------------- serialization ---

inline std::string json_escape(const std::string& value) {
  std::string out;
  for (char ch : value) {
    switch (ch) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
          out += buffer;
        } else {
          out += ch;
        }
    }
  }
  return out;
}

inline std::string quote(const std::string& value) {
  return "\"" + json_escape(value) + "\"";
}

inline void tensor_json(std::ostream& out, const TensorRecord& record, int indent) {
  const std::string pad(static_cast<std::size_t>(indent), ' ');
  out << pad << "{\n";
  out << pad << "  \"name\": " << quote(record.name) << ",\n";
  out << pad << "  \"dtype\": " << quote(record.dtype) << ",\n";
  out << pad << "  \"shape\": [" << record.shape[0] << ", " << record.shape[1] << ", "
      << record.shape[2] << ", " << record.shape[3] << "],\n";
  out << pad << "  \"stride\": [" << record.stride[0] << ", " << record.stride[1] << ", "
      << record.stride[2] << ", " << record.stride[3] << "],\n";
  out << pad << "  \"aligned_byte_size\": " << record.aligned_byte_size << ",\n";
  out << pad << "  \"quanti\": " << quote(record.quanti) << ",\n";
  out << pad << "  \"scale_len\": " << record.scale_len << ",\n";
  out << pad << "  \"zero_point_len\": " << record.zero_point_len << ",\n";
  out << pad << "  \"quantize_axis\": " << record.quantize_axis << ",\n";
  out << pad << "  \"scale_values\": [";
  for (std::size_t i = 0; i < record.scale_values.size(); ++i)
    out << (i ? ", " : "") << std::setprecision(17) << record.scale_values[i];
  out << "],\n";
  out << pad << "  \"zero_point_values\": [";
  for (std::size_t i = 0; i < record.zero_point_values.size(); ++i)
    out << (i ? ", " : "") << record.zero_point_values[i];
  out << "],\n";
  out << pad << "  \"payload_file\": " << quote(record.payload_file) << ",\n";
  out << pad << "  \"payload_bytes\": " << record.payload_bytes << ",\n";
  out << pad << "  \"payload_sha256\": " << quote(record.payload_sha256) << "\n";
  out << pad << "}";
}

inline void detections_json(std::ostream& out, const std::vector<DetectionRecord>& list,
                            int indent) {
  const std::string pad(static_cast<std::size_t>(indent), ' ');
  out << "[\n";
  for (std::size_t i = 0; i < list.size(); ++i) {
    const DetectionRecord& det = list[i];
    out << pad << "  {\"x1\": " << std::setprecision(17) << det.x1
        << ", \"y1\": " << det.y1 << ", \"x2\": " << det.x2 << ", \"y2\": " << det.y2
        << ", \"score\": " << det.score << ", \"class_id\": " << det.class_id << "}";
    out << (i + 1 == list.size() ? "\n" : ",\n");
  }
  out << pad << "]";
}

inline void finish(int return_code) {
  if (state() == State::disabled) return;
  Capture& cap = capture();
  cap.utc_finish = utc_now();
  cap.return_code = return_code;
  cap.finished = true;
  const bool failed = state() == State::failed;
  const std::filesystem::path dir = capture_dir();
  std::ofstream out(dir / "capture.json", std::ios::binary | std::ios::trunc);
  if (!out) return;  // nothing else we can do; the marker stays behind
  out << "{\n";
  out << "  \"schema\": " << quote(cap.schema) << ",\n";
  out << "  \"utc_start\": " << quote(cap.utc_start) << ",\n";
  out << "  \"utc_finish\": " << quote(cap.utc_finish) << ",\n";
  out << "  \"cwd\": " << quote(cap.cwd) << ",\n";
  out << "  \"argv\": " << quote(cap.argv) << ",\n";
  out << "  \"model_path\": " << quote(cap.model_path) << ",\n";
  out << "  \"model_sha256\": " << quote(cap.model_sha256) << ",\n";
  out << "  \"image_path\": " << quote(cap.image_path) << ",\n";
  out << "  \"image_sha256\": " << quote(cap.image_sha256) << ",\n";
  out << "  \"label_path\": " << quote(cap.label_path) << ",\n";
  out << "  \"score_threshold\": " << std::setprecision(17) << cap.score_threshold
      << ",\n";
  out << "  \"nms_threshold\": " << cap.nms_threshold << ",\n";
  out << "  \"soc_name\": " << quote(cap.soc_name) << ",\n";
  out << "  \"board_soc\": " << quote(cap.board_soc) << ",\n";
  out << "  \"return_code\": " << cap.return_code << ",\n";
  out << "  \"failed\": " << (failed ? "true" : "false") << ",\n";
  out << "  \"error\": " << quote(cap.error) << ",\n";
  out << "  \"warning\": " << quote(cap.warning) << ",\n";
  out << "  \"inputs\": [\n";
  for (std::size_t i = 0; i < cap.inputs.size(); ++i) {
    tensor_json(out, cap.inputs[i], 4);
    out << (i + 1 == cap.inputs.size() ? "\n" : ",\n");
  }
  out << "  ],\n";
  out << "  \"outputs\": [\n";
  for (std::size_t i = 0; i < cap.outputs.size(); ++i) {
    tensor_json(out, cap.outputs[i], 4);
    out << (i + 1 == cap.outputs.size() ? "\n" : ",\n");
  }
  out << "  ],\n";
  out << "  \"detections\": ";
  detections_json(out, cap.detections, 2);
  out << ",\n";
  out << "  \"detections_original\": ";
  detections_json(out, cap.detections_original, 2);
  out << "\n}\n";
  out.flush();
  if (!out) return;
  if (!failed) {
    std::error_code code;
    std::filesystem::remove(dir / "capture-in-progress.json", code);
  }
}

}  // namespace ycap

#endif  // RDK_MODEL_ZOO_YOLOV5_OBSERVER_HPP_
