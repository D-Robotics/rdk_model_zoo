#ifndef RDK_MODEL_ZOO_YOLOV5_DETECT_HPP_
#define RDK_MODEL_ZOO_YOLOV5_DETECT_HPP_
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// YOLOv5 detect model header. This header owns the task's whole vocabulary:
// the SDK-free tensor metadata gates, the head decoder, the S dequantizer and
// scheduler mapping, the per-run evidence record the CLI serializes, and the
// model class itself. The board SDK headers and OpenCV are deliberately
// absent, so a host without either compiles the SDK-free core and its
// behaviour checks directly. The model does no file IO: the CLI loads the
// image and passes caller-owned pixels in, and every stage returns owned
// per-call values (nothing per-call lives on the instance).

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace yolov5 {

// ----------------------------------------------------------------- gates ----
// The adapters translate their SDK enum values into these codes; unknown values
// translate to -1 and are always rejected. No SDK numeric value is assumed.
enum DtypeCode : int {
  kDtypeUnknown = -1,
  kDtypeF32 = 0,
  kDtypeS32 = 1,
  kDtypeS8 = 2,
  kDtypeU8 = 3,
  kDtypeS16 = 4,
};

enum QuantiCode : int {
  kQuantiUnknown = -1,
  kQuantiNone = 0,
  kQuantiScale = 1,
  kQuantiShift = 2,
};

enum ImageCode : int {
  kImageUnknown = -1,
  kImageNv12 = 0,
};

// Plain projection of the runtime tensor properties the gates inspect.
struct TensorMeta {
  int dtype = kDtypeUnknown;
  int quanti_type = kQuantiUnknown;
  int num_dimensions = 0;
  long long valid[4] = {0, 0, 0, 0};
  long long aligned[4] = {0, 0, 0, 0};
  long long aligned_byte_size = 0;
  long long storage_bytes = 0;
  long long stride[4] = {0, 0, 0, 0};
  long long scale_len = 0;
  long long zero_point_len = 0;
  long long quantize_axis = -1;
};

struct Gate {
  bool ok = true;
  std::string reason;

  explicit operator bool() const { return ok; }
};

Gate accept();
Gate reject(std::string reason);

// Stable names for the SDK-free dtype/quanti codes, used by the model and the
// dump manifest so both platforms describe tensors identically.
std::string dtype_name(int code);
std::string quanti_name(int code);

// X5: exactly one packed NV12 input of expected_size x expected_size. The
// model copies a compact NV12 payload into the buffer, so a padded/aligned
// layout and an undersized allocation are both rejected instead of being
// silently reinterpreted.
Gate check_x5_nv12_input(int image_type, const TensorMeta& meta,
                         long long expected_size);

// X5: one native F32, NONE-quantized NHWC detection head. Requires the aligned
// layout to equal the valid layout (the reader does a flat float read) and the
// allocation to cover height * width * channels floats. On success writes the
// head stride level (8, 16 or 32).
Gate check_x5_head(const TensorMeta& meta, long long input_size, long long classes,
                   int* stride_level);

// S: one split NV12 plane (Y or UV). Validates the logical shape and requires
// byte strides and allocation that cover every row the preprocessing writer
// touches.
Gate check_s_nv12_plane(const TensorMeta& meta, long long rows, long long cols,
                        long long channels);

// S: one output tensor that the dequantizer may read. Validates the native
// dtype, the quantization descriptor length and the byte strides against the
// addressing actually performed: element (h, w, c) is read at byte offset
// (h*W + w) * stride[2] + c * stride[3]. stride[3] must be a positive
// element-size multiple (channel padding is genuinely supported), stride[2]
// must cover a full pixel (channels elements — smaller values make pixels
// overlap and are rejected), and stride[1] must equal W*stride[2] exactly
// (rows are addressed at a uniform pitch; H-level padding is rejected). A
// scalar scale/zero-point descriptor (length 1) is accepted for the model's
// broadcasting dequantizer, not for the shared c_utils dequantizer. The
// allocation must cover the exact last addressed byte, with overflow-checked
// arithmetic. On success writes the element count.
Gate check_s32_dequant(const TensorMeta& meta, long long* element_count);

// A native binary is configured for exactly one target (S alignment differs
// between S600 and the rest), so it must refuse to run as another target.
Gate check_target_matches_build(const std::string& requested,
                                const std::string& build_target);

// ---------------------------------------------------------------- decode ----

struct HeadShape {
  int height;
  int width;
  int channels;
};

// Per-target decode policy. The two fixed sources really differ here, so the
// difference is an explicit parameter rather than a silently unified rule:
//   * X5 runs cv::dnn::NMSBoxes(..., score_threshold, nms_threshold, top_k=300)
//     per class, which keeps only scores strictly greater than the threshold and
//     caps each class at 300 boxes.
//   * S uses yolov5_decode_all_layers (conf < threshold discards, i.e. equal
//     scores are kept) followed by nms_bboxes with no per-class cap.
struct DecodePolicy {
  float score_threshold = 0.25F;
  float nms_threshold = 0.45F;
  // -1 keeps every surviving box; a positive value caps the boxes kept per
  // class (source X5 NMS_TOP_K is 300).
  int top_k_per_class = -1;
  // true keeps score > threshold (source X5 OpenCV boundary); false keeps
  // score >= threshold (source S boundary).
  bool strict_score_boundary = false;
};

struct Detection {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int class_id;
};

bool validate_head_shapes(const std::vector<HeadShape>& heads,
                          int input_size, int classes);

std::vector<int> order_heads_by_shape(const std::vector<HeadShape>& heads,
                                      int input_size, int classes);

std::vector<Detection> decode_heads(
    const std::vector<std::vector<float>>& raw_heads,
    const std::vector<HeadShape>& heads, int input_size, int classes,
    const DecodePolicy& policy,
    const std::array<float, 18>& anchors);

// ----------------------------------------------------- S numeric helpers ----

// Dequantizes one S output that passed check_s32_dequant. Element (h, w, c) is
// read at byte offset (h*W + w)*stride[2] + c*stride[3], exactly like the
// fixed-source dequantizeTensorS32, but with two deliberate differences that
// the shared helper lacks:
//   * a scalar descriptor (scale_len/zero_point_len == 1) broadcasts its single
//     value to every channel instead of reading scale_data[c] out of bounds;
//   * zero_point_len == 0 means no zero point (0).
// SCALE outputs are read as native int32, NONE outputs as native float32, and
// the result is the compact NHWC float vector of height*width*channels values.
// Throws std::invalid_argument for a quantization kind the gate rejects.
std::vector<float> dequant_s32_nhwc(const unsigned char* base, const TensorMeta& meta,
                                    const float* scale_data, long long scale_len,
                                    const std::int32_t* zero_point_data,
                                    long long zero_point_len);

// Real hb_ucp.h defines the scheduler backend as a bitmask: HB_UCP_BPU_CORE_0
// through _3 are 1ULL<<0..3 and HB_UCP_BPU_CORE_ANY is 1ULL<<7. The CLI's
// --bpu-core is a core *index* (-1 = any), so the conversion has to be
// explicit: assigning the index directly would send 0 as "no backend" and 1 as
// core 0. Returns false (leaving *backend untouched) for indices outside
// -1..3, which the caller must reject instead of silently scheduling.
bool bpu_core_to_backend(long long bpu_core, unsigned long long* backend);

// --------------------------------------------------------------- evidence ----
// Machine-comparable run evidence. Rendering a picture is not evidence, so the
// model records the raw and transformed tensors, metadata, parameters and the
// detected boxes here; the CLI owns serializing this record to disk.

// One tensor payload plus the shape/dtype needed to read it back.
struct DumpTensor {
  std::string name;
  std::string dtype;  // "float32" or "int32"
  std::vector<long long> shape;
  std::vector<unsigned char> bytes;
};

// Metadata-only description of a tensor the model actually exposed. The
// layout fields make a padded run diagnosable from the manifest alone:
// aligned_byte_size and stride[] are what the runtime reported, and aligned[]
// is the alignedShape the X5 SDK reports (the S SDK has no such field, so its
// entries stay unreported). -1 serializes as null. The quant arrays record the
// complete descriptor the runtime exposed so a board run can be replayed
// without the model file: a SCALE descriptor without readable values, or one
// longer than kMaxQuantValues entries, is an error instead of a silent
// truncation.
struct DumpTensorInfo {
  std::string name;
  std::string dtype;
  std::vector<long long> shape;
  std::string quanti;  // "none", "scale", "shift" or "unknown"
  long long scale_len = 0;
  long long aligned_byte_size = -1;
  long long stride[4] = {-1, -1, -1, -1};
  long long aligned[4] = {-1, -1, -1, -1};
  long long quantize_axis = -1;
  std::vector<double> scale_values;        // empty unless quanti == "scale"
  std::vector<long long> zero_point_values;  // empty unless a descriptor exists
};

// Sanity bound for descriptor copying: descriptors above this size are
// rejected outright (the gates accept at most one value per channel of real
// heads), never silently truncated.
constexpr long long kMaxQuantValues = 1LL << 20;

// Fills a DumpTensorInfo from a projected TensorMeta using the shared
// dtype/quanti names, including the reported layout fields. For a SCALE
// tensor the full scale (and zero-point, when declared) descriptor is copied;
// a missing buffer for a declared descriptor, or a descriptor longer than
// kMaxQuantValues, throws std::invalid_argument so the run fails loudly
// instead of recording an empty array as if it were complete.
DumpTensorInfo dump_tensor_info(const std::string& name, const TensorMeta& meta,
                                const float* scale_data = nullptr,
                                const std::int32_t* zero_point_data = nullptr);

struct RunEvidence {
  std::string dir;
  std::string utc;
  std::string target;
  std::string build_target;
  std::string asset_id;
  std::string model_path;
  std::string image_path;
  std::string cwd;
  // Executable that produced this run; its SHA-256 is recorded so the dump
  // binds to the deployed binary, not only to the model and image.
  std::string binary_path;
  std::vector<std::string> argv;
  int return_code = 0;
  std::string error;
  std::vector<std::string> notes;
  std::vector<DumpTensorInfo> inputs;
  std::vector<DumpTensorInfo> outputs;
  std::vector<std::pair<std::string, std::string>> options;
  // The input buffers actually submitted with this inference (deterministic
  // payload bytes; layout lives in the inputs metadata).
  std::vector<DumpTensor> input_tensors;
  std::vector<DumpTensor> raw_tensors;
  std::vector<DumpTensor> transformed_tensors;
  std::vector<Detection> detections;
  // The same detections mapped to final ORIGINAL-image coordinates with the
  // exact arithmetic the renderer applies (independent of the fixed source's
  // own mapping; never derived from it). Compared separately from the
  // model-space detections.
  std::vector<Detection> detections_original;
};

// Maps model-space letterbox detections to final ORIGINAL-image coordinates
// with exactly the arithmetic the renderer applies ((coord - pad) / scale,
// unclamped). SDK-free so the mapping is host-testable.
std::vector<Detection> map_to_original(const std::vector<Detection>& detections,
                                       int image_cols, int image_rows, int model_size);

// ----------------------------------------------------------------- model ----
// The YOLOv5 detection model. One translation unit (detect.cpp) owns the whole
// task: the unconditional SDK-free gates/decoder/dequantizer above and, behind
// compile-time target guards, the X5 HB-DNN and S UCP runtime stages — no
// per-board task files. The constructor performs the build-identity and
// scheduling gates and loads the runtime (RAII; a partially failed
// initialization frees exactly what it allocated). Stage data is per call and
// owned end to end: preprocess returns the NV12 payload it produced, infer
// uploads exactly its prepared argument and copies the raw outputs into the
// returned value, postprocess only decodes (the output cache flush happens
// inside infer), and predict returns the completed result and evidence so a
// later call can never overwrite what an earlier one returned.
class Yolov5 {
 public:
  struct Config {
    // Requested target; must equal the compiled build identity.
    std::string target;
    std::string model_path;
    float score_threshold = 0.25F;
    float nms_threshold = 0.45F;
    int priority = 0;
    int bpu_core = -1;
  };

  // Caller-owned source pixels: 8-bit interleaved BGR, source_rows *
  // source_cols * 3 bytes. The CLI loads the image file and keeps the frame
  // for rendering; the model receives pixels only, never a path.
  struct Input {
    std::vector<unsigned char> bgr;
    int source_cols = 0;
    int source_rows = 0;
  };

  // The owned output of preprocess: the letterboxed NV12 payload for exactly
  // this call plus the source geometry its detections map back to. X5 fills
  // nv12 (compact packed frame); S fills y_plane/uv_plane (split planes,
  // compact rows). A later preprocess cannot touch an earlier Prepared.
  struct Prepared {
    int source_cols = 0;
    int source_rows = 0;
    std::vector<unsigned char> nv12;    // X5: kInput * kInput * 3/2 bytes
    std::vector<unsigned char> y_plane;  // S: kInputSize^2 bytes
    std::vector<unsigned char> uv_plane;  // S: kInputSize^2 / 2 bytes
  };

  // The owned output of infer: the heads read back from this call's forward
  // pass, the per-call geometry, and the evidence stages accumulated so far.
  // The head floats are copies, so a later infer never rewrites them.
  struct RawResult {
    std::vector<HeadShape> shapes;
    std::vector<std::vector<float>> heads;
    int source_cols = 0;
    int source_rows = 0;
    std::vector<DumpTensorInfo> inputs;
    std::vector<DumpTensor> input_tensors;
    std::vector<DumpTensorInfo> outputs;
    std::vector<DumpTensor> raw_tensors;
    std::vector<DumpTensor> transformed_tensors;
  };

  struct Result {
    std::vector<Detection> detections;
  };

  // The owned outcome of one full predict chain: the decoded result and the
  // completed evidence for exactly that call.
  struct Prediction {
    Result result;
    RunEvidence evidence;
  };

  explicit Yolov5(const Config& config);
  ~Yolov5();
  Yolov5(const Yolov5&) = delete;
  Yolov5& operator=(const Yolov5&) = delete;

  // Compiled input size (640 on X5, 672 on S); 0 on a host-only build.
  int input_size() const;

  // Letterboxes the caller's pixels into the model's NV12 contract (X5: one
  // compact packed frame; S: split Y/UV planes) and returns the owned payload.
  // Pure conversion: no SDK call, no instance mutation.
  Prepared preprocess(const Input& input);
  // Uploads exactly its prepared argument, runs the forward pass, flushes the
  // outputs and copies the raw heads (flat F32 on X5, dequantized NHWC on S)
  // into the returned value together with the stage evidence.
  RawResult infer(const Prepared& prepared);
  // Decodes the heads with this build's NMS policy. Never touches the SDK.
  Result postprocess(const RawResult& raw);
  // The visible chain: preprocess -> infer -> postprocess, returning the
  // completed result and evidence of exactly this call.
  Prediction predict(const Input& input);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace yolov5

#endif  // RDK_MODEL_ZOO_YOLOV5_DETECT_HPP_
