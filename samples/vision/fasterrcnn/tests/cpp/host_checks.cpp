// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Host math harness for the SDK-free C++ stages of the Faster R-CNN sample.
// The Python runner (run_host_checks.py) exports the captured board-stage
// fixtures (tests/fixtures/*.npz) plus Python-computed expectations as flat
// binary files into an input directory; this program replays every CPU stage
// against them and writes its own outputs to a second directory, where the
// runner compares the two bit-for-bit (or within the documented exp()
// tolerances). The board SDK backends are not exercised here: the host build
// compiles the same stage functions the board builds use.
//
// Usage: host_checks <input-dir> <output-dir>

#include "detect.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> read_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  const std::streamoff size = f.tellg();
  f.seekg(0);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  if (size > 0) f.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

template <typename T>
std::vector<T> read_file(const std::string& dir, const std::string& name,
                         std::size_t expected_count) {
  const std::vector<std::uint8_t> raw = read_bytes(dir + "/" + name);
  if (raw.size() != expected_count * sizeof(T))
    throw std::runtime_error(name + " has " + std::to_string(raw.size()) +
                             " bytes, expected " +
                             std::to_string(expected_count * sizeof(T)));
  std::vector<T> values(expected_count);
  if (expected_count > 0)
    std::memcpy(values.data(), raw.data(), raw.size());
  return values;
}

template <typename T>
void write_file(const std::string& dir, const std::string& name,
                const T* values, std::size_t count) {
  std::ofstream f(dir + "/" + name, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + dir + "/" + name);
  if (count > 0) f.write(reinterpret_cast<const char*>(values),
                         static_cast<std::streamsize>(count * sizeof(T)));
}

void write_text(const std::string& dir, const std::string& name,
                const std::string& text) {
  std::ofstream f(dir + "/" + name, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + dir + "/" + name);
  f << text;
}

std::string g17(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return std::string(buf);
}

std::size_t read_size(const std::string& dir, const std::string& name) {
  const std::vector<std::int64_t> v = read_file<std::int64_t>(dir, name, 1);
  if (v[0] < 0) throw std::runtime_error(name + " is negative");
  return static_cast<std::size_t>(v[0]);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: host_checks <input-dir> <output-dir>\n");
    return 2;
  }
  const std::string in = argv[1];
  const std::string out = argv[2];
  try {
    // ---- static anchor table -------------------------------------------
    const std::vector<float>& anchors = fasterrcnn::anchor_table();
    write_file(out, "anchors_out.f32", anchors.data(), anchors.size());

    // ---- RPN selection on the captured PartA outputs --------------------
    const std::vector<float> rpn_obj = read_file<float>(
        in, "rpn_obj.f32", static_cast<std::size_t>(fasterrcnn::kAnchorCount));
    const std::vector<float> rpn_deltas = read_file<float>(
        in, "rpn_deltas.f32",
        static_cast<std::size_t>(fasterrcnn::kAnchorCount) * 4);
    const fasterrcnn::MiddleResult middle =
        fasterrcnn::rpn_select(rpn_obj.data(), rpn_deltas.data(), anchors.data());
    write_file(out, "proposals_out.f32", middle.proposals.data(),
               middle.proposals.size());
    const std::int64_t n_valid = middle.n_valid;
    write_file(out, "n_valid_out.i64", &n_valid, 1);

    // ---- RoIAlign on the captured feat0 / proposal rows ------------------
    const std::vector<float> feat0 = read_file<float>(
        in, "feat0.f32", static_cast<std::size_t>(256 * 10 * 10));
    const std::size_t roi_count = read_size(in, "roi_count.i64");
    const std::vector<float> roi_rois = read_file<float>(
        in, "roi_rois.f32", roi_count * 4);
    std::vector<float> pooled_rows(roi_count * 256 * 7 * 7, 0.0F);
    fasterrcnn::roi_align_feat0(feat0.data(), 256, 10, 10, roi_rois.data(),
                                static_cast<int>(roi_count), pooled_rows.data());
    write_file(out, "pooled_out.f32", pooled_rows.data(), pooled_rows.size());

    // ---- PartB layouts on the full padded batch --------------------------
    const std::vector<float> pooled_full = read_file<float>(
        in, "pooled_full.f32",
        static_cast<std::size_t>(fasterrcnn::kNumProposals) * 256 * 7 * 7);
    std::vector<float> bank(pooled_full.size());
    fasterrcnn::pack_bank(pooled_full.data(), bank.data());
    write_file(out, "bank_out.f32", bank.data(), bank.size());

    // ---- packed-output unpacking roundtrip --------------------------------
    const std::vector<float> cls_packed = read_file<float>(
        in, "cls_packed.f32",
        static_cast<std::size_t>(fasterrcnn::kNumProposals) *
            fasterrcnn::kNumClasses);
    const std::vector<float> reg_packed = read_file<float>(
        in, "reg_packed.f32",
        static_cast<std::size_t>(fasterrcnn::kNumProposals) *
            fasterrcnn::kNumClasses * 4);
    std::vector<float> cls(static_cast<std::size_t>(fasterrcnn::kNumProposals) *
                           fasterrcnn::kNumClasses);
    std::vector<float> reg(static_cast<std::size_t>(fasterrcnn::kNumProposals) *
                           fasterrcnn::kNumClasses * 4);
    fasterrcnn::unpack_bank_outputs(cls_packed.data(), reg_packed.data(),
                                    cls.data(), reg.data());
    write_file(out, "cls_out.f32", cls.data(), cls.size());
    write_file(out, "reg_out.f32", reg.data(), reg.size());

    // ---- decoding tail on the captured PartB outputs ----------------------
    const std::size_t tail_rows = read_size(in, "tail_n_valid.i64");
    const std::vector<float> tail_cls = read_file<float>(
        in, "cls.f32",
        static_cast<std::size_t>(fasterrcnn::kNumProposals) *
            fasterrcnn::kNumClasses);
    const std::vector<float> tail_reg = read_file<float>(
        in, "reg.f32",
        static_cast<std::size_t>(fasterrcnn::kNumProposals) *
            fasterrcnn::kNumClasses * 4);
    const std::vector<float> proposals = read_file<float>(
        in, "proposals.f32",
        static_cast<std::size_t>(fasterrcnn::kNumProposals) * 4);
    const fasterrcnn::DetectionResult dets = fasterrcnn::tail(
        tail_cls.data(), tail_reg.data(), proposals.data(),
        static_cast<int>(tail_rows));
    write_file(out, "det_boxes_out.f32", dets.boxes.data(), dets.boxes.size());
    write_file(out, "det_scores_out.f32", dets.scores.data(), dets.scores.size());
    write_file(out, "det_labels_out.i64", dets.labels.data(), dets.labels.size());

    // ---- decoding tail on an exact-score-tie fixture -----------------------
    // Rows 12 and 68 share one logit vector (their person scores tie
    // bit-identically in every runtime) but decode to boxes IoU ~0.92
    // apart, and row 30 overlaps one of them above the threshold and the
    // other below it.  Stable-order NMS must keep the lower flatten index
    // (row 12) in both runtimes, and row 30 must survive that winner.
    const std::size_t tie_rows = read_size(in, "tie_n_valid.i64");
    const std::vector<float> tie_cls = read_file<float>(
        in, "tie_cls.f32",
        static_cast<std::size_t>(fasterrcnn::kNumProposals) *
            fasterrcnn::kNumClasses);
    const std::vector<float> tie_reg = read_file<float>(
        in, "tie_reg.f32",
        static_cast<std::size_t>(fasterrcnn::kNumProposals) *
            fasterrcnn::kNumClasses * 4);
    const std::vector<float> tie_proposals = read_file<float>(
        in, "tie_proposals.f32",
        static_cast<std::size_t>(fasterrcnn::kNumProposals) * 4);
    const fasterrcnn::DetectionResult tie_dets = fasterrcnn::tail(
        tie_cls.data(), tie_reg.data(), tie_proposals.data(),
        static_cast<int>(tie_rows));
    write_file(out, "tie_det_boxes_out.f32", tie_dets.boxes.data(),
               tie_dets.boxes.size());
    write_file(out, "tie_det_scores_out.f32", tie_dets.scores.data(),
               tie_dets.scores.size());
    write_file(out, "tie_det_labels_out.i64", tie_dets.labels.data(),
               tie_dets.labels.size());

    // ---- letterbox geometry + inverse mapping -----------------------------
    const std::size_t rows = read_size(in, "letterbox_rows.i64");
    const std::size_t cols = read_size(in, "letterbox_cols.i64");
    const fasterrcnn::LetterboxContext ctx =
        fasterrcnn::letterbox_geometry(static_cast<int>(rows),
                                       static_cast<int>(cols));
    {
      std::ostringstream text;
      text << ctx.net_rows << " " << ctx.net_cols << " " << g17(ctx.scale)
           << "\n";
      write_text(out, "letterbox_out.txt", text.str());
    }
    const std::size_t box_count = read_size(in, "canvas_box_count.i64");
    std::vector<float> canvas_boxes =
        read_file<float>(in, "canvas_boxes.f32", box_count * 4);
    for (std::size_t i = 0; i < box_count; ++i)
      ctx.to_original(canvas_boxes.data() + i * 4);
    write_file(out, "original_out.f32", canvas_boxes.data(), canvas_boxes.size());

    // ---- NMS ----------------------------------------------------------------
    const std::size_t nms_count = read_size(in, "nms_count.i64");
    const std::vector<float> nms_boxes =
        read_file<float>(in, "nms_boxes.f32", nms_count * 4);
    const std::vector<float> nms_scores =
        read_file<float>(in, "nms_scores.f32", nms_count);
    const std::vector<float> nms_threshold = read_file<float>(in, "nms_thresh.f32", 1);
    const std::vector<int> keep = fasterrcnn::nms(
        nms_boxes.data(), nms_scores.data(), static_cast<int>(nms_count),
        nms_threshold[0]);
    write_file(out, "keep_out.i32", keep.data(), keep.size());

    // The .7 boundary pair: float32 IoU keeps both boxes, double arithmetic
    // would suppress the second (see the comment on nms_ordered).
    const std::size_t boundary_count = read_size(in, "boundary_count.i64");
    const std::vector<float> boundary_boxes =
        read_file<float>(in, "boundary_boxes.f32", boundary_count * 4);
    const std::vector<float> boundary_scores =
        read_file<float>(in, "boundary_scores.f32", boundary_count);
    const std::vector<float> boundary_threshold =
        read_file<float>(in, "boundary_thresh.f32", 1);
    const std::vector<int> keep_boundary = fasterrcnn::nms(
        boundary_boxes.data(), boundary_scores.data(),
        static_cast<int>(boundary_count), boundary_threshold[0]);
    write_file(out, "keep_boundary_out.i32", keep_boundary.data(),
               keep_boundary.size());

    // ---- streaming sha256 -----------------------------------------------------
    const std::string sha = fasterrcnn::file_sha256(in + "/sha_input.bin");
    write_text(out, "sha256_out.txt", sha + "\n");

    // ---- compiled-in descriptor table -----------------------------------------
    {
      std::ostringstream text;
      for (const fasterrcnn::DescriptorRow& row : fasterrcnn::descriptor_rows()) {
        text << row.target << " " << row.march << " " << row.parta_file << " "
             << row.parta_sha256 << " " << row.partb_file << " "
             << row.partb_sha256 << " "
             << (row.partb_layout == fasterrcnn::PartbLayout::kPooled
                     ? "pooled" : "packed")
             << " " << row.parta_latency_us << " " << row.partb_latency_us
             << "\n";
      }
      write_text(out, "descriptor_out.txt", text.str());
    }

    std::printf("host_checks: %zu anchors, %d proposals, %zu detections\n",
                anchors.size() / 4, middle.n_valid, dets.size());
    return 0;
  } catch (const std::exception& exc) {
    std::fprintf(stderr, "host_checks: %s\n", exc.what());
    return 1;
  }
}
