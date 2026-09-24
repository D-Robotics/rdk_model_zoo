#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Generate instrumented copies of the FIXED YOLOv5 C++ sources.

The fixed sources are taken verbatim from the repository snapshots pinned by
the migration (X5 ``ac115717197920355fc390bb04299b20e6436864``, S
``380e1a2bf42041af54be6f34935e50197cfadff9``), verified by SHA-256 through
``git cat-file`` before anything is generated. Read-only observation hooks
(from ``ycap_observer.hpp``) are inserted at explicitly listed anchors; every
anchor must match exactly once or the tool fails closed. The only permitted
source edits are the anchor insertions plus an explicit, audited whitelist of
path rebindings. The original algorithm (preprocess, inference, decode, NMS)
is untouched, and the unified decoder is never used as the legacy side.

Host usage (also the board usage; see evaluator README):
    python3 samples/vision/yolov5/evaluator/native/instrument.py \
        --target x5 --repo-root <repo> --work-dir /tmp/yolov5-x5-src \
        --model-path <model.bin> --image-path <bus.jpg> --label-path <names>
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

X5_COMMIT = "ac115717197920355fc390bb04299b20e6436864"
S_COMMIT = "380e1a2bf42041af54be6f34935e50197cfadff9"

# SHA-256 of the git blob contents (not the blob ids) for the pinned sources.
PINNED_SHA256 = {
    "x5": {
        "samples/vision/yolov5/runtime/cpp/main.cc":
            "e5c4eb4a445749ad8e009b9ddf9f412f533628fef390aa5d7e8a944357a07257",
    },
    "s100": {
        "samples/vision/yolov5/runtime/cpp/src/main.cpp":
            "f010965a35432c24397ab05b10c0af73eae120db0d0f5f12a6a9535219bb919a",
        "samples/vision/yolov5/runtime/cpp/src/yolov5.cpp":
            "b5ef5427dcba7066d917d733dbdd2c183113985b92e6c110e5e59eaeb9985eb3",
    },
}

# The S build closure is pinned too: the fixed S delivery compiles these exact
# files, so the instrumented build must not silently pick up mutable working-
# tree copies. Every closure file is verified by SHA and copied INTO the work
# dir; the generated CMake references only the copies.
S_CLOSURE_SHA256 = {
    "samples/vision/yolov5/runtime/cpp/inc/yolov5.hpp":
        "c0d70786c426fd1979c00aae51c31d9b188ad0daca93c3aa8c060463abc65dfa",
    "utils/c_utils/inc/file_io.hpp":
        "517381b34ab0fec57754c9094b5e0a7d7fc1a30a5908d0094534c34c82d7a7c8",
    "utils/c_utils/inc/model_types.hpp":
        "b6f57be3651309c0ff1e6a73aac702220ef04f21c9e1ae320b7a8a35864e52c0",
    "utils/c_utils/inc/nn_math.hpp":
        "a8391935e22100e60725ce15ffe374858ef81d34be0e2f7fa27c2369d359dd87",
    "utils/c_utils/inc/postprocess.hpp":
        "b94d3f63883ad62be7bc8a05e118ea9070833c3ac8a6b93f7de33b049045d9be",
    "utils/c_utils/inc/preprocess.hpp":
        "147de18b3d2b4c46ccd02ab34592f7226d7d1ae9aa7c6afcf34f547c9520a38d",
    "utils/c_utils/inc/runtime.hpp":
        "6e9d9e399d41116010da2841f2bc3073bbb5f33a75db87c2a254fa6a9fa365ba",
    "utils/c_utils/inc/visualize.hpp":
        "fb57c7a34a74a1d4df2408293d05bf57a47b031a6db8a8b63455bfe108127df9",
    "utils/c_utils/src/file_io.cpp":
        "36f4ffebcb29e0c3ff560954b553f2e3fe24db938f5119fce9fad864e76cab95",
    "utils/c_utils/src/nn_math.cpp":
        "b1cbc6b5463b2f1c9fbece5348d85d38e718578ba104373c49a098e4267eb2cf",
    "utils/c_utils/src/preprocess.cpp":
        "eb37287b296508b7d824d52a2de909ec4f5e59c52144da55c81ae2e43abba30f",
    "utils/c_utils/src/postprocess.cpp":
        "dd9fbf9eb0c24d1b86c51c3caf576f7e00228710a6de9fe349e6f9bc96dcc6af",
    "utils/c_utils/src/visualize.cpp":
        "514208d0097cc46e415f1e9cc4d7f428f7d3efad93c91eaddcb9988ec86b5696",
}


class InstrumentationError(RuntimeError):
    """Raised for any fail-closed condition (bad SHA, anchor drift, ...)."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_blob(repo_root: pathlib.Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "blob", f"{commit}:{path}"],
        capture_output=True)
    if result.returncode != 0:
        # A shallow clone (the usual board checkout) does not carry the pinned
        # commits. Fail closed with the exact preparation commands instead of
        # silently falling back to mutable working-tree files.
        raise InstrumentationError(
            f"cannot read {path} from {commit} in {repo_root} "
            f"(rc={result.returncode}: {result.stderr.decode(errors='replace').strip()}). "
            "The pinned snapshot must be fetched first, e.g.:\n"
            f"  git -C {repo_root} fetch --depth=1 origin {commit}\n"
            f"  (or a non-shallow clone / GitHub archive of {commit}); "
            "working-tree files are never accepted as a substitute")
    return result.stdout


def pinned_blob(repo_root: pathlib.Path, commit: str, path: str, expected: str) -> bytes:
    """Returns the pinned blob content or fails closed on a SHA mismatch."""
    blob = git_blob(repo_root, commit, path)
    observed = hashlib.sha256(blob).hexdigest()
    if observed != expected:
        raise InstrumentationError(
            f"{path}@{commit}: SHA-256 {observed} != pinned {expected}; refusing")
    return blob


def insert_after_line(text: str, anchor: str, snippet: str, label: str) -> tuple[str, int]:
    """Inserts ``snippet`` after the line containing ``anchor`` exactly once."""
    occurrences = text.count(anchor)
    if occurrences != 1:
        raise InstrumentationError(
            f"anchor '{label}' matched {occurrences} times (expected exactly 1); "
            "the pinned source changed — refusing to instrument")
    lines = text.splitlines(keepends=True)
    out = []
    inserted = 0
    for line in lines:
        out.append(line)
        if anchor in line:
            out.append(snippet)
            inserted += 1
    assert inserted == 1
    return "".join(out), inserted


HOOK_INCLUDE = '#include "ycap_observer.hpp"\n'

X5_ANCHORS = [
    # id, anchor text (must occur exactly once), insertion position
    ("x5-hook-include", '#include "dnn/hb_sys.h"', HOOK_INCLUDE),
    ("x5-begin", 'std::cout << "[OpenCV] Version: " << CV_VERSION << std::endl;',
     "    ycap::begin(\"yolov5_fixed_capture\");\n"
     "    ycap::note_path(\"model\", MODEL_PATH);\n"
     "    ycap::note_path(\"image\", TESR_IMG_PATH);\n"
     "    ycap::note_thresholds(SCORE_THRESHOLD, NMS_THRESHOLD);\n"),
    ("x5-input-nv12", "hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);",
     "    ycap::tensor_meta(\"input0\", \"uint8\", 1, 3, input_H, input_W, 0, 0, 0, 0,\n"
     "                      (long long)(3 * input_H * input_W / 2), \"none\", 0, 0, -1, 1);\n"
     "    ycap::payload(\"input0\", ynv12, (long long)(3 * input_H * input_W / 2));\n"),
    ("x5-output-meta", "hbSysAllocCachedMem(&mem, out_aligned_size);",
     "        {\n"
     "            const char* _dtype = output_properties.tensorType == HB_DNN_TENSOR_TYPE_F32\n"
     "                                    ? \"float32\"\n"
     "                          : output_properties.tensorType == HB_DNN_TENSOR_TYPE_S32 ? \"int32\"\n"
     "                          : output_properties.tensorType == HB_DNN_TENSOR_TYPE_S8 ? \"int8\"\n"
     "                          : output_properties.tensorType == HB_DNN_TENSOR_TYPE_U8 ? \"uint8\"\n"
     "                          : output_properties.tensorType == HB_DNN_TENSOR_TYPE_S16 ? \"int16\"\n"
     "                          : \"unknown\";\n"
     "            const char* _quanti = output_properties.quantiType == NONE ? \"none\"\n"
     "                                : output_properties.quantiType == SCALE ? \"scale\"\n"
     "                                : \"unknown\";\n"
     "            ycap::tensor_meta((std::string(\"output\") + std::to_string(i)).c_str(),\n"
     "                              _dtype,\n"
     "                              output_properties.validShape.dimensionSize[0],\n"
     "                              output_properties.validShape.dimensionSize[1],\n"
     "                              output_properties.validShape.dimensionSize[2],\n"
     "                              output_properties.validShape.dimensionSize[3],\n"
     "                              output_properties.stride[0], output_properties.stride[1],\n"
     "                              output_properties.stride[2], output_properties.stride[3],\n"
     "                              out_aligned_size, _quanti, 0, 0,\n"
     "                              output_properties.quantizeAxis, 0);\n"
     "        }\n"),
    ("x5-raw-small", "hbSysFlushMem(&(output[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);",
     "    ycap::payload((std::string(\"output\") + std::to_string(order[0])).c_str(),\n"
     "                  output[order[0]].sysMem[0].virAddr,\n"
     "                  output[order[0]].properties.alignedByteSize);\n"),
    ("x5-raw-mid", "hbSysFlushMem(&(output[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);",
     "    ycap::payload((std::string(\"output\") + std::to_string(order[1])).c_str(),\n"
     "                  output[order[1]].sysMem[0].virAddr,\n"
     "                  output[order[1]].properties.alignedByteSize);\n"),
    ("x5-raw-large", "hbSysFlushMem(&(output[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);",
     "    ycap::payload((std::string(\"output\") + std::to_string(order[2])).c_str(),\n"
     "                  output[order[2]].sysMem[0].virAddr,\n"
     "                  output[order[2]].properties.alignedByteSize);\n"),
    ("x5-detections", "cv::dnn::NMSBoxes(bboxes[i], scores[i], SCORE_THRESHOLD, NMS_THRESHOLD, indices[i], 1.f, NMS_TOP_K);",
     "        for (std::vector<int>::iterator _it = indices[i].begin();\n"
     "             _it != indices[i].end(); ++_it) {\n"
     "            ycap::detection(bboxes[i][*_it].x, bboxes[i][*_it].y,\n"
     "                            bboxes[i][*_it].x + bboxes[i][*_it].width,\n"
     "                            bboxes[i][*_it].y + bboxes[i][*_it].height,\n"
     "                            scores[i][*_it], i);\n"
     "        }\n"),
    ("x5-detections-original", "float score = scores[cls_id][*it];",
     "            ycap::detection_original(x1, y1, x2, y2, score, cls_id);\n"),
    ("x5-finish", "hbDNNRelease(packed_dnn_handle);", "    ycap::finish(0);\n"),
]

S_MAIN_ANCHORS = [
    ("s-main-hook-include", '#include "yolov5.hpp"', HOOK_INCLUDE),
    ("s-main-begin", "gflags::ParseCommandLineFlags(&argc, &argv, true);",
     "    {\n"
     "        std::string _argv_line;\n"
     "        for (int _i = 0; _i < argc; ++_i) {\n"
     "            if (_i) _argv_line += \" \";\n"
     "            _argv_line += argv[_i];\n"
     "        }\n"
     "        ycap::begin(_argv_line.c_str());\n"
     "    }\n"
     "    ycap::note_path(\"model\", FLAGS_model_path.c_str());\n"
     "    ycap::note_path(\"image\", FLAGS_test_img.c_str());\n"
     "    ycap::note_path(\"labels\", FLAGS_label_file.c_str());\n"),
    ("s-main-finish", 'std::cout << "[Saved] Result saved to: result.jpg" << std::endl;',
     "    ycap::finish(0);\n"),
]

S_YOLOV5_ANCHORS = [
    ("s-yolov5-hook-include", '#include "yolov5.hpp"', HOOK_INCLUDE),
    ("s-output-meta", "prepare_output_tensor(output_tensors);",
     "    for (int _i = 0; _i < (int)output_tensors.size(); ++_i) {\n"
     "        const hbDNNTensorProperties& _p = output_tensors[_i].properties;\n"
     "        // dtype comes from the SDK tensorType enum, never inferred from\n"
     "        // the quantization kind.\n"
     "        const char* _dtype = _p.tensorType == HB_DNN_TENSOR_TYPE_S32 ? \"int32\"\n"
     "                          : _p.tensorType == HB_DNN_TENSOR_TYPE_F32 ? \"float32\"\n"
     "                          : _p.tensorType == HB_DNN_TENSOR_TYPE_S8 ? \"int8\"\n"
     "                          : _p.tensorType == HB_DNN_TENSOR_TYPE_U8 ? \"uint8\"\n"
     "                          : _p.tensorType == HB_DNN_TENSOR_TYPE_S16 ? \"int16\"\n"
     "                          : \"unknown\";\n"
     "        const char* _quanti = _p.quantiType == SCALE ? \"scale\"\n"
     "                          : _p.quantiType == NONE ? \"none\" : \"unknown\";\n"
     "        ycap::tensor_meta((std::string(\"output\") + std::to_string(_i)).c_str(),\n"
     "                          _dtype, _p.validShape.dimensionSize[0],\n"
     "                          _p.validShape.dimensionSize[1], _p.validShape.dimensionSize[2],\n"
     "                          _p.validShape.dimensionSize[3], _p.stride[0], _p.stride[1],\n"
     "                          _p.stride[2], _p.stride[3],\n"
     "                          (long long)_p.alignedByteSize, _quanti,\n"
     "                          _p.scale.scaleLen, _p.scale.zeroPointLen,\n"
     "                          _p.quantizeAxis, 0);\n"
     "        ycap::tensor_scale((std::string(\"output\") + std::to_string(_i)).c_str(),\n"
     "                           _p.scale.scaleData, _p.scale.scaleLen);\n"
     "        if (_p.scale.zeroPointLen > 0 && _p.scale.zeroPointData != nullptr)\n"
     "            ycap::tensor_zero_i32((std::string(\"output\") + std::to_string(_i)).c_str(),\n"
     "                                  _p.scale.zeroPointData, _p.scale.zeroPointLen);\n"
     "    }\n"),
    ("s-input-planes", "bgr_to_nv12_tensor(resized_mat, input_tensors, input_h, input_w);",
     "    for (int _i = 0; _i < (int)input_tensors.size(); ++_i) {\n"
     "        const hbDNNTensorProperties& _p = input_tensors[_i].properties;\n"
     "        const char* _idtype = _p.tensorType == HB_DNN_TENSOR_TYPE_U8 ? \"uint8\"\n"
     "                           : _p.tensorType == HB_DNN_TENSOR_TYPE_S8 ? \"int8\"\n"
     "                           : \"unknown\";\n"
     "        ycap::tensor_meta((std::string(\"input\") + std::to_string(_i)).c_str(),\n"
     "                          _idtype, _p.validShape.dimensionSize[0],\n"
     "                          _p.validShape.dimensionSize[1], _p.validShape.dimensionSize[2],\n"
     "                          _p.validShape.dimensionSize[3], _p.stride[0], _p.stride[1],\n"
     "                          _p.stride[2], _p.stride[3],\n"
     "                          (long long)_p.alignedByteSize, \"none\", 0, 0, -1, 1);\n"
     "        ycap::payload((std::string(\"input\") + std::to_string(_i)).c_str(),\n"
     "                      input_tensors[_i].sysMem.virAddr,\n"
     "                      (long long)(_p.stride[0] * _p.validShape.dimensionSize[0]));\n"
     "    }\n"),
    ("s-raw-outputs", "hbUCPMemFlush(&output_tensors[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);",
     "        ycap::payload((std::string(\"output\") + std::to_string(i)).c_str(),\n"
     "                      output_tensors[i].sysMem.virAddr,\n"
     "                      output_tensors[i].properties.alignedByteSize);\n"),
    ("s-detections", "results = nms_bboxes(decode_all, config.nms_thresh);",
     "    ycap::note_thresholds(config.score_thresh, config.nms_thresh);\n"
     "    for (const Detection& _d : results)\n"
     "        ycap::detection(_d.bbox[0], _d.bbox[1], _d.bbox[2], _d.bbox[3],\n"
     "                        _d.score, _d.class_id);\n"),
    ("s-detections-original", "scale_letterbox_bboxes_back(results, orig_img_w, orig_img_h, input_w, input_h);",
     "    for (const Detection& _d : results)\n"
     "        ycap::detection_original(_d.bbox[0], _d.bbox[1], _d.bbox[2], _d.bbox[3],\n"
     "                                _d.score, _d.class_id);\n"),
]


def apply_anchors(text: str, anchors: list[tuple[str, str, str]]) -> tuple[str, list[dict]]:
    applied = []
    for label, anchor, snippet in anchors:
        text, count = insert_after_line(text, anchor, snippet, label)
        applied.append({"anchor_id": label, "anchor": anchor, "matched": count})
    return text, applied


def apply_path_rebinding(text: str, model_path: str, image_path: str) -> tuple[str, list[dict]]:
    """Whitelisted X5 macro rebinding. Every rewrite is recorded verbatim."""
    rewrites = []
    for macro, value in (("MODEL_PATH", model_path), ("TESR_IMG_PATH", image_path)):
        if not value:
            continue
        old_prefix = f"#define {macro} "
        lines = text.splitlines(keepends=True)
        hits = [index for index, line in enumerate(lines) if line.startswith(old_prefix)]
        if len(hits) != 1:
            raise InstrumentationError(
                f"path rebinding: {macro} define matched {len(hits)} lines (expected 1)")
        index = hits[0]
        old_line = lines[index]
        lines[index] = f'#define {macro} "{value}"\n'
        rewrites.append({"kind": "macro-rebind", "macro": macro,
                         "old": old_line.rstrip("\n"), "new": lines[index].rstrip("\n")})
        text = "".join(lines)
    return text, rewrites


X5_CMAKE = """\
cmake_minimum_required(VERSION 3.16)
project(yolov5_fixed_x5_capture LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)
include_directories(${{OpenCV_INCLUDE_DIRS}} /usr/include/dnn {work_dir})
link_directories(/usr/lib)
add_executable(yolov5_fixed_capture main.cc)
target_link_libraries(yolov5_fixed_capture ${{OpenCV_LIBS}} dnn pthread rt dl)
"""

S_CMAKE = """\
cmake_minimum_required(VERSION 3.16)
project(yolov5_fixed_s_capture LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)
include_directories(${{OpenCV_INCLUDE_DIRS}} /usr/hobot/include
                    /usr/include/hobot/dnn /usr/include/hobot
                    {work_dir} {work_dir}/utils/c_utils/inc {work_dir}/inc)
link_directories(/usr/hobot/lib)
add_executable(yolov5_fixed_capture
    src/main.cpp
    src/yolov5.cpp
    utils/c_utils/src/file_io.cpp
    utils/c_utils/src/nn_math.cpp
    utils/c_utils/src/preprocess.cpp
    utils/c_utils/src/postprocess.cpp
    utils/c_utils/src/visualize.cpp)
target_link_libraries(yolov5_fixed_capture dnn hbucp gflags fmt ${{OpenCV_LIBS}})
set_target_properties(yolov5_fixed_capture PROPERTIES COMPILE_FLAGS "-pthread")
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["x5", "s100"])
    parser.add_argument("--repo-root", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--model-path", default="", help="X5 only: rebind MODEL_PATH")
    parser.add_argument("--image-path", default="", help="X5 only: rebind TESR_IMG_PATH")
    parser.add_argument("--label-path", default="", help="recorded in the audit only (S uses gflags)")
    args = parser.parse_args()

    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        print(f"refusing to overwrite non-empty work dir: {work}", file=sys.stderr)
        return 2
    commit = X5_COMMIT if args.target == "x5" else S_COMMIT
    here = pathlib.Path(__file__).resolve().parent
    audit = {
        "schema": "rdk-model-zoo/yolov5-cpp-instrumentation/v1",
        "utc": utc_now(),
        "target": args.target,
        "pinned_commit": commit,
        "argv": sys.argv,
        "sources": [],
        "path_rebindings": [],
        "build_command": "",
    }

    sources = dict(PINNED_SHA256[args.target])

    instrumented = {}
    x5_main = "samples/vision/yolov5/runtime/cpp/main.cc"
    s_main = "samples/vision/yolov5/runtime/cpp/src/main.cpp"
    s_yolov5 = "samples/vision/yolov5/runtime/cpp/src/yolov5.cpp"
    for rel, expected in sources.items():
        blob = pinned_blob(args.repo_root, commit, rel, expected)
        text = blob.decode("utf-8")
        anchors = (X5_ANCHORS if rel == x5_main
                   else S_MAIN_ANCHORS if rel == s_main
                   else S_YOLOV5_ANCHORS)
        instrumented_text, applied = apply_anchors(text, anchors)
        rewrites = []
        if args.target == "x5" and rel == x5_main:
            instrumented_text, rewrites = apply_path_rebinding(
                instrumented_text, args.model_path, args.image_path)
            audit["path_rebindings"].extend(rewrites)
        # The work dir mirrors the fixed source layout below the sample's
        # runtime/cpp directory, so main.cc and src/*.cpp keep their relative
        # shape for the generated CMake.
        out_rel = rel.replace("samples/vision/yolov5/runtime/cpp/", "")
        (work / out_rel).parent.mkdir(parents=True, exist_ok=True)
        (work / out_rel).write_text(instrumented_text, encoding="utf-8")
        instrumented[rel] = instrumented_text
        audit["sources"].append({
            "source_path": rel,
            "pinned_sha256": expected,
            "blob_sha256": hashlib.sha256(blob).hexdigest(),
            "instrumented_sha256": hashlib.sha256(instrumented_text.encode()).hexdigest(),
            "anchors": applied,
        })

    # Pin and copy the build closure. The S delivery compiles these exact
    # files; the instrumented build must not reach into the mutable working
    # tree for them, so verified blob copies land inside the work dir and the
    # generated CMake references only those.
    audit["closure"] = []
    if args.target != "x5":
        for rel, expected in S_CLOSURE_SHA256.items():
            blob = pinned_blob(args.repo_root, commit, rel, expected)
            out_rel = rel.replace("samples/vision/yolov5/runtime/cpp/", "")
            (work / out_rel).parent.mkdir(parents=True, exist_ok=True)
            (work / out_rel).write_bytes(blob)
            audit["closure"].append({
                "source_path": rel,
                "pinned_sha256": expected,
                "blob_sha256": hashlib.sha256(blob).hexdigest(),
                "work_copy": out_rel,
            })

    (work / "ycap_observer.hpp").write_text(
        (here / "ycap_observer.hpp").read_text(encoding="utf-8"), encoding="utf-8")
    cmake = X5_CMAKE.format(work_dir=work) if args.target == "x5" else \
        S_CMAKE.format(work_dir=work)
    (work / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
    audit["cmake_sha256"] = hashlib.sha256(cmake.encode()).hexdigest()
    audit["hooks_sha256"] = hashlib.sha256(
        (here / "ycap_observer.hpp").read_bytes()).hexdigest()
    if args.label_path:
        audit["label_path"] = args.label_path
    audit["build_command"] = (
        f"cmake -S {work} -B {work}/build && cmake --build {work}/build -j2")
    (work / "instrumentation-audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"work_dir": str(work), "sources": len(audit["sources"]),
                      "anchors": sum(len(s["anchors"]) for s in audit["sources"]),
                      "build": audit["build_command"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InstrumentationError as error:
        print(f"instrumentation failed closed: {error}", file=sys.stderr)
        raise SystemExit(3)
