#!/usr/bin/env python3
"""Static checks for the B7-DOC-R1 status-docs remediation (2026-09-28).

Read-only host checks over the 11 README pairs under samples/vision/yolov5 and
samples/vision/bytetrack:
  1. every relative link (and image link) resolves from its file's directory;
  2. EN/CN pairs keep identical anchor-ID sets, heading counts/level sequences,
     fence counts and image-link lists (readme-contract bilingual pairing);
  3. preserved content is still present verbatim: YOLOv5 historical X5
     performance tables, ByteTrack source figures/captions and the
     threshold/association tuning paragraphs;
  4. the erased-history phrasings this remediation targeted are gone from the
     in-scope pairs (blanket "board not-run" statuses, "did not download /
     never used here" claims that contradict the 2026-09-24 records);
  5. every docs/releases/unified-migration evidence path referenced by the
     edited files exists.
"""
import os
import re
import sys

PAIRS = [
    ("samples/vision/yolov5/README.md", "samples/vision/yolov5/README_cn.md"),
    ("samples/vision/yolov5/model/README.md", "samples/vision/yolov5/model/README_cn.md"),
    ("samples/vision/yolov5/runtime/python/README.md", "samples/vision/yolov5/runtime/python/README_cn.md"),
    ("samples/vision/yolov5/runtime/cpp/README.md", "samples/vision/yolov5/runtime/cpp/README_cn.md"),
    ("samples/vision/yolov5/conversion/README.md", "samples/vision/yolov5/conversion/README_cn.md"),
    ("samples/vision/yolov5/evaluator/README.md", "samples/vision/yolov5/evaluator/README_cn.md"),
    ("samples/vision/bytetrack/README.md", "samples/vision/bytetrack/README_cn.md"),
    ("samples/vision/bytetrack/model/README.md", "samples/vision/bytetrack/model/README_cn.md"),
    ("samples/vision/bytetrack/runtime/python/README.md", "samples/vision/bytetrack/runtime/python/README_cn.md"),
    ("samples/vision/bytetrack/conversion/README.md", "samples/vision/bytetrack/conversion/README_cn.md"),
    ("samples/vision/bytetrack/evaluator/README.md", "samples/vision/bytetrack/evaluator/README_cn.md"),
]

MUST_KEEP = [
    ("samples/vision/yolov5/README.md", "| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |"),
    ("samples/vision/yolov5/README.md", "| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |"),
    ("samples/vision/yolov5/evaluator/README.md", "| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |"),
    ("samples/vision/yolov5/evaluator/README_cn.md", "| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |"),
    ("samples/vision/bytetrack/README.md",
     "![ByteTrack detection example strip embedded by the source README](test_data/readme_img/image1.png)"),
    ("samples/vision/bytetrack/README.md",
     "![Three-row (a)/(b)/(c) association illustration bundled in the source tree, not embedded by the source README](test_data/readme_img/image.png)"),
    ("samples/vision/bytetrack/README.md",
     "(c) tracklets by associating every detection box, where that person's low-score detections (dashed, annotated 0.4 and 0.1) are associated again."),
    ("samples/vision/bytetrack/README.md", "These are tuning directions, not recalibrated thresholds."),
    ("samples/vision/bytetrack/README_cn.md",
     "(c) tracklets by associating every detection box，该行人的低分检测（虚线框，标注 0.4 与 0.1）被重新关联。"),
    ("samples/vision/bytetrack/README_cn.md", "这些是调参方向，不是重新标定的阈值。"),
    ("samples/vision/bytetrack/evaluator/README.md", "![MOT17-01-SDP](../test_data/readme_img/MOT17-01-SDP.gif)"),
    ("samples/vision/bytetrack/evaluator/README.md", "![MOT17-07-SDP](../test_data/readme_img/MOT17-07-SDP.gif)"),
    ("samples/vision/bytetrack/evaluator/README_cn.md", "![MOT17-01-SDP](../test_data/readme_img/MOT17-01-SDP.gif)"),
    ("samples/vision/bytetrack/evaluator/README_cn.md", "![MOT17-07-SDP](../test_data/readme_img/MOT17-07-SDP.gif)"),
    ("samples/vision/bytetrack/evaluator/README.md",
     "- `--track-thresh` (default `0.3`): partitions tracker input each frame — scores above it enter first association; scores in (0.1, track-thresh) enter second association against still-tracked targets at a fixed cost limit of `0.5`; new tracks initiate only from unmatched first-association boxes with score ≥ `track_thresh + 0.1` (`det_thresh`)."),
    ("samples/vision/bytetrack/evaluator/README_cn.md",
     "- `--match-thresh`（默认 `0.8`）：第一次关联分配接受的最大代价（代价 = 1 − IoU，默认模式与检测分数融合；`--mot20` 开关（默认 `false`）关闭融合，此时代价即 1 − IoU）。调大允许更不相似的匹配，调小则只允许更接近的重叠。第二次关联保持固定 `0.5` 上限。"),
]

# Phrasings that erase the 2026-09-24 board history; none may remain.
STALE_PATTERNS = [
    "host contract fixtures; board not-run",
    "主机契约 fixture 通过；板测未运行",
    "host tracker fixtures; board not-run",
    "主机 tracker fixture；板测未运行",
    "this README does not claim either language was run on a board",
    "这里不把任一语言写成已板测",
    "it was not downloaded or run in this migration",
    "本轮没有下载或运行",
    "This migration did not run it:",
    "本轮没有执行：",
    "this migration did not run it",
    "本轮未运行：",
    "neither was used here",
    "本轮均未使用",
    "so the native binary was not compiled or executed on hardware",
    "未在硬件上编译或运行",
    "no board, SDK or `yolov5x_672x672_nv12.hbm` asset was available",
    "无板卡、SDK 及 `yolov5x_672x672_nv12.hbm` 资产",
    "not built or run |",
    "未编译、未运行",
    "This migration did not run it on a board",
    "本轮没有板端运行",
    "S100/S600 and current source/unified board comparison are `not-run`",
    "S100/S600 以及当前源/统一板端比较均为 `not-run`",
    "No conversion or board validation was performed",
    "本轮没有转换或板测",
    "All compiler/export/calibration/board results are `not-run`",
    "编译、导出、校准和板测均为 `not-run`",
    "No model/video download was performed here",
    "本轮没有下载模型或视频",
    "The first two commands are explicit preparation commands and were not run",
    "前两条是显式准备命令，本轮未执行",
    "this migration did not download it",
    "本轮未下载",
    "Current board capture and MOT benchmark status are `not-run`",
    "当前板端 capture 和 MOT benchmark 状态为 `not-run`",
    "No export, compile, board, or video validation was run",
    "本轮未运行导出、编译、板测或视频验证",
    "Conversion and board validation are `not-run`",
    "转换和板端验证为 `not-run`",
    "**Board status: not-run in this migration",
    "**板端状态：本轮 not-run",
    # B7-DOC-R2: exact-command execution overclaims and universal "ever" claims
    "Both preparation commands are exactly what",
    "no S100P inference has ever been run",
    "no S100P download or inference has ever run",
    "no positive run exists",
    "This exact preparation ran",
    "The S100 and S600 selections were prepared this way",
    "downloaded exactly this model and video",
    "正是 2026-09-24",
    "从未运行过",
    "没有正向运行 |",
    "就是这样准备的",
    "下载的正是该模型和视频",
    # B7-DOC-R3: invented support state outside readme-contract §4.1
    "supported-smoke",
]


def strip_fences(text):
    return re.sub(r"```.*?```", "", text, flags=re.S)


def anchors(text):
    return re.findall(r'<a id="([^"]+)"></a>', text)


def heading_seq(text):
    return [len(m) for m in re.findall(r"^(#{1,6})(?=\s)", strip_fences(text), re.M)]


def fence_lines(text):
    return re.findall(r"^```.*$", text, re.M)


def image_links(text):
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)


def main():
    failures = []

    for en, cn in PAIRS:
        te, tc = open(en).read(), open(cn).read()
        if anchors(te) != anchors(tc):
            failures.append(f"anchor mismatch: {en} vs {cn}")
        if heading_seq(te) != heading_seq(tc):
            failures.append(f"heading count/level mismatch: {en} vs {cn}")
        if len(fence_lines(te)) != len(fence_lines(tc)):
            failures.append(f"fence count mismatch: {en} vs {cn}")
        if image_links(te) != image_links(tc):
            failures.append(f"image-link mismatch: {en} vs {cn}")

    link_re = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
    for path in {f for pair in PAIRS for f in pair}:
        text = open(path).read()
        for target in link_re.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
            if not os.path.exists(resolved):
                failures.append(f"broken link: {path}: {target}")

    for path, needle in MUST_KEEP:
        if needle not in open(path).read():
            failures.append(f"preserved content missing: {path}: {needle[:60]}")

    for path in {f for pair in PAIRS for f in pair}:
        text = open(path).read()
        for needle in STALE_PATTERNS:
            if needle in text:
                failures.append(f"stale erase-history phrasing still present: {path}: {needle}")

    print(f"pairs checked: {len(PAIRS)}")
    print(f"stale patterns swept: {len(STALE_PATTERNS)}")
    print("sweep covers: B7-DOC-R1 erase-history phrasings, B7-DOC-R2 exact-command/universal-ever overclaims, B7-DOC-R3 contract-external status token")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
