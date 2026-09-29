#!/usr/bin/env python3
"""Static checks for the YOLOE-DOC-R2 source-docs remediation (2026-09-28).

Read-only host checks over the two README pairs in samples/vision/yoloe that
this remediation touched (root and test_data):
  1. byte identity of the two restored source figures against the fixed
     platforms/ snapshots (SHA-256 + cmp);
  2. EN/CN pairs keep identical anchor-ID lists, heading level sequences,
     fence counts and image-link target lists (readme-contract §2–3);
  3. every relative link and image link resolves from its file's directory;
  4. every fenced bash command block is byte-identical to git HEAD
     (command invariance);
  5. preserved content is still present verbatim: support-matrix rows, the
     S float-publication gap sentences, historical benchmark rows, quickstart
     commands and the pre-existing test_data provenance facts;
  6. new content probes: inherited paper/repo links, both figure embeds, the
     runtime/cpp directory-tree line, full figure hashes in the test_data
     pair, and the historical/not-AP boundary sentences.
"""
import hashlib
import os
import re
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../../"))
os.chdir(REPO)

PAIRS = [
    ("samples/vision/yoloe/README.md", "samples/vision/yoloe/README_cn.md"),
    ("samples/vision/yoloe/test_data/README.md", "samples/vision/yoloe/test_data/README_cn.md"),
]

COPIES = [
    ("samples/vision/yoloe/test_data/source_s11_result_figure.jpg",
     "platforms/s/samples/vision/yoloe11_seg/test_data/result.jpg",
     "c53242d5fb3da45dc21736e12356811d41d4a73642ce45b956f70105c3a39cc3"),
    ("samples/vision/yoloe/test_data/source_s26_result_figure.jpg",
     "platforms/s/samples/vision/yoloe26_seg/test_data/result.jpg",
     "95b1c217eeefcb64b635828e8a073a04fc305bceabc0b740b99ea7e25914ee79"),
]

MUST_KEEP = [
    ("samples/vision/yoloe/README.md",
     "| 11s | supported-not-run | not-supported* | not-supported | not-supported | X5; local float S route* | implemented; SDK/board not-run* |"),
    ("samples/vision/yoloe/README.md",
     "| 26n/s/m/l/x | not-supported | not-supported* | not-supported* | not-supported | local float S route* | implemented; SDK/board not-run* |"),
    ("samples/vision/yoloe/README.md",
     "S11 final HBM has mixed outputs and S26 public HBM declares quantized outputs"),
    ("samples/vision/yoloe/README.md",
     "no compatible S float HBM has been compiled/verified"),
    ("samples/vision/yoloe/README.md",
     "| YOLOE-11s PF | X5 | 146.16 ms / 6.84 |"),
    ("samples/vision/yoloe/README.md",
     "| YOLOE-26x PF | S100 | 22.013 ms / 45.31 |"),
    ("samples/vision/yoloe/README.md",
     "All current board checks are `not-run`."),
    ("samples/vision/yoloe/README_cn.md",
     "| 11s | supported-not-run | not-supported* | not-supported | not-supported | X5; local float S route* | implemented; SDK/board not-run* |"),
    ("samples/vision/yoloe/README_cn.md",
     "S11 最终 HBM 为混合精度，S26 公开 HBM 声明量化输出"),
    ("samples/vision/yoloe/README_cn.md",
     "兼容的 S 浮点 HBM 尚未编译验证"),
    ("samples/vision/yoloe/README_cn.md",
     "| YOLOE-11s PF | X5 | 146.16 ms / 6.84 |"),
    ("samples/vision/yoloe/README_cn.md",
     "本轮板测全部 `not-run`"),
    ("samples/vision/yoloe/test_data/README.md",
     "380e1a2bf42041af54be6f34935e50197cfadff9"),
    ("samples/vision/yoloe/test_data/README.md",
     "1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3"),
    ("samples/vision/yoloe/test_data/README.md",
     "`result.jpg` is generated visualization, not a checked-in expected result. No current board test has run."),
    ("samples/vision/yoloe/test_data/README_cn.md",
     "380e1a2bf42041af54be6f34935e50197cfadff9"),
    ("samples/vision/yoloe/test_data/README_cn.md",
     "1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3"),
    ("samples/vision/yoloe/test_data/README_cn.md",
     "`result.jpg` 为运行时产生的可视化，不是签入的期望结果。本轮未进行板测。"),
]

NEW_CONTENT = [
    ("samples/vision/yoloe/README.md",
     "[YOLOE: Real-Time Seeing Anything](https://arxiv.org/pdf/2503.07465v1)"),
    ("samples/vision/yoloe/README.md", "[um-assn/yoloe](https://github.com/um-assn/yoloe)"),
    ("samples/vision/yoloe/README.md", "[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)"),
    ("samples/vision/yoloe/README.md",
     "![Historical S11 source result figure](test_data/source_s11_result_figure.jpg)"),
    ("samples/vision/yoloe/README.md",
     "![Historical S26 source result figure](test_data/source_s26_result_figure.jpg)"),
    ("samples/vision/yoloe/README.md", "├── runtime/cpp/       # reusable three-stage C++ library, SDK adapter, CLI, E11/E26 decoding"),
    ("samples/vision/yoloe/README.md",
     "Both are historical quantized S publication results — not output of this sample's floating route, not expected results for the current code, and not accuracy/AP evidence."),
    ("samples/vision/yoloe/README_cn.md",
     "[YOLOE: Real-Time Seeing Anything](https://arxiv.org/pdf/2503.07465v1)"),
    ("samples/vision/yoloe/README_cn.md", "[um-assn/yoloe](https://github.com/um-assn/yoloe)"),
    ("samples/vision/yoloe/README_cn.md", "[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)"),
    ("samples/vision/yoloe/README_cn.md",
     "![S11 源历史结果插图](test_data/source_s11_result_figure.jpg)"),
    ("samples/vision/yoloe/README_cn.md",
     "![S26 源历史结果插图](test_data/source_s26_result_figure.jpg)"),
    ("samples/vision/yoloe/README_cn.md", "├── runtime/cpp/       # reusable three-stage C++ library, SDK adapter, CLI, E11/E26 decoding"),
    ("samples/vision/yoloe/README_cn.md",
     "两者均为历史量化 S 发布结果——不是本 sample 浮点路径的输出，不是当前代码的预期结果，也不构成精度/AP 证据。"),
    ("samples/vision/yoloe/test_data/README.md",
     "c53242d5fb3da45dc21736e12356811d41d4a73642ce45b956f70105c3a39cc3"),
    ("samples/vision/yoloe/test_data/README.md",
     "95b1c217eeefcb64b635828e8a073a04fc305bceabc0b740b99ea7e25914ee79"),
    ("samples/vision/yoloe/test_data/README_cn.md",
     "c53242d5fb3da45dc21736e12356811d41d4a73642ce45b956f70105c3a39cc3"),
    ("samples/vision/yoloe/test_data/README_cn.md",
     "95b1c217eeefcb64b635828e8a073a04fc305bceabc0b740b99ea7e25914ee79"),
]


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def strip_fences(text):
    return re.sub(r"```.*?```", "", text, flags=re.S)


def anchors(text):
    return re.findall(r'<a id="([^"]+)"></a>', text)


def heading_seq(text):
    return [len(m) for m in re.findall(r"^(#{1,6})(?=\s)", strip_fences(text), re.M)]


def fence_lines(text):
    return re.findall(r"^```.*$", text, re.M)


def image_targets(text):
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)


def bash_blocks(text):
    return re.findall(r"```bash\n(.*?)```", text, re.S)


def main():
    failures = []

    for copy, source, digest in COPIES:
        if sha256(copy) != digest:
            failures.append(f"sha mismatch: {copy}")
        if sha256(copy) != sha256(source):
            failures.append(f"not byte-exact vs source: {copy} vs {source}")
        with open(copy, "rb") as a, open(source, "rb") as b:
            if a.read() != b.read():
                failures.append(f"cmp mismatch: {copy} vs {source}")

    for en, cn in PAIRS:
        te, tc = open(en).read(), open(cn).read()
        if anchors(te) != anchors(tc):
            failures.append(f"anchor mismatch: {en} vs {cn}")
        if heading_seq(te) != heading_seq(tc):
            failures.append(f"heading count/level mismatch: {en} vs {cn}")
        if len(fence_lines(te)) != len(fence_lines(tc)):
            failures.append(f"fence count mismatch: {en} vs {cn}")
        if image_targets(te) != image_targets(tc):
            failures.append(f"image-link target mismatch: {en} vs {cn}")
        if bash_blocks(te) != bash_blocks(tc):
            failures.append(f"bash block mismatch: {en} vs {cn}")

    link_re = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
    for path in {f for pair in PAIRS for f in pair}:
        text = open(path).read()
        for target in link_re.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
            if not os.path.exists(resolved):
                failures.append(f"broken link: {path}: {target}")

    for path in {f for pair in PAIRS for f in pair}:
        text = open(path).read()
        head = subprocess.run(
            ["git", "show", f"HEAD:{path}"], capture_output=True, text=True, check=True
        ).stdout
        if bash_blocks(text) != bash_blocks(head):
            failures.append(f"command block changed vs HEAD: {path}")

    for path, needle in MUST_KEEP + NEW_CONTENT:
        if needle not in open(path).read():
            failures.append(f"probe missing: {path}: {needle[:60]}")

    print(f"pairs checked: {len(PAIRS)}; byte-identity copies: {len(COPIES)}")
    print(f"preserved probes: {len(MUST_KEEP)}; new-content probes: {len(NEW_CONTENT)}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
