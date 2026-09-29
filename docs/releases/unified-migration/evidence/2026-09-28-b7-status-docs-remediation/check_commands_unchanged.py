#!/usr/bin/env python3
"""Command-preservation check for the B7 status-docs remediation.

For every edited README pair, compares every fenced code block against the
HEAD (`git show HEAD:<path>`) version byte-for-byte. R2 requires CLI/recipe
commands to be preserved exactly; this proves the remediation changed prose
only.
"""
import re
import subprocess
import sys

PAIRS = [
    "samples/vision/yolov5/README.md",
    "samples/vision/yolov5/README_cn.md",
    "samples/vision/yolov5/model/README.md",
    "samples/vision/yolov5/model/README_cn.md",
    "samples/vision/yolov5/runtime/python/README.md",
    "samples/vision/yolov5/runtime/python/README_cn.md",
    "samples/vision/yolov5/runtime/cpp/README.md",
    "samples/vision/yolov5/runtime/cpp/README_cn.md",
    "samples/vision/yolov5/conversion/README.md",
    "samples/vision/yolov5/conversion/README_cn.md",
    "samples/vision/yolov5/evaluator/README.md",
    "samples/vision/yolov5/evaluator/README_cn.md",
    "samples/vision/bytetrack/README.md",
    "samples/vision/bytetrack/README_cn.md",
    "samples/vision/bytetrack/model/README.md",
    "samples/vision/bytetrack/model/README_cn.md",
    "samples/vision/bytetrack/runtime/python/README.md",
    "samples/vision/bytetrack/runtime/python/README_cn.md",
    "samples/vision/bytetrack/conversion/README.md",
    "samples/vision/bytetrack/conversion/README_cn.md",
    "samples/vision/bytetrack/evaluator/README.md",
    "samples/vision/bytetrack/evaluator/README_cn.md",
]


def blocks(text):
    return re.findall(r"```.*?```", text, flags=re.S)


def main():
    failures = []
    total = 0
    for path in PAIRS:
        head = subprocess.run(
            ["git", "show", f"HEAD:{path}"], capture_output=True, text=True, check=True
        ).stdout
        work = open(path).read()
        hb, wb = blocks(head), blocks(work)
        total += len(wb)
        if hb != wb:
            failures.append(f"fenced blocks changed in {path}: {len(hb)} -> {len(wb)}")
    print(f"fenced blocks compared against HEAD: {total} across {len(PAIRS)} files")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL COMMAND BLOCKS BYTE-IDENTICAL TO HEAD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
