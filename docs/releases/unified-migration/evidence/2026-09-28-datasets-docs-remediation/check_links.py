import re, sys
from pathlib import Path

FILES = [
    "datasets/README.md", "datasets/README_cn.md",
    "datasets/coco/README.md", "datasets/coco/README_cn.md",
    "datasets/imagenet/README.md", "datasets/imagenet/README_cn.md",
    "datasets/dotav1/README.md", "datasets/dotav1/README_cn.md",
    "datasets/PascalVOC/README.md", "datasets/PascalVOC/README_cn.md",
    "datasets/yoloe/README.md", "datasets/yoloe/README_cn.md",
    "samples/vision/ultralytics_yolo/evaluator/README.md",
    "samples/vision/ultralytics_yolo/evaluator/README_cn.md",
    "samples/vision/yoloe/evaluator/README.md",
    "samples/vision/yoloe/evaluator/README_cn.md",
]
ROOT = Path(".").resolve()
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
fail = 0
for rel in FILES:
    text = Path(rel).read_text(encoding="utf-8")
    for m in LINK.finditer(text):
        target = m.group(1)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path, _, frag = target.partition("#")
        resolved = (ROOT / rel).parent / path
        if not resolved.exists():
            print(f"MISSING  {rel}: {target}")
            fail += 1
            continue
        if frag:
            dest = resolved.read_text(encoding="utf-8")
            if f'<a id="{frag}">' not in dest:
                print(f"NO-ANCHOR {rel}: {target}")
                fail += 1
print("FAIL" if fail else f"OK: all local links and anchors resolve ({len(FILES)} files)")
sys.exit(1 if fail else 0)
