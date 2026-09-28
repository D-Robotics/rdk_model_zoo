import re, sys
from pathlib import Path

PAIRS = [
    ("datasets/README.md", "datasets/README_cn.md"),
    ("datasets/coco/README.md", "datasets/coco/README_cn.md"),
    ("datasets/imagenet/README.md", "datasets/imagenet/README_cn.md"),
    ("datasets/dotav1/README.md", "datasets/dotav1/README_cn.md"),
    ("datasets/PascalVOC/README.md", "datasets/PascalVOC/README_cn.md"),
    ("datasets/yoloe/README.md", "datasets/yoloe/README_cn.md"),
]
ANCHOR = re.compile(r'<a id="([^"]+)"></a>')
HEAD = re.compile(r"^#{1,6} (.+)$", re.M)
fail = 0
for en, cn in PAIRS:
    a = set(ANCHOR.findall(Path(en).read_text()))
    b = set(ANCHOR.findall(Path(cn).read_text()))
    if a != b:
        print(f"ANCHOR-MISMATCH {en}: only-EN={a-b} only-CN={b-a}")
        fail += 1
    ha = [m.group(1).strip() for m in HEAD.finditer(Path(en).read_text())]
    hc = [m.group(1).strip() for m in HEAD.finditer(Path(cn).read_text())]
    # headings differ by language; compare counts only
    if len(ha) != len(hc):
        print(f"HEADING-COUNT {en}: {len(ha)} vs {cn}: {len(hc)}")
        fail += 1
    # code fence parity: command blocks must correspond
    ca = re.findall(r"```(?:bash|text|python)?\n(.*?)```", Path(en).read_text(), re.S)
    cc = re.findall(r"```(?:bash|text|python)?\n(.*?)```", Path(cn).read_text(), re.S)
    if len(ca) != len(cc):
        print(f"BLOCK-COUNT {en}: {len(ca)} vs {cn}: {len(cc)}")
        fail += 1
if not fail:
    print("OK: anchors, heading counts and code-block counts match in all 6 pairs")
sys.exit(1 if fail else 0)
