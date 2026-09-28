import re, sys
from pathlib import Path
PAIRS = [
    ("datasets/coco/README.md", "datasets/coco/README_cn.md"),
    ("datasets/yoloe/README.md", "datasets/yoloe/README_cn.md"),
]
fail = 0
for en, cn in PAIRS:
    be = re.findall(r"```bash\n(.*?)```", Path(en).read_text(), re.S)
    bc = re.findall(r"```bash\n(.*?)```", Path(cn).read_text(), re.S)
    if len(be) != len(bc):
        print(f"BLOCK-COUNT {en}"); fail += 1; continue
    for i, (x, y) in enumerate(zip(be, bc)):
        xe = [l for l in x.splitlines() if l.strip() and not l.strip().startswith("#")]
        yc = [l for l in y.splitlines() if l.strip() and not l.strip().startswith("#")]
        if xe != yc:
            print(f"CMD-DIFF {en} block {i}:\nEN:{xe}\nCN:{yc}"); fail += 1
print("OK: executable command lines byte-identical in all pairs" if not fail else "FAIL")
sys.exit(1 if fail else 0)
