#!/usr/bin/env python3
"""Static checks for the 2026-09-28 B10 doc follow-up remediation.

Scope: samples/speech/paraformer test_data and conversion README pairs only.
Docs-only: no network access, no model/toolchain/board execution. Run from the
repository root with any Python 3.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

SAMPLE = Path("samples/speech/paraformer")
EDITED = [
    SAMPLE / "test_data/README.md",
    SAMPLE / "test_data/README_cn.md",
    SAMPLE / "conversion/README.md",
    SAMPLE / "conversion/README_cn.md",
]

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


LINK = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")


def link_targets(text):
    for match in LINK.finditer(text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        yield target


def anchor_present(text, anchor):
    if f'id="{anchor}"' in text:
        return True
    slug = re.sub(r"[^a-z0-9 -]", "", anchor.lower()).replace(" ", "-")
    return bool(re.search(rf"^#+ {re.escape(anchor.replace('-', ' '))}\b", text, re.I | re.M)) or bool(
        re.search(rf"^#+ .*{re.escape(slug)}", text, re.I | re.M)
    )


def main():
    for path in EDITED:
        if not path.exists():
            check(f"edited file exists: {path}", False)
            return

    # 1. Stale implementation-status phrases must be gone from every Paraformer README.
    stale = ["still being migrated", "仍待迁移", "待迁移", "not yet implemented", "尚未实现"]
    hits = []
    for path in sorted(SAMPLE.rglob("README*.md")):
        text = path.read_text(encoding="utf-8")
        for phrase in stale:
            if phrase in text:
                hits.append(f"{path}:{phrase}")
    check("no stale implementation-status phrases in sample READMEs", not hits, "; ".join(hits))

    # 2. Every relative link and explicit anchor in the edited files resolves.
    broken = []
    for path in EDITED:
        text = path.read_text(encoding="utf-8")
        for target in link_targets(text):
            if target.startswith("#"):
                if not anchor_present(text, target[1:]):
                    broken.append(f"{path}#{target}")
                continue
            file_part, _, anchor = target.partition("#")
            resolved = (path.parent / file_part).resolve()
            if not resolved.exists():
                broken.append(f"{path} -> {target}")
            elif anchor and not anchor_present(resolved.read_text(encoding="utf-8"), anchor):
                broken.append(f"{path} -> {target} (anchor)")
    check("relative links and anchors resolve in edited files", not broken, "; ".join(broken))

    # 3. Bilingual parity of the two edited pairs: anchors, headings, fences, code blocks.
    for en, cn in [(EDITED[0], EDITED[1]), (EDITED[2], EDITED[3])]:
        en_text = en.read_text(encoding="utf-8")
        cn_text = cn.read_text(encoding="utf-8")
        pair = en.parent.name
        check(
            f"{pair}: anchor id sets match EN/CN",
            re.findall(r'<a id="([^"]+)"></a>', en_text) == re.findall(r'<a id="([^"]+)"></a>', cn_text),
        )
        check(
            f"{pair}: heading count match EN/CN",
            len(re.findall(r"^#{1,6} ", en_text, re.M)) == len(re.findall(r"^#{1,6} ", cn_text, re.M)),
        )
        check(
            f"{pair}: code fence count match EN/CN",
            len(re.findall(r"^```", en_text, re.M)) == len(re.findall(r"^```", cn_text, re.M)),
        )
        en_blocks = Counter(re.findall(r"```[a-z]*\n(.*?)```", en_text, re.S))
        cn_blocks = Counter(re.findall(r"```[a-z]*\n(.*?)```", cn_text, re.S))
        check(f"{pair}: fenced code blocks byte-identical EN/CN", en_blocks == cn_blocks,
              "" if en_blocks == cn_blocks else f"EN-only={list((en_blocks - cn_blocks).elements())} CN-only={list((cn_blocks - en_blocks).elements())}")

    # 4. Facts asserted by the new sentences.
    cpp_en = (SAMPLE / "runtime/cpp/README.md").read_text(encoding="utf-8")
    cpp_cn = (SAMPLE / "runtime/cpp/README_cn.md").read_text(encoding="utf-8")
    check(
        "native guide shows the identical-directory preparation command (EN/CN)",
        "--preprocess-only --output-dir outputs/paraformer_features" in cpp_en
        and "--preprocess-only --output-dir outputs/paraformer_features" in cpp_cn,
    )
    check(
        'native guide exposes the linked #quickstart anchor (EN/CN)',
        '<a id="quickstart"></a>' in cpp_en and '<a id="quickstart"></a>' in cpp_cn,
    )
    root_en = (SAMPLE / "README.md").read_text(encoding="utf-8")
    root_cn = (SAMPLE / "README_cn.md").read_text(encoding="utf-8")
    check(
        "root quickstart example directory outputs/paraformer-prepared exists (EN/CN)",
        "outputs/paraformer-prepared" in root_en and "outputs/paraformer-prepared" in root_cn,
    )
    eval_en = (SAMPLE / "evaluator/README.md").read_text(encoding="utf-8")
    eval_cn = (SAMPLE / "evaluator/README_cn.md").read_text(encoding="utf-8")
    check(
        "evaluator example directory outputs/paraformer-features exists (EN/CN)",
        "outputs/paraformer-features" in eval_en and "outputs/paraformer-features" in eval_cn,
    )
    manifest = json.loads((SAMPLE / "test_data/manifest.json").read_text(encoding="utf-8"))
    check(
        "conversion example's .npy matches the first bundled utterance",
        manifest[0]["utt_id"] == "BAC009S0724W0121"
        and "--feature outputs/paraformer_features/feats/BAC009S0724W0121.npy" in (SAMPLE / "conversion/README.md").read_text(encoding="utf-8"),
    )
    application = (SAMPLE / "runtime/python/application.py").read_text(encoding="utf-8")
    check(
        "preparation really writes feats/<utt_id>.npy under the chosen output dir",
        '(args.output_dir / "feats")' in application and 'f"feats/{key}.npy"' in application,
    )

    # 5. Whitespace hygiene on the scoped diff.
    diff_check = subprocess.run(
        ["git", "diff", "--check", "--", *[str(p) for p in EDITED]],
        capture_output=True, text=True,
    )
    check("git diff --check clean on edited files", diff_check.returncode == 0, diff_check.stdout.strip())

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("All static checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
