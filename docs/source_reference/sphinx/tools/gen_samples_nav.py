#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Sphinx navigation RSTs for samples/* based on AutoAPI outputs.

Input (AutoAPI generated):
  <sphinx>/source/autoapi/samples/<task>/<model>/**/index.rst

Output (navigation we generate):
  <sphinx>/source/python/samples/index.rst
  <sphinx>/source/python/samples/<task>/index.rst
  <sphinx>/source/python/samples/<task>/<model>/index.rst

Design:
- task = first-level directory under samples (e.g., vision, vla)
- model = second-level directory under task (e.g., yolov5)
- model index.rst toctree points to all AutoAPI pages under that model.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import sys


def _underline(title: str, ch: str = "=") -> str:
    return ch * len(title)


def write_index(path: Path, title: str, entries: list[str], maxdepth: int = 3) -> None:
    """Write a minimal index.rst with a toctree."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines += [title, _underline(title, "="), ""]
    lines += [".. toctree::", f"   :maxdepth: {maxdepth}", ""]
    for e in entries:
        lines.append(f"   {e}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: tools/gen_samples_nav.py <docs/api/sphinx>", file=sys.stderr)
        return 2

    sphinx_dir = Path(sys.argv[1]).resolve()
    source_dir = sphinx_dir / "source"

    autoapi_samples = source_dir / "autoapi" / "samples"
    if not autoapi_samples.exists():
        print(f"[gen_samples_nav] autoapi samples not found: {autoapi_samples}", file=sys.stderr)
        return 1

    # Collect all autoapi sample index pages (module/package entry points)
    idx_pages = sorted(autoapi_samples.rglob("index.rst"))
    if not idx_pages:
        print(f"[gen_samples_nav] no index.rst under: {autoapi_samples}", file=sys.stderr)
        return 0

    # groups[task][model] = set of docnames (relative to source, no .rst)
    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for rst in idx_pages:
        # Example file:
        #   source/autoapi/samples/vision/yolov5/runtime/python/main/index.rst
        # docname should be:
        #   autoapi/samples/vision/yolov5/runtime/python/main/index
        docname = rst.relative_to(source_dir).with_suffix("").as_posix()
        parts = docname.split("/")
        # parts: ["autoapi","samples","vision","yolov5",...,"index"]
        if len(parts) < 6:
            continue
        if parts[0] != "autoapi" or parts[1] != "samples":
            continue

        task = parts[2]
        model = parts[3]
        groups[task][model].add(docname)

    # Output navigation root
    out_root = source_dir / "python" / "samples"

    # 1) python/samples/index.rst  -> list tasks
    task_entries = [f"{task}/index" for task in sorted(groups.keys())]
    write_index(out_root / "index.rst", "Samples", task_entries, maxdepth=3)

    # 2) python/samples/<task>/index.rst -> list models
    for task in sorted(groups.keys()):
        model_entries = [f"{model}/index" for model in sorted(groups[task].keys())]
        write_index(out_root / task / "index.rst", task, model_entries, maxdepth=3)

        # 3) python/samples/<task>/<model>/index.rst -> include all autoapi pages under model
        for model in sorted(groups[task].keys()):
            docnames = sorted(groups[task][model])

            # Important: use absolute docnames so Sphinx does not prefix with "python/samples/..."
            # In toctree entries, a leading "/" means "from source root".
            abs_docnames = [f"/{d}" for d in docnames]

            write_index(out_root / task / model / "index.rst", model, abs_docnames, maxdepth=6)

    print(f"[gen_samples_nav] Wrote samples nav under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
