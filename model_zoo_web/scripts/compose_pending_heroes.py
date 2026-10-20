#!/usr/bin/env python3
"""Retired. Pending heroes are rendered by infer_pending_heroes.py.

This file kept the old quote-and-crop composer. Running it would replace
host-float covers with rank-1 panels, so the entry point now refuses.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
REVISION = "7a97c1780b0cfa131ff0007d662e255e5e41c12e"
OUT_DIR = ROOT / "model_zoo_web" / "release" / "assets" / "pending-heroes"
WIDTH, HEIGHT = 1280, 720
NOTE = (
    "Composed for the pending hero from the pinned file at sample_revision. "
    "Runtime is display-composition, not a board run and not a new inference."
)

PANELS = [
    {
        "yaml": "model_zoo_web/data/vision/convnext/convnext/cls.yaml",
        "file": "convnext.webp",
        "input": "samples/vision/convnext/test_data/cheetah.JPEG",
        "source": "samples/vision/convnext/README.md",
        "quote": "class 293\n`cheetah, chetah, Acinonyx jubatus` with score 0.8048811.",
        "labels": ["cheetah, chetah, Acinonyx jubatus"],
        "scores": ["0.8048811"],
    },
    {
        "yaml": "model_zoo_web/data/vision/vargconvnet/vargconvnet/cls.yaml",
        "file": "vargconvnet.webp",
        "input": "samples/vision/vargconvnet/test_data/box_turtle.JPEG",
        "source": "samples/vision/vargconvnet/README.md",
        "quote": "rank 1 being class 37 (box turtle, box tortoise) with score 0.8582.",
        "labels": ["box turtle, box tortoise"],
        "scores": ["0.8582"],
    },
    {
        "yaml": "model_zoo_web/data/vision/repvgg/repvgg/cls.yaml",
        "file": "repvgg.webp",
        "input": "samples/vision/repvgg/test_data/gooze.JPEG",
        "source": "samples/vision/repvgg/README.md",
        "quote": "rank 1 being class 99 (goose).",
        "labels": ["goose"],
    },
    {
        "yaml": "model_zoo_web/data/vision/repghost/repghost/cls.yaml",
        "file": "repghost.webp",
        "input": "samples/vision/repghost/test_data/ibex.JPEG",
        "source": "samples/vision/repghost/README.md",
        "quote": "rank 1 being class 350 (ibex, Capra ibex).",
        "labels": ["ibex, Capra ibex"],
    },
    {
        "yaml": "model_zoo_web/data/vision/efficientnet/efficientnet/cls.yaml",
        "file": "efficientnet.webp",
        "input": "samples/vision/efficientnet/test_data/redshank.JPEG",
        "source": "samples/vision/efficientnet/README.md",
        "quote": "ranks `redshank` first, followed\nby ruddy turnstone, water ouzel, oystercatcher, and dowitcher.",
        "labels": ["redshank", "ruddy turnstone", "water ouzel", "oystercatcher", "dowitcher"],
    },
    {
        "yaml": "model_zoo_web/data/vision/fastvit/fastvit/cls.yaml",
        "file": "fastvit.webp",
        "input": "samples/vision/fastvit/test_data/bucket.JPEG",
        "source": "samples/vision/fastvit/README.md",
        "quote": "rank 1\nbeing class 463 (bucket).",
        "labels": ["bucket"],
    },
    {
        "yaml": "model_zoo_web/data/vision/efficientformer/efficientformer/cls.yaml",
        "file": "efficientformer.webp",
        "input": "samples/vision/efficientformer/test_data/bittern.JPEG",
        "source": "samples/vision/efficientformer/README.md",
        "quote": "ranks `bittern` first, followed by\npartridge, European gallinule, bustard, and coucal.",
        "labels": ["bittern", "partridge", "European gallinule", "bustard", "coucal"],
    },
    {
        "yaml": "model_zoo_web/data/vision/efficientformerv2/efficientformerv2/cls.yaml",
        "file": "efficientformerv2.webp",
        "input": "samples/vision/efficientformerv2/test_data/goldfish.JPEG",
        "source": "samples/vision/efficientformerv2/README.md",
        "quote": "ranks `goldfish` first, followed\nby tench, axolotl, rock beauty, and coral reef.",
        "labels": ["goldfish", "tench", "axolotl", "rock beauty", "coral reef"],
    },
    {
        "yaml": "model_zoo_web/data/vision/edgenext/edgenext/cls.yaml",
        "file": "edgenext.webp",
        "input": "samples/vision/edgenext/test_data/Zebra.jpg",
        "source": "samples/vision/edgenext/README.md",
        "quote": "rank 1\nbeing class 340 (zebra).",
        "labels": ["zebra"],
    },
    {
        "yaml": "model_zoo_web/data/vision/efficientvit/efficientvit/cls.yaml",
        "file": "efficientvit.webp",
        "input": "samples/vision/efficientvit/test_data/hook.JPEG",
        "source": "samples/vision/efficientvit/README.md",
        "quote": "ranks `hook` first, followed by crane,\nchain, seashore, and dock.",
        "labels": ["hook", "crane", "chain", "seashore", "dock"],
    },
    {
        "yaml": "model_zoo_web/data/vision/fasternet/fasternet/cls.yaml",
        "file": "fasternet.webp",
        "input": "samples/vision/fasternet/test_data/drake.JPEG",
        "source": "samples/vision/fasternet/README.md",
        "quote": "rank 1\nbeing class 97 (drake).",
        "labels": ["drake"],
    },
    {
        "yaml": "model_zoo_web/data/vision/resnext/resnext/cls.yaml",
        "file": "resnext.webp",
        "input": "samples/vision/resnext/test_data/bee_eater.JPEG",
        "source": "samples/vision/resnext/README.md",
        "quote": "rank 1 being class 92 (bee eater).",
        "labels": ["bee eater"],
    },
    {
        "yaml": "model_zoo_web/data/vision/repvit/repvit/cls.yaml",
        "file": "repvit.webp",
        "input": "samples/vision/repvit/test_data/yurt.JPEG",
        "source": "samples/vision/repvit/README.md",
        "quote": "rank 1 being class 915 (yurt).",
        "labels": ["yurt"],
    },
    {
        "yaml": "model_zoo_web/data/vision/mobileone/mobileone/cls.yaml",
        "file": "mobileone.webp",
        "input": "samples/vision/mobileone/test_data/tiger_beetle.JPEG",
        "source": "samples/vision/mobileone/README.md",
        "quote": "rank 1 being class 300 (tiger beetle).",
        "labels": ["tiger beetle"],
    },
    {
        "yaml": "model_zoo_web/data/vision/hgnetv2/hgnetv2/cls.yaml",
        "file": "hgnetv2.webp",
        "input": "samples/vision/hgnetv2/test_data/sandbar.JPEG",
        "source": "samples/vision/hgnetv2/README.md",
        "quote": "rank 1\nbeing class 977 (sandbar, sand bar).",
        "labels": ["sandbar, sand bar"],
    },
    {
        "yaml": "model_zoo_web/data/vision/googlenet/googlenet/cls.yaml",
        "file": "googlenet.webp",
        "input": "samples/vision/googlenet/test_data/indigo_bunting.JPEG",
        "source": "samples/vision/googlenet/README.md",
        "quote": "rank 1 being class 14 (indigo bunting).",
        "labels": ["indigo bunting"],
    },
]

CROPS = [
    {
        "yaml": "model_zoo_web/data/vision/yolov5/yolov5/detect.yaml",
        "file": "yolov5-detect.webp",
        "input": "samples/vision/yolov5/test_data/result.jpg",
        "crop": [0, 450, 1352, 900],
    },
    {
        "yaml": "model_zoo_web/data/vision/yoloe/yoloe11/seg.yaml",
        "file": "yoloe11-seg.webp",
        "input": "samples/vision/yoloe/test_data/source_s11_result_figure.jpg",
        "crop": [0, 33, 640, 393],
    },
    {
        "yaml": "model_zoo_web/data/vision/yoloe/yoloe26/seg.yaml",
        "file": "yoloe26-seg.webp",
        "input": "samples/vision/yoloe/test_data/source_s26_result_figure.jpg",
        "crop": [0, 33, 640, 393],
    },
]


def blob(path: str) -> bytes:
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{REVISION}:{path}"],
        check=True, capture_output=True,
    )
    return shown.stdout


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/macos/Inter-Medium.ttf" if not bold else "/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def cover_fit(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize((max(width, round(image.width * scale)), max(height, round(image.height * scale))), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start: int) -> ImageFont.FreeTypeFont:
    size = start
    while size > 16:
        face = font(size)
        if draw.textlength(text, font=face) <= max_width:
            return face
        size -= 1
    return font(16)


def compose_panel(photo: Image.Image, labels: list[str], scores: list[str]) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    pane = cover_fit(photo, 760, HEIGHT)
    canvas.paste(pane, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((760, 0, 776, HEIGHT), fill=(255, 255, 255))
    panel_x = 812
    max_text = WIDTH - panel_x - 48
    count = len(labels)
    block = 108 if scores else 72
    usable = HEIGHT - 96
    step = usable / count
    for index, label in enumerate(labels):
        y = int(48 + index * step + max(0, (step - block) / 2))
        rank_face = font(26, bold=True)
        draw.text((panel_x, y), str(index + 1), font=rank_face, fill=(40, 96, 232))
        label_face = fit_font(draw, label, max_text - 40, 28 if count <= 2 else 24)
        draw.text((panel_x + 36, y), label, font=label_face, fill=(29, 29, 31))
        if scores:
            value = float(scores[index])
            score_face = font(22)
            draw.text((panel_x + 36, y + 40), f"{value * 100:.2f}%", font=score_face, fill=(109, 117, 128))
            track = (panel_x + 36, y + 74, panel_x + 36 + 360, y + 84)
            draw.rounded_rectangle(track, radius=4, fill=(232, 234, 237))
            filled = max(8, round(360 * value))
            draw.rounded_rectangle((track[0], track[1], track[0] + filled, track[3]), radius=4, fill=(40, 96, 232))
    return canvas


def compose_crop(image: Image.Image, crop: list[int]) -> Image.Image:
    return cover_fit(image.crop(tuple(crop)), WIDTH, HEIGHT)


def hero_block(spec: dict, kind: str) -> str:
    lines = [
        "hero_visual:",
        f"  file: model_zoo_web/release/assets/pending-heroes/{spec['file']}",
        f"  kind: {kind}",
        f"  input: {spec['input']}",
        f"  predictions_source: {spec.get('source', spec['input'])}",
        "  runtime: display-composition",
        f"  note: {json.dumps(NOTE)}",
    ]
    if kind == "classification-panel":
        lines.append(f"  quote: {json.dumps(spec['quote'])}")
        lines.append("  labels:")
        for label in spec["labels"]:
            lines.append(f"  - {json.dumps(label)}")
        if spec.get("scores"):
            lines.append("  scores:")
            for score in spec["scores"]:
                lines.append(f"  - {json.dumps(score)}")
    else:
        crop = ", ".join(str(item) for item in spec["crop"])
        lines.append(f"  crop: [{crop}]")
    return "\n".join(lines) + "\n"


def upsert_yaml(relative: str, block: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    start = text.find("\nhero_visual:\n")
    if start >= 0:
        rest = text[start + 1:]
        end = len(text)
        for marker in ("\nexample_image:\n", "\nsource_input:\n", "\ndocumented_downloads:\n"):
            found = rest.find(marker)
            if found > 0:
                end = min(end, start + 1 + found)
        if end == len(text):
            # Replace through the next top-level key.
            import re
            match = re.search(r"\n[A-Za-z_][^:\n]*:\n", rest[len("hero_visual:\n"):])
            end = start + 1 + len("hero_visual:\n") + match.start() if match else len(text)
        text = text[:start + 1] + block + text[end:]
    else:
        marker = "\nexample_image:\n"
        if marker in text:
            text = text.replace(marker, "\n" + block + marker[1:], 1)
        else:
            if not text.endswith("\n"):
                text += "\n"
            text += block
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in PANELS:
        photo = Image.open(io.BytesIO(blob(spec["input"]))).convert("RGB")
        image = compose_panel(photo, spec["labels"], spec.get("scores") or [])
        image.save(OUT_DIR / spec["file"], "WEBP", quality=86, method=6)
        upsert_yaml(spec["yaml"], hero_block(spec, "classification-panel"))
        print(f"panel {spec['file']}")
    for spec in CROPS:
        source = Image.open(io.BytesIO(blob(spec["input"]))).convert("RGB")
        image = compose_crop(source, spec["crop"])
        image.save(OUT_DIR / spec["file"], "WEBP", quality=86, method=6)
        upsert_yaml(spec["yaml"], hero_block(spec, "result-crop"))
        print(f"crop {spec['file']}")


if __name__ == "__main__":
    raise SystemExit("use model_zoo_web/scripts/infer_pending_heroes.py")
