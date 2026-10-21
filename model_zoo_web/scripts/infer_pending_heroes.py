#!/usr/bin/env python3
"""Render pending heroes from host float inference.

Classification uses the timm checkpoint named in the sample conversion README.
Detection and instance segmentation use the checkpoint URL written in that
sample's conversion README. The pictures are the sample's own test images, or
another clean image already in the pinned tree. Nothing here is a board run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
REVISION = "7a97c1780b0cfa131ff0007d662e255e5e41c12e"
OUT_DIR = ROOT / "model_zoo_web" / "release" / "assets" / "pending-heroes"
CACHE = Path("/tmp/pending-hero-inputs")
NOTE = "Host float reference, not board output."

# Measured from release/assets/resnet-cls.webp (1672x941).
CANVAS = (1672, 941)
BG = (236, 236, 238)
PHOTO = (39, 40, 864, 864)
PANEL = (941, 40, 692, 862)
RADIUS = 40
ROW_Y = (132, 295, 458, 621, 784)
BAR = (989, 594, 13)  # x, width, height
BAR_DY = 54
LABEL_COLOR = (24, 24, 28)
BAR_TRACK = (226, 230, 235)
BAR_FILL = (46, 100, 234)
FONT_DIR = Path("/usr/share/fonts/truetype/macos")

# Detection and segmentation drawing, measured on the published 1672x941 covers
# release/assets/yolo11-detect.webp and release/assets/yolov8-seg.webp.
# Vertical box strokes are a 4px near-white run. The top edge of a box grows
# from a 6px corner inset to full width the way a radius-8 rounded stroke does.
# The "chair 57%" tag is 113px wide: DejaVu Sans 20px ink is 101px, plus 6px
# of white padding on each side. Tag height follows the 24px line box plus 5px
# of padding. Text is rgb(20, 20, 20) on white, radius 8. Masks are the same
# cyan wash recorded for the published segmentation banners, alpha 92/255,
# with a white contour. Supersampling keeps those widths after downscale.
DET_SUPERSAMPLE = 2
BOX_WIDTH = 4
BOX_RADIUS = 8
TAG_FONT_SIZE = 20
TAG_PAD_X = 6
TAG_PAD_Y = 5
TAG_RADIUS = 8
TAG_TEXT = (20, 20, 20)
WHITE = (255, 255, 255)
MASK_RGB = (49, 203, 220)
MASK_ALPHA = 92
CONTOUR_WIDTH = 2
CONFIDENCE_THRESHOLD = 0.5
# A box that meets the source edge is already cut off by the photograph.
# Drawing it would put the stroke on the cover edge.
SOURCE_EDGE_MARGIN = 8.0
DET_NOTE = (
    "Host float reference, not board output. "
    "Confidence threshold 0.5. Boxes that meet the source photo edge are omitted."
)
TAG_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
# Landscape street from the pinned tree. The cars sit well inside the photo,
# so the 16:9 cover can show every drawn box with margin and no crop.
DET_SCENE = "samples/vision/pp_liteseg/test_data/street.png"

CLASSIFIERS = [
    ("convnext", "model_zoo_web/data/vision/convnext/convnext/cls.yaml", "convnext_atto",
     "samples/vision/convnext/test_data/cheetah.JPEG", "samples/vision/convnext/conversion/README.md"),
    ("edgenext", "model_zoo_web/data/vision/edgenext/edgenext/cls.yaml", "edgenext_xx_small",
     "samples/vision/edgenext/test_data/Zebra.jpg", "samples/vision/edgenext/conversion/README.md"),
    ("efficientformer", "model_zoo_web/data/vision/efficientformer/efficientformer/cls.yaml", "efficientformer_l1",
     "samples/vision/efficientformer/test_data/bittern.JPEG", "samples/vision/efficientformer/conversion/README.md"),
    ("efficientformerv2", "model_zoo_web/data/vision/efficientformerv2/efficientformerv2/cls.yaml", "efficientformerv2_s0",
     "samples/vision/efficientformerv2/test_data/goldfish.JPEG", "samples/vision/efficientformerv2/conversion/README.md"),
    ("efficientnet", "model_zoo_web/data/vision/efficientnet/efficientnet/cls.yaml", "efficientnet_b2",
     "samples/vision/efficientnet/test_data/redshank.JPEG", "samples/vision/efficientnet/conversion/README.md"),
    ("efficientvit", "model_zoo_web/data/vision/efficientvit/efficientvit/cls.yaml", "efficientvit_m5",
     "samples/vision/efficientvit/test_data/hook.JPEG", "samples/vision/efficientvit/conversion/README.md"),
    ("fastvit", "model_zoo_web/data/vision/fastvit/fastvit/cls.yaml", "fastvit_t8",
     "samples/vision/fastvit/test_data/bucket.JPEG", "samples/vision/fastvit/conversion/README.md"),
    ("hgnetv2", "model_zoo_web/data/vision/hgnetv2/hgnetv2/cls.yaml", "hgnetv2_b0.ssld_stage2_ft_in1k",
     "samples/vision/hgnetv2/test_data/sandbar.JPEG", "samples/vision/hgnetv2/conversion/README.md"),
    ("repghost", "model_zoo_web/data/vision/repghost/repghost/cls.yaml", "repghostnet_100",
     "samples/vision/repghost/test_data/ibex.JPEG", "samples/vision/repghost/conversion/README.md"),
    ("repvit", "model_zoo_web/data/vision/repvit/repvit/cls.yaml", "repvit_m0_9",
     "samples/vision/repvit/test_data/yurt.JPEG", "samples/vision/repvit/conversion/README.md"),
    ("resnext", "model_zoo_web/data/vision/resnext/resnext/cls.yaml", "resnext50_32x4d",
     "samples/vision/resnext/test_data/bee_eater.JPEG", "samples/vision/resnext/conversion/README.md"),
]

# Samples with no documented runnable checkpoint. Keep a clean photo.
PLAIN = [
    ("model_zoo_web/data/vision/repvgg/repvgg/cls.yaml", "samples/vision/repvgg/test_data/gooze.JPEG"),
    ("model_zoo_web/data/vision/mobileone/mobileone/cls.yaml", "samples/vision/mobileone/test_data/tiger_beetle.JPEG"),
    ("model_zoo_web/data/vision/fasternet/fasternet/cls.yaml", "samples/vision/fasternet/test_data/drake.JPEG"),
    ("model_zoo_web/data/vision/googlenet/googlenet/cls.yaml", "samples/vision/googlenet/test_data/indigo_bunting.JPEG"),
    ("model_zoo_web/data/vision/vargconvnet/vargconvnet/cls.yaml", "samples/vision/vargconvnet/test_data/box_turtle.JPEG"),
    ("model_zoo_web/data/vision/yoloe/yoloe26/seg.yaml", "samples/vision/yoloe/test_data/office_desk.jpg"),
]


def blob(path: str) -> bytes:
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{REVISION}:{path}"],
        check=True, capture_output=True,
    )
    return shown.stdout


def materialize(path: str) -> Path:
    dest = CACHE / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        dest.write_bytes(blob(path))
    return dest


def font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    name = "Inter-SemiBold.ttf" if bold else "Inter-Regular.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def cover_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def first_name(description: str) -> str:
    return description.split(",")[0].strip().replace("_", " ")


def render_classification(photo: Image.Image, topk: list[dict]) -> Image.Image:
    canvas = Image.new("RGB", CANVAS, BG)
    fitted = cover_crop(photo.convert("RGB"), PHOTO[2], PHOTO[3])
    mask = Image.new("L", (PHOTO[2], PHOTO[3]), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PHOTO[2] - 1, PHOTO[3] - 1), radius=RADIUS, fill=255)
    canvas.paste(fitted, PHOTO[:2], mask)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (PANEL[0], PANEL[1], PANEL[0] + PANEL[2] - 1, PANEL[1] + PANEL[3] - 1),
        radius=RADIUS, fill=(255, 255, 255),
    )
    percent_right = BAR[0] + BAR[1]
    label_left = 990
    max_label = percent_right - 120 - label_left
    for index, row in enumerate(topk):
        bold = index == 0
        label = row["label"]
        chosen = font(32 if bold else 30, bold)
        for size in range(32 if bold else 30, 16, -1):
            chosen = font(size, bold)
            if draw.textlength(label, font=chosen) <= max_label:
                break
        y = ROW_Y[index]
        draw.text((label_left, y), label, font=chosen, fill=LABEL_COLOR)
        percent = f"{row['score'] * 100:.1f}%"
        percent_font = font(30 if bold else 28, bold)
        width = draw.textlength(percent, font=percent_font)
        draw.text((percent_right - width, y), percent, font=percent_font, fill=LABEL_COLOR)
        bar_y = y + BAR_DY
        draw.rounded_rectangle(
            (BAR[0], bar_y, BAR[0] + BAR[1], bar_y + BAR[2]),
            radius=BAR[2] // 2, fill=BAR_TRACK,
        )
        fill = max(0, round(row["score"] * BAR[1]))
        if fill > 0:
            draw.rounded_rectangle(
                (BAR[0], bar_y, BAR[0] + max(fill, BAR[2]), bar_y + BAR[2]),
                radius=BAR[2] // 2, fill=BAR_FILL,
            )
    return canvas


def tag_font(scale: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(TAG_FONT_PATH), TAG_FONT_SIZE * scale)


def shift_axis(start: float, end: float, limit: float, content_start: float, content_end: float) -> tuple[float, float]:
    """Slide a window along one axis so it uses the photo before any blurred overflow."""
    length = end - start
    if length <= limit:
        desired = min(max(start, 0.0), limit - length)
    else:
        desired = (limit - length) / 2
    # Keep the content inside the window. Never trade that for a smaller blur.
    desired = min(desired, content_start)
    desired = max(desired, content_end - length)
    return desired, desired + length


def detection_window(boxes: list[list[float]], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """16:9 window that contains every box, with room above each box for its tag.

    When the photo is wide enough, the window stays inside it. Overflow is only
    the part that cannot move without cutting a box, and the caller blur-fills that.
    """
    width, height = image_size
    target = CANVAS[0] / CANVAS[1]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    span = max(x2 - x1, y2 - y1, 1.0)
    margin = max(28.0, span * 0.06)
    top_extra = margin
    left = top = right = bottom = 0.0
    for _ in range(4):
        left, top = x1 - margin, y1 - margin - top_extra
        right, bottom = x2 + margin, y2 + margin
        box_w, box_h = max(right - left, 1.0), max(bottom - top, 1.0)
        if box_w / box_h < target:
            grow = (box_h * target - box_w) / 2
            left -= grow
            right += grow
        else:
            grow = (box_w / target - box_h) / 2
            top -= grow
            bottom += grow
        scale = (CANVAS[0] * DET_SUPERSAMPLE) / (right - left)
        wanted_top = (TAG_FONT_SIZE + TAG_PAD_Y * 2 + 16) * DET_SUPERSAMPLE / scale
        wanted_margin = 32 * DET_SUPERSAMPLE / scale
        if top_extra >= wanted_top - 0.5 and margin >= wanted_margin - 0.5:
            break
        top_extra = max(top_extra, wanted_top)
        margin = max(margin, wanted_margin)
    box_w, box_h = right - left, bottom - top
    if box_w / box_h < target:
        grow = (box_h * target - box_w) / 2
        left -= grow
        right += grow
    elif box_w / box_h > target:
        grow = (box_w / target - box_h) / 2
        top -= grow
        bottom += grow
    content = (x1 - margin, y1 - margin - top_extra, x2 + margin, y2 + margin)
    left, right = shift_axis(left, right, width, content[0], content[2])
    top, bottom = shift_axis(top, bottom, height, content[1], content[3])
    return left, top, right, bottom


def compose_window(image: Image.Image, window: tuple[float, float, float, float]) -> tuple[Image.Image, callable]:
    """Scale the photo into a 16:9 canvas. Overflow is a blurred extension of the photo."""
    left, top, right, bottom = window
    win_w = right - left
    out_w = CANVAS[0] * DET_SUPERSAMPLE
    out_h = CANVAS[1] * DET_SUPERSAMPLE
    src = image.convert("RGB")
    small = src.resize((max(8, src.width // 12), max(8, src.height // 12)), Image.Resampling.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(radius=8))
    cover_scale = max(out_w / small.width, out_h / small.height)
    cover = small.resize(
        (max(1, round(small.width * cover_scale)), max(1, round(small.height * cover_scale))),
        Image.Resampling.BILINEAR,
    )
    cx = max(0, (cover.width - out_w) // 2)
    cy = max(0, (cover.height - out_h) // 2)
    canvas = cover.crop((cx, cy, cx + out_w, cy + out_h))
    scale = out_w / win_w
    resized = src.resize(
        (max(1, round(src.width * scale)), max(1, round(src.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas.paste(resized, (int(round(-left * scale)), int(round(-top * scale))))

    def mapper(x: float, y: float) -> tuple[float, float]:
        return (x - left) * scale, (y - top) * scale

    return canvas, mapper


def render_detections(image: Image.Image, instances: list[dict]) -> Image.Image:
    """One drawer for boxes and masks. Coordinates are source pixels from the model."""
    if not instances:
        raise RuntimeError("no instances to draw")
    bounds = [list(instance["box"]) for instance in instances]
    for instance in instances:
        for polygon in instance.get("polygons") or []:
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            if xs:
                bounds.append([min(xs), min(ys), max(xs), max(ys)])
    canvas, mapper = compose_window(image, detection_window(bounds, image.size))
    scale = DET_SUPERSAMPLE
    mask = Image.new("L", canvas.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    contours = []
    for instance in instances:
        for polygon in instance.get("polygons") or []:
            points = [mapper(point[0], point[1]) for point in polygon]
            if len(points) >= 3:
                mask_draw.polygon(points, fill=255)
                contours.append(points)
    if any(instance.get("polygons") for instance in instances):
        faded = mask.point(lambda value: MASK_ALPHA if value else 0)
        canvas = Image.composite(Image.new("RGB", canvas.size, MASK_RGB), canvas, faded)
    draw = ImageDraw.Draw(canvas)
    contour_width = CONTOUR_WIDTH * scale
    for points in contours:
        draw.line(points + [points[0]], fill=WHITE, width=contour_width, joint="curve")
    font = tag_font(scale)
    ascent, descent = font.getmetrics()
    pad_x = TAG_PAD_X * scale
    pad_y = TAG_PAD_Y * scale
    placed = []
    for instance in sorted(instances, key=lambda item: item["score"]):
        x1, y1 = mapper(instance["box"][0], instance["box"][1])
        x2, y2 = mapper(instance["box"][2], instance["box"][3])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        inset = 2 * scale
        if x1 < inset or y1 < inset or x2 > canvas.width - inset or y2 > canvas.height - inset:
            raise RuntimeError(f"box leaves the frame: {instance['label']} {(x1, y1, x2, y2)}")
        radius = min(BOX_RADIUS * scale, (x2 - x1) / 2, (y2 - y1) / 2)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, outline=WHITE, width=BOX_WIDTH * scale)
        text = f"{instance['label']} {int(instance['score'] * 100 + 0.5)}%"
        ink = font.getbbox(text)
        tag_w = (ink[2] - ink[0]) + 2 * pad_x
        tag_h = ascent + descent + 2 * pad_y
        tag_x = (x1 + x2) / 2 - tag_w / 2
        tag_y = y1 - tag_h
        tag_x = min(max(tag_x, inset), canvas.width - inset - tag_w)
        tag_y = min(max(tag_y, inset), canvas.height - inset - tag_h)
        tag_radius = min(TAG_RADIUS * scale, tag_w / 2, tag_h / 2)
        draw.rounded_rectangle(
            (tag_x, tag_y, tag_x + tag_w, tag_y + tag_h),
            radius=tag_radius, fill=WHITE,
        )
        draw.text((tag_x + pad_x - ink[0], tag_y + pad_y), text, font=font, fill=TAG_TEXT)
        placed.append((tag_x, tag_y, tag_x + tag_w, tag_y + tag_h))
    for tag in placed:
        if tag[0] < 0 or tag[1] < 0 or tag[2] > canvas.width or tag[3] > canvas.height:
            raise RuntimeError(f"label leaves the frame: {tag}")
    return canvas.resize(CANVAS, Image.Resampling.LANCZOS)


def record_instances(instances: list[dict]) -> list[dict]:
    ordered = sorted(instances, key=lambda item: item["score"], reverse=True)
    return [
        {
            "label": instance["label"],
            "score": round(float(instance["score"]), 6),
            "box": [round(float(value), 2) for value in instance["box"]],
        }
        for instance in ordered
    ]


def upsert_hero(yaml_path: str, block: str) -> None:
    path = ROOT / yaml_path
    text = path.read_text(encoding="utf-8")
    start = text.find("\nhero_visual:\n")
    if start >= 0:
        rest = text[start + 1:]
        import re
        match = re.search(r"\n[A-Za-z_][^:\n]*:\n", rest[len("hero_visual:\n"):])
        end = start + 1 + len("hero_visual:\n") + match.start() if match else len(text)
        text = text[:start] + "\n" + block + text[end:]
    else:
        marker = "\nexample_image:\n"
        if marker not in text:
            raise SystemExit(f"{yaml_path} has no example_image block")
        text = text.replace(marker, "\n" + block + marker, 1)
    path.write_text(text, encoding="utf-8")


def strip_hero(yaml_path: str, plain_image: str) -> None:
    path = ROOT / yaml_path
    text = path.read_text(encoding="utf-8")
    start = text.find("\nhero_visual:\n")
    if start >= 0:
        import re
        rest = text[start + 1:]
        match = re.search(r"\n[A-Za-z_][^:\n]*:\n", rest[len("hero_visual:\n"):])
        end = start + 1 + len("hero_visual:\n") + match.start() if match else len(text)
        text = text[:start] + text[end - 1:]
    old = "example_image:\n"
    at = text.find(old)
    if at < 0:
        raise SystemExit(f"{yaml_path} has no example_image")
    tail = text[at:]
    lines = tail.splitlines()
    rebuilt = [lines[0]]
    for line in lines[1:]:
        if line.startswith("  path:") or line.startswith("  source:"):
            key = line.split(":", 1)[0]
            rebuilt.append(f"{key}: {plain_image}")
        else:
            rebuilt.append(line)
            if not line.startswith("  "):
                break
    # The loop above may stop early and drop the remainder. Rebuild only the two fields.
    text = text[:at] + f"example_image:\n  path: {plain_image}\n  source: {plain_image}\n" + text[at:].split("\n", 3)[-1] if False else text
    # Replace just the path and source lines inside the example_image mapping.
    import re
    text2, count = re.subn(
        r"(example_image:\n  path:)[^\n]*(\n  source:)[^\n]*",
        rf"\1 {plain_image}\2 {plain_image}",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"{yaml_path} example_image was not updated")
    path.write_text(text2, encoding="utf-8")


def write_record(stem: str, yaml_path: str, kind: str, image: str, documented: str,
                 model_file: str, model_url: str, framework: str, payload: dict,
                 note: str = NOTE) -> None:
    record = {
        "runtime": "host-float-reference",
        "commit": REVISION,
        "model_file": model_file,
        "model_url": model_url,
        "input": image,
        "framework": framework,
        "documented_in": documented,
        "note": note,
    }
    record.update(payload)
    canonical = json.loads(json.dumps(record))
    json_name = f"{stem}.json"
    (OUT_DIR / json_name).write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    hero = {
        "file": f"model_zoo_web/release/assets/pending-heroes/{stem}.webp",
        "kind": kind,
        "input": image,
        "predictions_source": f"model_zoo_web/release/assets/pending-heroes/{json_name}",
        "runtime": "host-float-reference",
        "note": note,
        "model_file": model_file,
        "model_url": model_url,
        "framework": framework,
        "commit": REVISION,
        "documented_in": documented,
    }
    if "confidence_threshold" in canonical:
        hero["confidence_threshold"] = canonical["confidence_threshold"]
    if kind == "classification-panel":
        hero["topk"] = canonical["topk"]
    else:
        hero["detections"] = canonical["detections"]
    dumped = yaml.safe_dump(hero, sort_keys=False, allow_unicode=True).rstrip() + "\n"
    loaded = yaml.safe_load(dumped)
    if kind == "classification-panel" and loaded["topk"] != canonical["topk"]:
        raise SystemExit(f"{stem} topk did not round-trip")
    if kind != "classification-panel" and loaded["detections"] != canonical["detections"]:
        raise SystemExit(f"{stem} detections did not round-trip")
    if "confidence_threshold" in canonical and loaded.get("confidence_threshold") != canonical["confidence_threshold"]:
        raise SystemExit(f"{stem} confidence threshold did not round-trip")
    upsert_hero(yaml_path, "hero_visual:\n" + "\n".join(f"  {line}" if line else "" for line in dumped.splitlines()) + "\n")


def run_classifiers() -> list[str]:
    import timm
    import torch
    from timm.data import ImageNetInfo, create_transform, resolve_data_config
    descriptions = [ImageNetInfo().index_to_description(i) for i in range(1000)]
    status = []
    for stem, yaml_path, model_name, image, documented in CLASSIFIERS:
        print(f"cls {stem} {model_name}", flush=True)
        try:
            model = timm.create_model(model_name, pretrained=True)
            model.eval()
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config, is_training=False)
            photo = Image.open(materialize(image)).convert("RGB")
            with torch.inference_mode():
                logits = model(transform(photo).unsqueeze(0))
                probabilities = torch.softmax(logits, dim=1)[0]
            values, indices = torch.topk(probabilities, 5)
            topk = []
            for score, index in zip(values.tolist(), indices.tolist()):
                topk.append({"label": first_name(descriptions[index]), "score": round(float(score), 6)})
            url = model.pretrained_cfg.get("url") or ""
            if not str(url).startswith("https://"):
                hub = model.pretrained_cfg.get("hf_hub_id")
                url = f"https://huggingface.co/{hub}" if hub else ""
            if not str(url).startswith("https://"):
                raise RuntimeError(f"no checkpoint URL for {model_name}")
            rendered = render_classification(photo, topk)
            rendered.save(OUT_DIR / f"{stem}.webp", "WEBP", quality=90, method=6)
            write_record(
                stem, yaml_path, "classification-panel", image, documented, model_name, url,
                f"pytorch-cpu/{torch.__version__} timm/{timm.__version__}",
                {"topk": topk},
            )
            status.append(f"ok {stem} {topk[0]['label']} {topk[0]['score']}")
        except Exception as exc:  # noqa: BLE001 - one failed checkpoint must not drop the rest
            status.append(f"FAIL {stem}: {exc}")
            print(status[-1], flush=True)
    return status


def inside_source(box: list[float], size: tuple[int, int]) -> bool:
    width, height = size
    x1, y1, x2, y2 = box
    return (
        x1 >= SOURCE_EDGE_MARGIN and y1 >= SOURCE_EDGE_MARGIN
        and x2 <= width - SOURCE_EDGE_MARGIN and y2 <= height - SOURCE_EDGE_MARGIN
    )


def save_overlay(stem: str, image: Image.Image, instances: list[dict]) -> list[dict]:
    kept = [
        instance for instance in instances
        if instance["score"] >= CONFIDENCE_THRESHOLD and inside_source(instance["box"], image.size)
    ]
    if not kept:
        raise RuntimeError(f"{stem}: nothing at or above {CONFIDENCE_THRESHOLD} fully inside the photo")
    rendered = render_detections(image, kept)
    rendered.save(OUT_DIR / f"{stem}.webp", "WEBP", quality=90, method=6)
    return record_instances(kept)


def run_yolov5(stem: str, yaml_path: str, weights_url: str, image: str, documented: str) -> str:
    """YOLOv5 v7 checkpoints need the ultralytics/yolov5 loader, not the YOLOv8 package."""
    import torch
    print(f"yolov5 {stem}", flush=True)
    weights = CACHE / Path(weights_url).name
    if not weights.is_file():
        subprocess.run(["curl", "-fsSL", "-o", str(weights), weights_url], check=True)
    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights), trust_repo=True)
    model.conf = CONFIDENCE_THRESHOLD
    photo = Image.open(materialize(image)).convert("RGB")
    prediction = model(str(materialize(image))).xyxy[0].detach().cpu().numpy()
    names = model.names
    instances = []
    for row in prediction:
        score = float(row[4])
        if score < CONFIDENCE_THRESHOLD:
            continue
        instances.append({
            "label": str(names[int(row[5])]),
            "score": score,
            "box": [float(value) for value in row[:4]],
        })
    drawn = save_overlay(stem, photo, instances)
    write_record(
        stem, yaml_path, "detection-overlay", image, documented, weights.name, weights_url,
        f"yolov5-torch-hub-cpu/{torch.__version__}",
        {"confidence_threshold": CONFIDENCE_THRESHOLD, "detections": drawn},
        note=DET_NOTE,
    )
    return f"ok {stem} {drawn[0]['label']} {drawn[0]['score']} n={len(drawn)}"


def run_yolo(stem: str, yaml_path: str, kind: str, weights_url: str, image: str, documented: str) -> str:
    from ultralytics import YOLO
    print(f"yolo {stem}", flush=True)
    weights = CACHE / Path(weights_url).name
    if not weights.is_file():
        subprocess.run(["curl", "-fsSL", "-o", str(weights), weights_url], check=True)
    model = YOLO(str(weights))
    photo = Image.open(materialize(image)).convert("RGB")
    result = model.predict(
        str(materialize(image)), conf=CONFIDENCE_THRESHOLD, iou=0.7, verbose=False, max_det=30,
    )[0]
    boxes = result.boxes
    if boxes is None:
        return f"FAIL {stem}: no detections"
    names = result.names
    instances = []
    for index in range(len(boxes)):
        score = float(boxes.conf[index])
        if score < CONFIDENCE_THRESHOLD:
            continue
        polygons = []
        if result.masks is not None and index < len(result.masks.xy):
            points = [[float(x), float(y)] for x, y in result.masks.xy[index]]
            if len(points) >= 3:
                polygons.append(points)
        instances.append({
            "label": str(names[int(boxes.cls[index])]),
            "score": score,
            "box": [float(value) for value in boxes.xyxy[index].tolist()],
            "polygons": polygons,
        })
    if kind == "segmentation-overlay" and not any(instance.get("polygons") for instance in instances):
        return f"FAIL {stem}: model returned boxes without masks"
    drawn = save_overlay(stem, photo, instances)
    payload = {"confidence_threshold": CONFIDENCE_THRESHOLD, "detections": drawn}
    if kind == "segmentation-overlay":
        payload["masks"] = True
    write_record(
        stem, yaml_path, kind, image, documented, weights.name, weights_url,
        "ultralytics-cpu", payload, note=DET_NOTE,
    )
    return f"ok {stem} {drawn[0]['label']} {drawn[0]['score']} n={len(drawn)}"


def run_fasterrcnn(image: str) -> str:
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    print("fasterrcnn", flush=True)
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    url = weights.url
    expected = "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
    if url != expected:
        return f"FAIL fasterrcnn: weight URL changed to {url}"
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval()
    photo = Image.open(materialize(image)).convert("RGB")
    tensor = weights.transforms()(photo)
    with torch.inference_mode():
        prediction = model([tensor])[0]
    instances = []
    for score, label, box in zip(prediction["scores"], prediction["labels"], prediction["boxes"]):
        value = float(score)
        if value < CONFIDENCE_THRESHOLD:
            continue
        instances.append({
            "label": weights.meta["categories"][int(label)],
            "score": value,
            "box": [float(item) for item in box.tolist()],
        })
    drawn = save_overlay("fasterrcnn-detect", photo, instances)
    write_record(
        "fasterrcnn-detect",
        "model_zoo_web/data/vision/fasterrcnn/fasterrcnn/detect.yaml",
        "detection-overlay",
        image,
        "samples/vision/fasterrcnn/conversion/README.md",
        "fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
        url,
        f"pytorch-cpu/{torch.__version__} torchvision",
        {"confidence_threshold": CONFIDENCE_THRESHOLD, "detections": drawn},
        note=DET_NOTE,
    )
    return f"ok fasterrcnn {drawn[0]['label']} {drawn[0]['score']} n={len(drawn)}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    lines = [] if "--det-only" in sys.argv else run_classifiers()
    try:
        lines.append(run_yolov5(
            "yolov5-detect",
            "model_zoo_web/data/vision/yolov5/yolov5/detect.yaml",
            "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
            DET_SCENE,
            "samples/vision/yolov5/conversion/README.md",
        ))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"FAIL yolov5: {exc}")
    try:
        lines.append(run_yolo(
        "yoloe11-seg",
        "model_zoo_web/data/vision/yoloe/yoloe11/seg.yaml",
        "segmentation-overlay",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt",
        DET_SCENE,
        "samples/vision/yoloe/conversion/README.md",
        ))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"FAIL yoloe11: {exc}")
    try:
        lines.append(run_fasterrcnn(DET_SCENE))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"FAIL fasterrcnn: {exc}")
    if "--det-only" in sys.argv:
        print("\n".join(lines))
        return
    for yaml_path, plain in PLAIN:
        strip_hero(yaml_path, plain)
        lines.append(f"plain {yaml_path} -> {plain}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
