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
from PIL import Image, ImageDraw, ImageFont

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


def frame_bgr(image, boxes: list[list[float]]):
    """Cover-crop a BGR frame to the published 16:9 hero, keeping the top box in view."""
    import cv2
    height, width = image.shape[:2]
    target = CANVAS[0] / CANVAS[1]
    if width / height > target:
        crop_h = height
        crop_w = int(round(height * target))
        x0 = max(0, (width - crop_w) // 2)
        y0 = 0
    else:
        crop_w = width
        crop_h = int(round(width / target))
        x0 = 0
        y0 = max(0, (height - crop_h) // 2)
    if boxes:
        cx = (boxes[0][0] + boxes[0][2]) / 2
        cy = (boxes[0][1] + boxes[0][3]) / 2
        if width / height > target:
            x0 = int(min(max(cx - crop_w / 2, 0), width - crop_w))
        else:
            y0 = int(min(max(cy - crop_h / 2, 0), height - crop_h))
    cropped = image[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(cropped, CANVAS, interpolation=cv2.INTER_AREA)


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
                 model_file: str, model_url: str, framework: str, payload: dict) -> None:
    record = {
        "runtime": "host-float-reference",
        "commit": REVISION,
        "model_file": model_file,
        "model_url": model_url,
        "input": image,
        "framework": framework,
        "documented_in": documented,
        "note": NOTE,
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
        "note": NOTE,
        "model_file": model_file,
        "model_url": model_url,
        "framework": framework,
        "commit": REVISION,
        "documented_in": documented,
    }
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


def run_yolov5(stem: str, yaml_path: str, weights_url: str, image: str, documented: str) -> str:
    """YOLOv5 v7 checkpoints need the ultralytics/yolov5 loader, not the YOLOv8 package."""
    import cv2
    import torch
    print(f"yolov5 {stem}", flush=True)
    weights = CACHE / Path(weights_url).name
    if not weights.is_file():
        subprocess.run(["curl", "-fsSL", "-o", str(weights), weights_url], check=True)
    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights), trust_repo=True)
    model.conf = 0.35
    source = str(materialize(image))
    results = model(source)
    prediction = results.xyxy[0].detach().cpu().numpy()
    if len(prediction) == 0:
        return f"FAIL {stem}: no detections"
    names = model.names
    drawn = []
    for row in prediction[:15]:
        drawn.append({
            "label": str(names[int(row[5])]),
            "score": round(float(row[4]), 6),
            "box": [round(float(value), 2) for value in row[:4]],
        })
    rendered = results.render()[0]
    if rendered.shape[2] == 3:
        rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
    framed = frame_bgr(rendered, [row["box"] for row in drawn])
    Image.fromarray(cv2.cvtColor(framed, cv2.COLOR_BGR2RGB)).save(OUT_DIR / f"{stem}.webp", "WEBP", quality=90, method=6)
    write_record(stem, yaml_path, "detection-overlay", image, documented, weights.name, weights_url,
                 f"yolov5-torch-hub-cpu/{torch.__version__}", {"detections": drawn})
    return f"ok {stem} {drawn[0]['label']} {drawn[0]['score']} n={len(drawn)}"


def run_yolo(stem: str, yaml_path: str, kind: str, weights_url: str, image: str, documented: str) -> str:
    import cv2
    from ultralytics import YOLO
    print(f"yolo {stem}", flush=True)
    weights = CACHE / Path(weights_url).name
    if not weights.is_file():
        subprocess.run(["curl", "-fsSL", "-o", str(weights), weights_url], check=True)
    model = YOLO(str(weights))
    source = str(materialize(image))
    result = model.predict(source, conf=0.35, iou=0.7, verbose=False, max_det=15)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return f"FAIL {stem}: no detections"
    names = result.names
    drawn = []
    for index in range(len(boxes)):
        score = round(float(boxes.conf[index]), 6)
        cls = int(boxes.cls[index])
        xyxy = [round(float(value), 2) for value in boxes.xyxy[index].tolist()]
        drawn.append({"label": str(names[cls]), "score": score, "box": xyxy})
    annotated = result.plot(conf=True, labels=True, boxes=True, masks=bool(result.masks))
    framed = frame_bgr(annotated, [row["box"] for row in drawn])
    Image.fromarray(cv2.cvtColor(framed, cv2.COLOR_BGR2RGB)).save(OUT_DIR / f"{stem}.webp", "WEBP", quality=90, method=6)
    payload = {"detections": drawn}
    if kind == "segmentation-overlay":
        if result.masks is None:
            return f"FAIL {stem}: model returned boxes without masks"
        payload["masks"] = True
    write_record(stem, yaml_path, kind, image, documented, weights.name, weights_url,
                 "ultralytics-cpu", payload)
    return f"ok {stem} {drawn[0]['label']} {drawn[0]['score']} n={len(drawn)}"


def run_fasterrcnn() -> str:
    import cv2
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    from ultralytics.utils.plotting import Annotator, colors
    print("fasterrcnn", flush=True)
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    url = weights.url
    expected = "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
    if url != expected:
        return f"FAIL fasterrcnn: weight URL changed to {url}"
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval()
    image = "samples/vision/fasterrcnn/test_data/bus.jpg"
    frame = cv2.imread(str(materialize(image)))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = weights.transforms()(Image.fromarray(rgb))
    with torch.inference_mode():
        prediction = model([tensor])[0]
    drawn = []
    for score, label, box in zip(prediction["scores"], prediction["labels"], prediction["boxes"]):
        value = float(score)
        if value < 0.5:
            continue
        name = weights.meta["categories"][int(label)]
        xyxy = [round(float(item), 2) for item in box.tolist()]
        drawn.append({"label": name, "score": round(value, 6), "box": xyxy})
        if len(drawn) == 15:
            break
    if not drawn:
        return "FAIL fasterrcnn: no detections"
    annotator = Annotator(frame, line_width=max(2, round(frame.shape[0] / 300)))
    for index, row in enumerate(drawn):
        annotator.box_label(row["box"], f"{row['label']} {row['score']:.2f}", color=colors(index, True))
    framed = frame_bgr(annotator.result(), [row["box"] for row in drawn])
    Image.fromarray(cv2.cvtColor(framed, cv2.COLOR_BGR2RGB)).save(
        OUT_DIR / "fasterrcnn-detect.webp", "WEBP", quality=90, method=6,
    )
    write_record(
        "fasterrcnn-detect",
        "model_zoo_web/data/vision/fasterrcnn/fasterrcnn/detect.yaml",
        "detection-overlay",
        image,
        "samples/vision/fasterrcnn/conversion/README.md",
        "fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
        url,
        f"pytorch-cpu/{torch.__version__} torchvision",
        {"detections": drawn},
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
            "samples/vision/yolov5/test_data/bus.jpg",
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
        "samples/vision/yolov5/test_data/bus.jpg",
        "samples/vision/yoloe/conversion/README.md",
        ))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"FAIL yoloe11: {exc}")
    try:
        lines.append(run_fasterrcnn())
    except Exception as exc:  # noqa: BLE001
        lines.append(f"FAIL fasterrcnn: {exc}")
    for yaml_path, plain in PLAIN:
        strip_hero(yaml_path, plain)
        lines.append(f"plain {yaml_path} -> {plain}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
