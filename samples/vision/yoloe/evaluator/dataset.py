# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Strict local COCO image identity and explicitly reviewed PF category mappings."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from samples.vision.yoloe.model.vocabulary import LABELS_SHA256


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    raw = Path(path).read_bytes()

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return (
        json.loads(raw, object_pairs_hook=_object, parse_constant=invalid),
        sha256(raw).hexdigest(),
    )


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


@dataclass(frozen=True)
class Dataset:
    document: dict
    annotation_sha256: str
    image_root: Path
    images: tuple
    categories: dict


def load_dataset(annotation, image_root, limit=0):
    integer(limit, "limit")
    document, digest = read_json(annotation)
    if not isinstance(document, dict):
        raise ValueError("Annotation must be a COCO JSON object.")
    root = Path(image_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory missing: {root}")
    rows = document.get("images")
    cats = document.get("categories")
    annotations = document.get("annotations", [])
    if not isinstance(rows, list) or not rows or not isinstance(cats, list) or not cats:
        raise ValueError("Nonempty COCO images and categories are required.")
    if not isinstance(annotations, list):
        raise ValueError("annotations must be a list.")
    images = {}
    paths = set()
    categories = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each image must be an object.")
        identity = integer(row.get("id"), "image id")
        name = row.get("file_name")
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or Path(name).is_absolute()
        ):
            raise ValueError("Image file_name must be a portable relative path.")
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path escapes image_root.")
        if identity in images or path in paths:
            raise ValueError("Duplicate image id or file path.")
        integer(row.get("width"), "image width", 1)
        integer(row.get("height"), "image height", 1)
        images[identity] = dict(row)
        paths.add(path)
    for row in cats:
        if not isinstance(row, dict):
            raise ValueError("Each category must be an object.")
        identity = integer(row.get("id"), "category id")
        if identity in categories:
            raise ValueError("Duplicate category id.")
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError("Category name is required.")
        categories[identity] = row["name"]
    annotation_ids = set()
    for row in annotations:
        if not isinstance(row, dict):
            raise ValueError("Each annotation must be an object.")
        identity = integer(row.get("id"), "annotation id")
        if identity in annotation_ids:
            raise ValueError("Duplicate annotation id.")
        annotation_ids.add(identity)
        if (
            integer(row.get("image_id"), "annotation image id") not in images
            or integer(row.get("category_id"), "annotation category id")
            not in categories
        ):
            raise ValueError("Annotation refers to an unknown image/category.")
    selected = tuple(images[key] for key in sorted(images))
    return Dataset(
        document, digest, root, selected[:limit] if limit else selected, categories
    )


@dataclass(frozen=True)
class CategoryMapping:
    ids: dict
    document: dict
    sha256: str


def load_category_map(path, categories, names):
    if (
        len(names) != 4585
        or any(not isinstance(name, str) for name in names)
        or sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()
        != LABELS_SHA256
    ):
        raise ValueError("Loaded vocabulary differs from the fixed PF label identity.")
    mapping, digest = read_json(path)
    if (
        not isinstance(mapping, dict)
        or mapping.get("vocabulary_sha256") != LABELS_SHA256
    ):
        raise ValueError("Category map must bind the fixed PF vocabulary SHA-256.")
    entries = mapping.get("mapping")
    if not isinstance(entries, list) or not entries:
        raise ValueError("An explicit nonempty category mapping is required.")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Mapping entries must be objects.")
        pf = integer(entry.get("pf_id"), "PF class id")
        category = integer(entry.get("category_id"), "mapped category id")
        if pf >= len(names) or entry.get("pf_name") != names[pf]:
            raise ValueError("PF class id/name mismatch.")
        if (
            category not in categories
            or entry.get("category_name") != categories[category]
        ):
            raise ValueError("Dataset category id/name mismatch.")
        if pf in result:
            raise ValueError("Duplicate PF class mapping.")
        result[pf] = category
    if set(result.values()) != set(categories):
        raise ValueError(
            "Mapping must cover every annotation category; partial category scoring is not implicit."
        )
    return CategoryMapping(result, mapping, digest)
