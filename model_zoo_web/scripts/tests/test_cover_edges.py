import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "check_cover_edges.py"
SPEC = importlib.util.spec_from_file_location("cover_edges", SCRIPT)
edges = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(edges)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "release" / "assets" / "resnet-cls.webp"
HEROES = ROOT / "release" / "assets" / "pending-heroes"
SIDES = ("top", "bottom", "left", "right")


def scene(height, width, border, color):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    image[..., 0] = (xx * 3) % 251
    image[..., 1] = (yy * 5) % 251
    image[..., 2] = ((xx + yy) * 2) % 251
    if border:
        image[:border] = color
        image[-border:] = color
        image[:, :border] = color
        image[:, -border:] = color
    return image


def matte_reference(thickness, color=(236, 236, 238)):
    return {side: {"thickness": thickness, "color": list(color)} for side in SIDES}


class CoverEdgeTests(unittest.TestCase):
    def test_shipped_pending_heroes_match_the_published_card(self):
        self.assertEqual(edges.check_covers(REFERENCE, HEROES, HEROES), [])

    def test_plain_cover_may_keep_a_two_pixel_fringe(self):
        image = scene(24, 40, 2, (255, 255, 255))
        found = edges.problems_for(
            image,
            classification=False,
            reference_strips=matte_reference(40),
            expected_size=(40, 24),
        )
        self.assertEqual(found, [])

    def test_white_border_fails_even_on_a_classification_panel(self):
        image = scene(24, 40, 8, (255, 255, 255))
        found = edges.problems_for(
            image,
            classification=True,
            reference_strips=matte_reference(8, (255, 255, 255)),
            expected_size=(40, 24),
        )
        self.assertTrue(any("near-white" in item for item in found), found)

    def test_other_uniform_border_fails_for_detection_covers(self):
        image = scene(24, 40, 6, (236, 236, 238))
        found = edges.problems_for(
            image,
            classification=False,
            reference_strips=matte_reference(6),
            expected_size=(40, 24),
        )
        self.assertTrue(any("uniform" in item for item in found), found)

    def test_classification_matte_must_match_the_published_resnet_edge(self):
        image = scene(24, 40, 6, (236, 236, 238))
        matching = edges.problems_for(
            image,
            classification=True,
            reference_strips=matte_reference(10),
            expected_size=(40, 24),
        )
        self.assertEqual(matching, [])
        drifted = edges.problems_for(
            image,
            classification=True,
            reference_strips=matte_reference(1),
            expected_size=(40, 24),
        )
        self.assertTrue(any("uniform" in item for item in drifted), drifted)
        wrong_color = edges.problems_for(
            scene(24, 40, 6, (210, 210, 210)),
            classification=True,
            reference_strips=matte_reference(6),
            expected_size=(40, 24),
        )
        self.assertTrue(any("uniform" in item for item in wrong_color), wrong_color)

    def test_wrong_dimensions_fail(self):
        found = edges.problems_for(
            scene(10, 10, 0, (0, 0, 0)),
            classification=False,
            reference_strips=matte_reference(40),
            expected_size=(40, 24),
        )
        self.assertIn("size 10x10, expected 40x24", found)

    def test_saved_white_border_fails_the_directory_check(self):
        source = np.array(Image.open(HEROES / "yolov5-detect.webp").convert("RGB"))
        source[:8] = 255
        source[-8:] = 255
        source[:, :8] = 255
        source[:, -8:] = 255
        with tempfile.TemporaryDirectory() as tmp:
            heroes = Path(tmp) / "heroes"
            heroes.mkdir()
            Image.fromarray(source).save(heroes / "yolov5-detect.webp", lossless=True)
            found = edges.check_covers(REFERENCE, heroes, Path(tmp))
        self.assertTrue(any("near-white" in item for item in found), found)
