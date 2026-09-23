# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for EfficientSAM's bilingual customer documentation."""

import contextlib
from pathlib import Path
import re
import shlex
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from samples.vision.efficient_sam.runtime.python.main import build_parser
from samples.vision.efficient_sam.model.download import build_parser as build_download_parser
from samples._shared.sam_evaluator import build_parser as build_evaluator_parser
from samples._shared.runtime_meta import RuntimeMetadata
from samples._shared.sam_binding import bind_model, resolve_selection


ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "samples/vision/efficient_sam"
DOCS = (SAMPLE / "README.md", SAMPLE / "model/README.md", SAMPLE / "runtime/python/README.md", SAMPLE / "evaluator/README.md")


class ReadmeContractTests(unittest.TestCase):
    def test_bilingual_anchor_sets_and_links(self):
        expected = {
            "README.md": {"overview", "support-matrix", "prerequisites", "quickstart", "expected-results", "directory", "entry-points", "license"},
            "model/README.md": {"artifacts", "preparation", "accompanying-files", "local-paths", "formats-checksums"},
            "runtime/python/README.md": {"environment", "usage", "parameters", "results", "integration-example", "stage-io", "troubleshooting"},
            "evaluator/README.md": {"dataset", "environment", "command", "metrics", "outputs", "reference-results", "boundaries"},
        }
        for english in DOCS:
            for language in (english, english.with_name("README_cn.md")):
                text = language.read_text(encoding="utf-8")
                self.assertTrue(text.startswith("English") or text.startswith("[English]"), language)
                anchors = set(re.findall(r'<a id="([^"]+)"></a>', text))
                self.assertEqual(anchors, expected[english.relative_to(SAMPLE).as_posix()], language)
                for target in re.findall(r'\]\(([^)]+)\)', text):
                    if "://" not in target:
                        self.assertTrue((language.parent / target.split("#", 1)[0]).exists(), (language, target))

    def test_documented_cli_commands_parse_and_fixture_paths_exist(self):
        runtime_parser = build_parser()
        download_parser = build_download_parser()
        evaluator_parser = build_evaluator_parser("efficient_sam")
        for path in SAMPLE.rglob("README*.md"):
            text = path.read_text(encoding="utf-8").replace("\\\n", " ")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("python3 samples/", ".venv/bin/python samples/")):
                    argv = shlex.split(stripped)
                    if "/runtime/python/main.py" in argv[1]:
                        options = runtime_parser.parse_args(argv[2:])
                        self.assertTrue(Path(options.test_img).is_file(), (path, options.test_img))
                    elif "/model/download.py" in argv[1]:
                        download_parser.parse_args(argv[2:])
                    elif "/evaluator/compare.py" in argv[1]:
                        evaluator_parser.parse_args(argv[2:])

    def test_runtime_api_examples_compile_and_use_qualified_modules(self):
        runner_module = __import__("samples.vision.efficient_sam.runtime.python.model_runner", fromlist=["RuntimeModelRunner"])

        class Stage:
            def __init__(self, output):
                self.output = output
            def __call__(self, _inputs):
                return self.output

        encoder_meta = RuntimeMetadata.from_mapping({
            "model_name": "encoder", "input_names": ("batched_images",),
            "input_shapes": {"batched_images": (1, 3, 512, 512)},
            "input_dtypes": {"batched_images": "float32"},
            "output_names": ("image_embeddings",),
            "output_shapes": {"image_embeddings": (1, 256, 32, 32)},
            "output_dtypes": {"image_embeddings": "float16"},
        })
        decoder_meta = RuntimeMetadata.from_mapping({
            "model_name": "decoder", "input_names": ("image_embeddings",),
            "input_shapes": {"image_embeddings": (1, 256, 32, 32)},
            "input_dtypes": {"image_embeddings": "float32"},
            "output_names": ("low_res_masks", "iou_predictions"),
            "output_shapes": {"low_res_masks": (1, 3, 128, 128), "iou_predictions": (1, 3)},
            "output_dtypes": {"low_res_masks": "float16", "iou_predictions": "float16"},
        })

        class FakeRunner:
            def __init__(self, selection):
                self.binding = bind_model(selection, encoder_meta, decoder_meta)
                self.encoder = Stage({"image_embeddings": np.ones((1, 256, 32, 32), np.float16)})
                self.decoder = Stage({
                    "low_res_masks": np.ones((1, 3, 128, 128), np.float16),
                    "iou_predictions": np.array([[0.1, 0.9, 0.2]], np.float16),
                })
            def load(self):
                return self.binding
            def set_scheduling_params(self, **_kwargs):
                return None

        for language in ("README.md", "README_cn.md"):
            text = (SAMPLE / "runtime/python" / language).read_text(encoding="utf-8")
            snippets = re.findall(r"```python\n(.*?)```", text, re.S)
            self.assertEqual(len(snippets), 1)
            with patch.object(runner_module, "RuntimeModelRunner", FakeRunner), contextlib.redirect_stdout(None):
                scope = {}
                exec(compile(snippets[0], language, "exec"), scope)
            self.assertEqual(scope["result"]["mask"].shape, (512, 512))
            self.assertIn("EfficientSAMPipeline", snippets[0])


if __name__ == "__main__":
    unittest.main()
