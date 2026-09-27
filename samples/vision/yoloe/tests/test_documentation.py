"""Run the complete bilingual library examples with a fake SDK and real binding."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_runtime import ROOT, fake_runtime
from samples.vision.yoloe.runtime.python.model_binding import list_models
from samples.vision.yoloe.runtime.python.model_runner import build_runner


class DocumentationTests(unittest.TestCase):
    def test_bilingual_library_examples(self):
        def factory(selection):
            runtime, _ = fake_runtime(selection)
            return build_runner(
                selection,
                runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
            )

        for lang in ("README.md", "README_cn.md"):
            text = (ROOT / "samples/vision/yoloe/runtime/python" / lang).read_text()
            blocks = re.findall(r"```python\n(.*?)```", text, re.S)
            self.assertEqual(len(blocks), 1)
            for code in blocks:
                with patch(
                    "samples.vision.yoloe.runtime.python.model_runner.build_runner",
                    side_effect=factory,
                ), redirect_stdout(io.StringIO()):
                    scope = {}
                    exec(compile(code, lang, "exec"), scope)
                self.assertEqual(scope["result"].mask_layout, "full")
                self.assertGreater(len(scope["result"].boxes), 0)

    def test_publication_documentation_covers_exact_assets_and_hashes(self):
        for lang in ("README.md", "README_cn.md"):
            text = (ROOT / "samples/vision/yoloe/model" / lang).read_text()
            for _, _, asset in list_models():
                self.assertIn(asset.reference, text)
                self.assertIn(asset.sha256 or "null (unknown)", text)
