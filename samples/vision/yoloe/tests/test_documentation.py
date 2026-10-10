"""Run the complete bilingual library examples with a fake SDK and real binding."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from test_runtime import ROOT, fake_runtime
from samples.vision.yoloe.runtime.python.cli import list_models
from samples.vision.yoloe.runtime.python.yoloe import build_runner


class DocumentationTests(unittest.TestCase):
    @staticmethod
    def _section(text, anchor):
        start = text.index(f'<a id="{anchor}"></a>')
        end = text.find('<a id="', start + 1)
        return text[start:] if end == -1 else text[start:end]

    def _python_blocks(self, text, anchor):
        return re.findall(r"```python\n(.*?)```", self._section(text, anchor), re.S)

    def test_bilingual_library_examples(self):
        def factory(selection):
            runtime, _ = fake_runtime(selection)
            created.append(runtime)
            return build_runner(
                selection,
                runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
            )

        for lang in ("README.md", "README_cn.md"):
            text = (ROOT / "samples/vision/yoloe/runtime/python" / lang).read_text()
            integration = self._python_blocks(text, "integration-example")
            stages = self._python_blocks(text, "stage-io")
            self.assertEqual(len(integration), 1, (lang, "integration-example"))
            self.assertEqual(len(stages), 1, (lang, "stage-io"))
            created = []
            scope = {}
            with patch(
                "samples.vision.yoloe.runtime.python.yoloe.build_runner",
                side_effect=factory,
            ), redirect_stdout(io.StringIO()):
                exec(compile(integration[0], lang, "exec"), scope)
                primary = scope["result"]
                self.assertEqual(len(created), 1)
                self.assertEqual(created[0].calls, 1)
                exec(compile(stages[0], lang, "exec"), scope)
                staged = scope["result"]
                self.assertEqual(created[0].calls, 2)
            self.assertEqual(primary.mask_layout, "full")
            self.assertGreater(len(primary.boxes), 0)
            np.testing.assert_array_equal(primary.boxes, staged.boxes)
            np.testing.assert_array_equal(primary.scores, staged.scores)
            np.testing.assert_array_equal(primary.class_ids, staged.class_ids)
            np.testing.assert_array_equal(np.asarray(primary.masks), np.asarray(staged.masks))

    def test_publication_documentation_covers_exact_assets_and_hashes(self):
        for lang in ("README.md", "README_cn.md"):
            text = (ROOT / "samples/vision/yoloe/model" / lang).read_text()
            for _, _, asset in list_models():
                self.assertIn(asset.reference, text)
                self.assertIn(asset.sha256 or "null (unknown)", text)
