"""Conversion-README shape <-> binding-contract consistency (B1-R4).

The validation section of the conversion READMEs states the input shapes
a customer should accept per target x variant. Those statements must equal
the runtime binding's contract facts (S series: Y [1,H,W,1] / UV
[1,H/2,W/2,2]; X5: one packed NV12 input of HxW). A section-presence check
cannot catch this drift; parsing the stated shapes and comparing them to
BINDING_TABLE can.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


CONV_DIR = (
    Path(__file__).resolve().parents[1] / "conversion"
)

#: S-side rows: "<targets>, <variant> | Y `[1,H,W,1]`, UV `[1,H/2,W/2,2]` ..."
#: (the Chinese README separates the target list and variant with a
#: full-width comma).
S_ROW = re.compile(
    r"s100/s100p/s600[，,]\s*(\w+)[^\n]*?"
    r"Y\s*`\[1,(\d+),(\d+),1\]`[^\n]*?"
    r"UV\s*`\[1,(\d+),(\d+),2\]`"
)
#: X5 rows: "x5, <variants> | one packed NV12 input, HxW (...)"
X5_ROW = re.compile(
    r"\|\s*x5[，,]([^\n|]*)\|[^\n|]*?(\d+)x(\d+)[^\n|]*\|"
)


class ConversionReadmeShapeTests(unittest.TestCase):
    def _binding_facts(self):
        from samples.vision.mobilenetv1.runtime.python.cli import (
            BINDING_TABLE,
        )

        return BINDING_TABLE.facts

    def _validation_section(self, readme_name):
        text = (CONV_DIR / readme_name).read_text("utf-8")
        start = text.index('<a id="validation"></a>')
        end = text.index('<a id="artifacts"></a>')
        return text[start:end]

    def test_s_rows_state_binding_shapes_per_variant(self):
        from testsupport import VARIANTS

        facts = self._binding_facts()
        for readme_name in ("README.md", "README_cn.md"):
            with self.subTest(readme=readme_name):
                rows = S_ROW.findall(self._validation_section(readme_name))
                self.assertEqual(
                    sorted(r[0] for r in rows), sorted(VARIANTS),
                    "expected one S row per published variant",
                )
                for variant, y_h, y_w, uv_h, uv_w in rows:
                    for target in ("s100", "s100p", "s600"):
                        with self.subTest(variant=variant, target=target):
                            fact = facts[(variant, target)]
                            self.assertEqual(
                                (int(y_h), int(y_w), int(uv_h), int(uv_w)),
                                (
                                    fact.input_height,
                                    fact.input_width,
                                    fact.input_height // 2,
                                    fact.input_width // 2,
                                ),
                            )

    def test_x5_rows_state_packed_geometry_of_their_variants(self):
        from testsupport import VARIANTS

        facts = self._binding_facts()
        for readme_name in ("README.md", "README_cn.md"):
            with self.subTest(readme=readme_name):
                rows = X5_ROW.findall(self._validation_section(readme_name))
                self.assertTrue(rows, "X5 validation row not found")
                covered = set()
                for names, height, width in rows:
                    variants = [v for v in VARIANTS if re.search(rf"\b{re.escape(v)}\b", names)]
                    self.assertTrue(variants, names)
                    for variant in variants:
                        fact = facts[(variant, "x5")]
                        self.assertEqual((int(height), int(width)), (fact.input_height, fact.input_width), variant)
                        covered.add(variant)
                self.assertEqual(covered, set(VARIANTS))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
