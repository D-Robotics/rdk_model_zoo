"""Conversion-README shape ↔ binding-contract consistency (B1-R4).

The validation section of the conversion READMEs states the input shapes
a customer should accept per target×variant. Those statements must equal
the runtime binding's contract facts — for the S medium artifact that is
Y [1,256,256,1] / UV [1,128,128,2], not the 224 shapes of every other
variant. A section-presence check cannot catch this drift; parsing the
stated shapes and comparing them to BINDING_TABLE can.
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
    r"s100/s600[，,]\s*(small|medium)[^\n]*?"
    r"Y\s*`\[1,(\d+),(\d+),1\]`[^\n]*?"
    r"UV\s*`\[1,(\d+),(\d+),2\]`"
)
#: X5 row: "x5, small and medium | one packed NV12 input, 224x224 (...)"
X5_ROW = re.compile(
    r"\|\s*x5[，,][^\n|]*\|[^\n|]*?(\d+)x(\d+)[^\n|]*\|"
)


class ConversionReadmeShapeTests(unittest.TestCase):
    def _binding_facts(self):
        from samples.vision.mobilenetv4.runtime.python.model_binding import (
            BINDING_TABLE,
        )

        return BINDING_TABLE.facts

    def _validation_section(self, readme_name):
        text = (CONV_DIR / readme_name).read_text("utf-8")
        start = text.index('<a id="validation"></a>')
        end = text.index('<a id="artifacts"></a>')
        return text[start:end]

    def test_s_rows_state_binding_shapes_per_variant(self):
        facts = self._binding_facts()
        for readme_name in ("README.md", "README_cn.md"):
            with self.subTest(readme=readme_name):
                section = self._validation_section(readme_name)
                rows = S_ROW.findall(section)
                self.assertEqual(
                    len(rows), 2, "expected small and medium S rows"
                )
                for variant, y_h, y_w, uv_h, uv_w in rows:
                    with self.subTest(variant=variant):
                        fact = facts[(variant, "s100")]
                        self.assertEqual(
                            (int(y_h), int(y_w), int(uv_h), int(uv_w)),
                            (
                                fact.input_height,
                                fact.input_width,
                                fact.input_height // 2,
                                fact.input_width // 2,
                            ),
                        )

    def test_medium_s_shapes_are_256(self):
        # The regression B1-R4 pinned: S medium is 256x256, and its UV
        # plane is 128x128x2 — previously both languages stated 224/112.
        facts = self._binding_facts()
        self.assertEqual(
            (facts[("medium", "s100")].input_height,
             facts[("medium", "s100")].input_width),
            (256, 256),
        )
        for readme_name in ("README.md", "README_cn.md"):
            with self.subTest(readme=readme_name):
                section = self._validation_section(readme_name)
                # Match on the variant token, not row order.
                rows = {
                    variant: (h, w, uh, uw)
                    for variant, h, w, uh, uw in S_ROW.findall(section)
                }
                self.assertEqual(rows["medium"], ("256", "256", "128", "128"))
                self.assertEqual(rows["small"], ("224", "224", "112", "112"))

    def test_x5_row_states_packed_geometry_of_both_variants(self):
        facts = self._binding_facts()
        self.assertEqual(
            facts[("small", "x5")].input_height,
            facts[("medium", "x5")].input_height,
            "the README's single X5 row assumes both variants share a "
            "geometry; the binding disagrees",
        )
        for readme_name in ("README.md", "README_cn.md"):
            with self.subTest(readme=readme_name):
                section = self._validation_section(readme_name)
                match = X5_ROW.search(section)
                self.assertIsNotNone(match, "X5 validation row not found")
                self.assertEqual(
                    (int(match.group(1)), int(match.group(2))),
                    (
                        facts[("small", "x5")].input_height,
                        facts[("small", "x5")].input_width,
                    ),
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
