"""Conversion-README shape ↔ binding-contract consistency (B1-R4).

The validation section of the conversion READMEs states the input shapes
a customer should accept per target×variant. Those statements must equal
the runtime binding's contract facts — every published artifact takes a
224x224 input (S series: Y [1,224,224,1] / UV [1,112,112,2]). A
section-presence check cannot catch this drift; parsing the stated shapes
and comparing them to BINDING_TABLE can.
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
    r"s100/s100p/s600[，,]\s*(small|medium)[^\n]*?"
    r"Y\s*`\[1,(\d+),(\d+),1\]`[^\n]*?"
    r"UV\s*`\[1,(\d+),(\d+),2\]`"
)
#: X5 row: "x5, small and medium | one packed NV12 input, 224x224 (...)"
X5_ROW = re.compile(
    r"\|\s*x5[，,][^\n|]*\|[^\n|]*?(\d+)x(\d+)[^\n|]*\|"
)


class ConversionReadmeShapeTests(unittest.TestCase):
    def _binding_facts(self):
        from samples.vision.mobilenetv4.runtime.python.cli import (
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

    def test_s_shapes_are_224_for_both_variants(self):
        # Every published S artifact is 224x224 (Y 224x224, UV 112x112x2),
        # including Medium, which earlier builds shipped at 256x256.
        facts = self._binding_facts()
        for variant in ("small", "medium"):
            for target in ("s100", "s100p", "s600"):
                fact = facts[(variant, target)]
                self.assertEqual(
                    (fact.input_height, fact.input_width), (224, 224),
                    f"{variant}/{target}",
                )
        for readme_name in ("README.md", "README_cn.md"):
            with self.subTest(readme=readme_name):
                section = self._validation_section(readme_name)
                # Match on the variant token, not row order.
                rows = {
                    variant: (h, w, uh, uw)
                    for variant, h, w, uh, uw in S_ROW.findall(section)
                }
                self.assertEqual(rows["medium"], ("224", "224", "112", "112"))
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
