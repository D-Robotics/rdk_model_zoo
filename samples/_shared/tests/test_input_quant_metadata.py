"""Multi-input physical quantization metadata survives without SDK object copying."""

import unittest
from types import SimpleNamespace
import numpy as np
from samples._shared.runtime_meta import (
    RuntimeMetadata,
    canonicalise_dtype,
    metadata_evidence,
)


class InputQuantMetadataTests(unittest.TestCase):
    def test_unsigned_source_tokens_normalize_without_accepting_unknown_widths(self):
        for token, expected in [
            ("U16", "uint16"),
            ("hbDNNDataType.U16", "uint16"),
            ("U32", "uint32"),
            ("hbDNNDataType.U32", "uint32"),
        ]:
            self.assertEqual(canonicalise_dtype(token), expected)
        self.assertEqual(canonicalise_dtype("U64"), "u64")

    def test_named_input_quants_survive_mapping_runtime_and_evidence(self):
        class Quant:
            quant_type = "SCALE"
            scale = np.array([0.25], np.float32)
            zero_point = np.array([3], np.int32)
            axis = 0

            def __deepcopy__(self, memo):
                raise TypeError("SDK descriptors cannot be copied")

        q = Quant()
        runtime = SimpleNamespace(
            model_names=["plan"],
            input_names={"plan": ["camera", "status"]},
            input_shapes={"plan": {"camera": (1, 3, 2, 2), "status": (1, 8)}},
            input_dtypes={"plan": {"camera": "U16", "status": "S16"}},
            input_quants={"plan": {"camera": q, "status": q}},
            output_names={"plan": ["trajectory"]},
            output_shapes={"plan": {"trajectory": (1, 8, 3)}},
            output_dtypes={"plan": {"trajectory": "F32"}},
        )
        meta = RuntimeMetadata.from_runtime(runtime)
        self.assertIs(meta.input_quants["camera"], q)
        self.assertEqual(meta.input_dtypes["camera"], "uint16")
        view = metadata_evidence(meta)
        self.assertEqual(view["input_quants"]["camera"]["scale"], [0.25])
        self.assertEqual(view["input_quants"]["camera"]["zero_point"], [3])
        view["input_quants"]["camera"]["scale"][0] = 99
        self.assertEqual(float(q.scale[0]), 0.25)
        flat = RuntimeMetadata.from_mapping(
            {"model_name": "plan", "input_quants": {"status": q}}
        )
        self.assertIs(flat.input_quants["status"], q)
        self.assertEqual(
            RuntimeMetadata.from_mapping({"model_name": "plan"}).input_quants, {}
        )


if __name__ == "__main__":
    unittest.main()
