"""The maintained YOLO inventory excludes redundant standalone S implementations."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/python"))
from dataclasses import replace
from utils.py_utils.assets import resolve_asset
from test_detection_binding import _metadata, _contract
from samples.vision.ultralytics_yolo.runtime.python.backend import (
    ModelSelection,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.cli import (
    available_families,
)
from samples.vision.ultralytics_yolo.runtime.python.cli import (
    resolve_platform,
)


class MaintainedScope(unittest.TestCase):
    def test_removed_standalone_ids_cannot_be_resolved(self):
        for sample, filename in [
            ("yolo11", "s100/yolo11n_detect_nashe_640x640_nv12.hbm"),
            ("yolo11_pose", "s100/yolo11n_pose_nashe_640x640_nv12.hbm"),
            ("yolo11_seg", "s100/yolo11n_seg_nashe_640x640_nv12.hbm"),
            ("yolov13_imoonlab", "s100/yolo13n_detect_nashe_640x640_nv12.hbm"),
        ]:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                resolve_asset(f"s:{sample}:{filename}")

    def test_s_imoonlab_family_removed_without_removing_unified_yolo(self):
        families = available_families(resolve_platform("s100"))
        self.assertNotIn("yolov13", families)
        for family in ("yolov8", "yolo11", "yolo26"):
            self.assertIn(family, families)
        self.assertIn("yolov13", available_families(resolve_platform("x5")))

    def test_public_commands_reject_retired_choices_and_keep_unified_pose(self):
        import subprocess
        runtime=Path(__file__).resolve().parents[1]/'runtime/python'
        cases=[(['--platform','s100','--task','pose','--asset-id','s:yolo11_pose:s100/yolo11n_pose_nashe_640x640_nv12.hbm','--dry-run'],2),
               (['--platform','s100','--task','pose','--family','yolo11','--dry-run'],0),
               (['--platform','s100','--task','detect','--family','yolov13','--dry-run'],2)]
        for args,code in cases:
            result=subprocess.run([sys.executable,str(runtime/'main.py'),*args],capture_output=True,text=True)
            self.assertEqual(result.returncode,code,result.stdout+result.stderr)
        result=subprocess.run([sys.executable,str(runtime/'main.py'),'--platform','s100','--list-models'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for retired in ('s:yolo11:', 's:yolo11_pose:', 's:yolo11_seg:', 's:yolov13_imoonlab:'):
            self.assertNotIn(retired,result.stdout)

    def test_only_runtime_dequantized_float_outputs_are_accepted(self):
        selection = ModelSelection("fixture.bin", target="x5", contract=_contract())
        metadata = _metadata()
        import numpy as np

        for dtype in (np.int8, np.int16, np.int32, np.float32):
            quants = {
                n: {"scale": 0.25, "zero_point": 3} for n in metadata.output_names
            }
            changed = replace(
                metadata,
                output_dtypes={n: dtype for n in metadata.output_names},
                output_quantization=quants,
            )
            with self.subTest(dtype=dtype), self.assertRaisesRegex(
                ValueError, "floating|floating-point|float"
            ):
                bind_model(selection, changed)
        self.assertIsNotNone(bind_model(selection, metadata))
