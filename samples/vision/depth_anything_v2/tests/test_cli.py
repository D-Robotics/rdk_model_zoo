"""Host CLI evidence, exact preparation and no-board inspection."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from samples.vision.depth_anything_v2.runtime.python import main
from samples.vision.depth_anything_v2.model import download
from samples.vision.depth_anything_v2.runtime.python.model_runner import (
    RuntimeModelRunner,
)
from test_depth import metadata


class CliTests(unittest.TestCase):
    def test_inspect_and_unsupported_targets(self):
        for args in (["--list-models"], ["--target", "s100", "--dry-run"]):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main.main(args), 0)
            json.loads(out.getvalue())
        for args in (
            ["--target", "s100p"],
            ["--priority", "256"],
            ["--bpu-cores", "-1"],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main.main([*args, "--dry-run"]), 2)

    def test_real_cli_with_injected_runtime_saves_float_and_color(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            image = p / "input.png"
            cv2.imwrite(str(image), np.zeros((7, 13, 3), np.uint8))
            model = p / "depth.hbm"
            model.write_bytes(b"host fixture")
            raw = np.arange(518 * 686, dtype=np.float32).reshape(1, 518, 686)
            m = metadata()
            schedule = []
            runtime = SimpleNamespace(
                **{k: v for k, v in m.items() if k != "model_name"},
                run=lambda data: {"depth": {"pred": raw}},
                set_scheduling_params=lambda **kw: schedule.append(kw)
            )
            with patch(
                "samples.vision.depth_anything_v2.runtime.python.main.RuntimeModelRunner",
                side_effect=lambda s: RuntimeModelRunner(s, runtime=runtime),
            ), contextlib.redirect_stdout(io.StringIO()):
                rc = main.main(
                    [
                        "--asset-id",
                        "s:depth_anything_v2:s100/depth_any.hbm",
                        "--model-path",
                        str(model),
                        "--test-img",
                        str(image),
                        "--output",
                        str(p / "out"),
                        "--img-save-path",
                        str(p / "legacy.jpg"),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(np.load(p / "out/depth_native.npy").shape, (7, 13))
            self.assertEqual(cv2.imread(str(p / "legacy.jpg")).shape, (7, 13, 3))
            report = json.loads((p / "out/report.json").read_text())
            self.assertEqual(
                report["input_normalization"], "pixelwise RGB z-score, epsilon=1e-5"
            )
            self.assertEqual(len(report["model_sha256"]), 64)
            self.assertEqual(
                schedule, [{"priority": {"depth": 0}, "bpu_cores": {"depth": [0]}}]
            )
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main.main(["--output", str(p / "out")]), 2)

    def test_additional_color_path_cannot_replace_canonical_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "out"
            for name in (
                "depth_gray.png",
                "depth_color.png",
                "raw_depth.npy",
                "depth_native.npy",
                "report.json",
            ):
                with patch(
                    "samples.vision.depth_anything_v2.runtime.python.main.RuntimeModelRunner"
                ) as runner, contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        main.main(
                            [
                                "--output",
                                str(output),
                                "--img-save-path",
                                str(output / name),
                            ]
                        ),
                        2,
                    )
                    runner.assert_not_called()

    def test_download_exact_single_manifest(self):
        with patch.object(
            download, "download_asset", return_value="a" * 64
        ) as call, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                download.main(["--target", "s100", "--output-dir", "/tmp/prepared"]), 0
            )
        asset, path = call.call_args.args
        self.assertEqual(asset.reference, "s:depth_anything_v2:s100/depth_any.hbm")
        self.assertEqual(path, Path("/tmp/prepared/s100/depth_any.hbm"))


if __name__ == "__main__":
    unittest.main()
