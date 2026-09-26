"""CLI writes raw numeric evidence separately from source-compatible displays."""

import contextlib, io, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
from samples.vision.lanenet.runtime.python import main
from samples.vision.lanenet.runtime.python.model_runner import RuntimeModelRunner
from test_lanenet import metadata, raw_outputs


class CliTests(unittest.TestCase):
    def test_host_inspection_and_rejection(self):
        for args in (["--list-models"], ["--target", "s100", "--dry-run"]):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main.main(args), 0)
            json.loads(out.getvalue())
        for args in (
            ["--target", "s100p"],
            ["--target", "s600"],
            ["--priority", "256"],
            ["--bpu-cores", "-1"],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main.main([*args, "--dry-run"]), 2)

    def test_all_outputs_saved_and_source_flags_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "fixture.hbm"
            model.write_bytes(b"host fixture")
            image = root / "input.png"
            cv2.imwrite(str(image), np.zeros((20, 30, 3), np.uint8))
            m = metadata(True)
            raw = raw_outputs(True)
            schedule = []
            runtime = SimpleNamespace(
                **{k: v for k, v in m.items() if k != "model_name"},
                run=lambda data: {"lane": raw},
                set_scheduling_params=lambda **kw: schedule.append(kw)
            )
            with patch(
                "samples.vision.lanenet.runtime.python.main.RuntimeModelRunner",
                side_effect=lambda s: RuntimeModelRunner(s, runtime=runtime),
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main.main(
                        [
                            "--asset-id",
                            "s:lanenet:s100/lanenet256x512.hbm",
                            "--model-path",
                            str(model),
                            "--test-img",
                            str(image),
                            "--output",
                            str(root / "out"),
                            "--instance-save-path",
                            str(root / "extra-instance.png"),
                            "--binary-save-path",
                            str(root / "extra-binary.png"),
                        ]
                    ),
                    0,
                )
            report = json.loads((root / "out/report.json").read_text())
            self.assertFalse(report["clustering_performed"])
            self.assertEqual(set(report["raw_tensor_keys"]), set(raw))
            with np.load(root / "out/raw_outputs.npz", allow_pickle=False) as saved:
                for name, key in report["raw_tensor_keys"].items():
                    np.testing.assert_array_equal(saved[key], raw[name])
            self.assertEqual(np.load(root / "out/embedding.npy").shape, (3, 256, 512))
            self.assertEqual(np.load(root / "out/binary.npy").dtype, np.uint8)
            self.assertEqual(
                cv2.imread(str(root / "extra-instance.png")).shape, (256, 512, 3)
            )
            self.assertEqual(
                schedule, [{"priority": {"lane": 0}, "bpu_cores": {"lane": [0]}}]
            )

    def test_extra_paths_cannot_collide_or_replace_canonical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            for args in (
                ["--instance-save-path", str(out / "binary_pred.png")],
                ["--instance-save-path", str(out / "report.json")],
                ["--binary-save-path", str(out / "embedding.npy")],
                [
                    "--instance-save-path",
                    str(out / "extra.png"),
                    "--binary-save-path",
                    str(out / "extra.png"),
                ],
            ):
                with patch(
                    "samples.vision.lanenet.runtime.python.main.RuntimeModelRunner"
                ) as factory, contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main.main(["--output", str(out), *args]), 2)
                    factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
