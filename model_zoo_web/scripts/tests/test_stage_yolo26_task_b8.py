import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).resolve().parents[1] / "stage_yolo26_task_b8.py"
SPEC = importlib.util.spec_from_file_location("stage_yolo26_task_b8", SCRIPT)
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


class CandidateRefreshEvidenceTests(unittest.TestCase):
    def test_explicit_snapshot_path_must_be_new_immutable_candidate_child(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, snapshot_id = stage.resolve_snapshot_output(
                root, "yolo26-task-b8-audit-20260930T041500Z-r1",
            )
            self.assertEqual(path, root / snapshot_id)
            self.assertEqual(snapshot_id, "yolo26-task-b8-audit-20260930T041500Z-r1")
            with self.assertRaisesRegex(ValueError, "direct candidate-staging child"):
                stage.resolve_snapshot_output(root, "nested/yolo26-task-b8-audit-20260930T041500Z")
            with self.assertRaisesRegex(ValueError, "stay below candidate-staging"):
                stage.resolve_snapshot_output(root, "../outside/yolo26-task-b8-audit-20260930T041500Z")

    def test_comparator_metrics_keep_full_precision_and_web_field_mapping(self):
        board = {"mAP50": 0.532333814483004}
        floating = {"mAP50": 0.547891237904812}
        retention = board["mAP50"] / floating["mAP50"]
        comparator = {
            "metrics": {
                "mAP50": {
                    "float": floating["mAP50"],
                    "board": board["mAP50"],
                    "board_minus_float": board["mAP50"] - floating["mAP50"],
                    "retention_ratio": retention,
                },
            },
        }
        float_web, board_web, comparison_web = stage.comparison_metrics(
            "obb", comparator, board, floating,
        )
        self.assertEqual(float_web["map_50"], floating["mAP50"])
        self.assertEqual(board_web["map_50"], board["mAP50"])
        self.assertEqual(comparison_web["map_50"]["retention_ratio"], retention)
        self.assertNotEqual(board_web["map_50"], round(board["mAP50"], 3))

    def test_comparator_metric_mismatch_fails_closed(self):
        board = {"top1": 0.4712, "top5": 0.7192}
        floating = {"top1": 0.4989, "top5": 0.7444}
        comparator = {
            "metrics": {
                key: {
                    "float": floating[key],
                    "board": board[key],
                    "board_minus_float": board[key] - floating[key],
                    "retention_ratio": board[key] / floating[key],
                }
                for key in board
            },
        }
        comparator["metrics"]["top1"]["board"] += 0.0001
        with self.assertRaisesRegex(ValueError, "authoritative board metric differs"):
            stage.comparison_metrics("cls", comparator, board, floating)

    def test_missing_authoritative_audit_check_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "did not pass"):
            stage.require_audit_check(
                {"checks": [{"name": "valid_float_board_comparison", "passed": False}]},
                "valid_float_board_comparison",
                "fixture",
            )


if __name__ == "__main__":
    unittest.main()
