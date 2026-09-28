import unittest
from samples.speech.kws.evaluator.metrics import evaluate


class MetricTests(unittest.TestCase):
    def test_confusion_and_threshold_equality(self):
        rows = [
            {"id": str(i), "label": label, "score": score}
            for i, (label, score) in enumerate([(1, 0.5), (1, 0.2), (0, 0.7), (0, 0.1)])
        ]
        out = evaluate(rows, 0.5)
        self.assertEqual(out["confusion"], dict(tp=1, tn=1, fp=1, fn=1))
        self.assertEqual(out["precision"], 0.5)
        self.assertEqual(out["recall"], 0.5)
        self.assertEqual(out["false_accept_rate"], 0.5)
        self.assertEqual(out["false_reject_rate"], 0.5)

    def test_undefined_denominators_and_invalid_data(self):
        out = evaluate([{"id": "a", "label": 0, "score": 0}], 0.5)
        self.assertIsNone(out["precision"])
        self.assertIsNone(out["recall"])
        for rows in (
            [],
            [{"id": "a", "label": True, "score": 0}],
            [{"id": "a", "label": 0, "score": float("nan")}],
            [{"id": "a", "label": 0, "score": 1}] * 2,
        ):
            with self.assertRaises(ValueError):
                evaluate(rows, 0.5)
        with self.assertRaises(ValueError):
            evaluate([{"id": "a", "label": 0, "score": 0}], float("nan"))
