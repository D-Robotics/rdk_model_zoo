import unittest
from samples._shared.text_metrics import edit_counts, score_transcripts


class TextMetricTests(unittest.TestCase):
    def test_operations_and_unicode(self):
        self.assertEqual(
            edit_counts("今天", "明天"),
            dict(substitutions=1, deletions=0, insertions=0, distance=1),
        )
        self.assertEqual(edit_counts("abc", "ac")["deletions"], 1)
        self.assertEqual(edit_counts("ac", "abc")["insertions"], 1)
        self.assertEqual(edit_counts("", "ab")["distance"], 2)
        self.assertEqual(edit_counts("ab", "")["distance"], 2)

    def test_aggregate_cer_is_micro_not_mean(self):
        result = score_transcripts(
            [
                {"id": "a", "reference": "a", "hypothesis": "b"},
                {"id": "b", "reference": "abcd", "hypothesis": "abcd"},
            ]
        )
        self.assertEqual(result["cer"], 0.2)
        self.assertEqual(result["reference_characters"], 5)
        self.assertEqual(result["exact_matches"], 1)
        self.assertIsNone(
            score_transcripts([{"id": "a", "reference": "", "hypothesis": "x"}])["cer"]
        )

    def test_no_hidden_normalization_and_invalid_rows(self):
        self.assertEqual(edit_counts("a b", "ab")["distance"], 1)
        self.assertEqual(edit_counts("A", "a")["distance"], 1)
        for rows in (
            [],
            [{"id": "a", "reference": 3, "hypothesis": "x"}],
            [{"id": "a", "reference": "a", "hypothesis": "a"}] * 2,
        ):
            with self.assertRaises(ValueError):
                score_transcripts(rows)
