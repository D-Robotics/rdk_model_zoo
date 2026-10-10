# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the bundled-audio prefix and decoder protocol checks."""
import unittest
from samples.speech.asr.evaluator.basic import evaluate_report, PUBLISHED_PREFIX


class BasicEvaluatorTests(unittest.TestCase):
    def report(self, text, decode_mode="legacy"):
        return {"schema": "rdk-model-zoo/asr-run/v1", "status": "completed",
                "audio_sha256": "audio", "decode_mode": decode_mode,
                "text": text, "chunks": [{"text": text}]}

    def test_reference_prefix_checked_independently(self):
        self.assertTrue(evaluate_report(self.report(PUBLISHED_PREFIX + "语言模型"), "audio")["passed"])
        self.assertFalse(evaluate_report(self.report("无法识别"), "audio")["passed"])

    def test_wrong_audio_or_ctc_literal_delimiter_fails(self):
        self.assertFalse(evaluate_report(self.report(PUBLISHED_PREFIX), "other")["passed"])
        # | is only a decoding defect under ctc; legacy keeps it verbatim.
        self.assertFalse(evaluate_report(self.report(PUBLISHED_PREFIX + "|", "ctc"), "audio")["passed"])
        self.assertTrue(evaluate_report(self.report(PUBLISHED_PREFIX + "|"), "audio")["passed"])

    def test_prefix_does_not_claim_remaining_characters_verified(self):
        result = evaluate_report(self.report(PUBLISHED_PREFIX + "剩余未知"), "audio")
        self.assertTrue(result["passed"])
        self.assertIn("remaining characters unverified", result["scope"])

    def test_run_and_chunks_must_agree(self):
        report = self.report(PUBLISHED_PREFIX)
        report["chunks"] = [{"text": "different"}]
        self.assertFalse(evaluate_report(report, "audio")["passed"])


if __name__ == "__main__":
    unittest.main()
