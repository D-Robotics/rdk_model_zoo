"""Tests for the sample-contract checker.

Positive and negative fixtures live in ``fixtures/``; every negative
fixture was constructed first and pins a rule ID plus a path:line, so the
checker cannot pass merely by running on well-formed input.

Run with:
  python3 -m unittest discover -s tools/sample_contract/tests -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES = TESTS_DIR / "fixtures"


def load_check():
    """Import check.py with proper sys.modules registration."""

    module_name = "sample_contract_check"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        module_name, TESTS_DIR.parent / "check.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CHECK = load_check()


def run_fixture(name: str, *extra: str):
    """Run check.main against one fixture; return (exit code, stdout)."""

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = CHECK.main(["--sample", str(FIXTURES / name), *extra])
    return code, stdout.getvalue()


def report_for(name: str):
    """Run run_sample directly; return the SampleReport."""

    templates = CHECK.load_template_anchors(
        CHECK.REPO_ROOT / "docs/sample-standards/templates")
    return CHECK.run_sample(FIXTURES / name, templates, "import")


def findings_for(report, rule: str):
    return [f for f in report.findings if f.rule == rule]


class GoodSampleTests(unittest.TestCase):
    def test_compliant_fixture_passes_with_exit_zero(self):
        code, output = run_fixture("good_sample")
        self.assertEqual(code, 0)
        self.assertIn("0 violations", output)

    def test_cli_defaults_actually_verified_in_import_mode(self):
        report = report_for("good_sample")
        cli_skips = [s for s in report.skips if s.rule == CHECK.RULE_CLI]
        self.assertEqual(cli_skips, [])
        self.assertEqual(findings_for(report, CHECK.RULE_CLI), [])

    def test_stage_purity_reports_policy_skip_visibly(self):
        report = report_for("good_sample")
        skips = [s for s in report.skips if s.rule == CHECK.RULE_PURITY]
        self.assertTrue(
            any(s.path.endswith("main.py") and "CLI layer" in s.reason
                for s in skips))
        self.assertEqual(findings_for(report, CHECK.RULE_PURITY), [])


class MissingSectionTests(unittest.TestCase):
    def test_missing_duplicate_and_order_findings_with_lines(self):
        report = report_for("bad_sections")
        findings = findings_for(report, CHECK.RULE_SECTIONS)
        messages = [f.message for f in findings]
        self.assertTrue(
            any("missing required section anchor 'quickstart'" in m
                and f.path.endswith("README.md")
                for f, m in zip(findings, messages)))
        duplicate = [f for f in findings if "duplicate anchor id" in f.message]
        self.assertEqual(len(duplicate), 1)
        self.assertIn("'overview'", duplicate[0].message)
        self.assertEqual(duplicate[0].line, 11)
        self.assertTrue(
            any("order deviates from template" in m and "support-matrix" in m
                for m in messages))

    def test_chinese_readme_stays_clean(self):
        report = report_for("bad_sections")
        self.assertEqual(
            [f for f in findings_for(report, CHECK.RULE_SECTIONS)
             if f.path.endswith("README_cn.md")], [])


class BrokenLinkTests(unittest.TestCase):
    def test_local_and_fragment_links_fail_with_lines(self):
        report = report_for("bad_links")
        findings = findings_for(report, CHECK.RULE_LINKS)
        self.assertEqual(len(findings), 3)
        by_line = {f.line: f.message for f in findings}
        self.assertIn("model/README.md", by_line[27])
        self.assertIn("test_data/missing.png", by_line[28])
        self.assertIn("#no-such-anchor", by_line[30])

    def test_external_links_are_out_of_scope(self):
        _, output = run_fixture("bad_links")
        self.assertNotIn("example.com", output)


class CliDriftTests(unittest.TestCase):
    def test_drift_missing_and_extra_options_reported_per_language(self):
        report = report_for("bad_cli_drift")
        findings = findings_for(report, CHECK.RULE_CLI)
        joined = " | ".join(f.message for f in findings)
        self.assertIn("[en] default drift for --top-k: parser '5' "
                      "vs README '3'", joined)
        self.assertIn("parser option --threshold (default 'null') "
                      "is not documented", joined)
        self.assertIn("documents option --extra-opt absent from the parser",
                      joined)
        self.assertNotIn("[zh]", joined)

    def test_drift_lines_point_at_table_rows(self):
        report = report_for("bad_cli_drift")
        findings = findings_for(report, CHECK.RULE_CLI)
        self.assertTrue(all(f.line >= 20 for f in findings))

    def test_static_mode_skips_cli_without_claiming_pass(self):
        code, output = run_fixture("bad_cli_drift", "--parser-mode", "static")
        self.assertEqual(code, 1)
        self.assertIn("static parser mode: build_parser not executed "
                      "(no pass implied)", output)
        self.assertNotIn("[en] default drift", output)


class StagePurityTests(unittest.TestCase):
    def test_download_save_subprocess_calls_flagged_with_lines(self):
        report = report_for("bad_stage_purity")
        findings = findings_for(report, CHECK.RULE_PURITY)
        by_line = {f.line: f.message for f in findings}
        self.assertEqual(sorted(by_line), [16, 21, 24, 28])
        self.assertIn("pre_process() calls open()", by_line[16])
        self.assertIn("file write/save", by_line[16])
        self.assertIn("forward() calls urllib.request.urlretrieve()",
                      by_line[21])
        self.assertIn("download/network", by_line[21])
        self.assertIn("forward() calls cv2.imwrite()", by_line[24])
        self.assertIn("post_process() calls subprocess.run()", by_line[28])
        self.assertIn("subprocess", by_line[28])

    def test_non_stage_helpers_are_not_flagged(self):
        report = report_for("bad_stage_purity")
        joined = " ".join(f.message for f in report.findings)
        self.assertNotIn("save_report", joined)


class I18nParamsTests(unittest.TestCase):
    def test_option_set_and_default_mismatch_between_languages(self):
        report = report_for("bad_i18n_params")
        findings = findings_for(report, CHECK.RULE_I18N)
        joined = " | ".join(f.message for f in findings)
        self.assertIn("option --top-k documented only in the English table",
                      joined)
        self.assertIn("option --topk documented only in the Chinese table",
                      joined)
        self.assertIn("default mismatch for --target: en 'auto' vs zh 'x5'",
                      joined)
        self.assertEqual(len(findings), 3)

    def test_missing_main_py_is_a_skip_not_a_pass(self):
        report = report_for("bad_i18n_params")
        skips = [s for s in report.skips if s.rule == CHECK.RULE_CLI]
        self.assertEqual(len(skips), 2)
        self.assertTrue(all("no runtime/python/main.py" in s.reason
                            for s in skips))


class ImportFailureTests(unittest.TestCase):
    def test_parser_import_failure_is_recorded_as_skip(self):
        report = report_for("bad_import_runtime")
        skips = [s for s in report.skips if s.rule == CHECK.RULE_CLI]
        self.assertEqual(len(skips), 2)
        self.assertTrue(all("parser import failed (RuntimeError: boom"
                            in s.reason for s in skips))
        self.assertEqual(findings_for(report, CHECK.RULE_CLI), [])


class PairingTests(unittest.TestCase):
    def test_missing_chinese_readme_reported_at_model_level(self):
        report = report_for("bad_pair")
        pair = findings_for(report, CHECK.RULE_PAIR)
        self.assertEqual(len(pair), 1)
        self.assertTrue(pair[0].path.endswith("bad_pair/model"))
        self.assertIn("README_cn.md missing", pair[0].message)


class ScopeTests(unittest.TestCase):
    def test_migration_scope_reads_refactor_column_of_progress_region(self):
        map_path = (CHECK.REPO_ROOT / "docs/releases/unified-migration"
                    / "x5-s-migration-map.md")
        paths, findings = CHECK.resolve_migration_scope(
            map_path, CHECK.REPO_ROOT / "samples")
        names = {path.name for path in paths}
        self.assertIn("resnet", names)
        self.assertIn("paddle_ocr", names)
        self.assertIn("ultralytics_yolo", names)
        self.assertEqual(findings, [])


class ExemptionTests(unittest.TestCase):
    def test_matching_exemption_suppresses_one_finding(self):
        target = CHECK.display_path(FIXTURES / "bad_links/README.md")
        with tempfile.TemporaryDirectory() as tmp:
            exempt = Path(tmp) / "exemptions.json"
            exempt.write_text(json.dumps({"exemptions": [{
                "rule": CHECK.RULE_LINKS, "path": target, "line": 27,
                "reason": "fixture link intentionally absent (test)",
            }]}), encoding="utf-8")
            code, output = run_fixture("bad_links", "--exemptions",
                                       str(exempt))
        self.assertEqual(code, 1)  # two real findings remain
        self.assertIn("1 exemptions applied", output)
        self.assertIn("fixture link intentionally absent", output)

    def test_unused_exemption_is_itself_a_violation(self):
        target = CHECK.display_path(FIXTURES / "bad_links/README.md")
        with tempfile.TemporaryDirectory() as tmp:
            exempt = Path(tmp) / "exemptions.json"
            exempt.write_text(json.dumps({"exemptions": [{
                "rule": CHECK.RULE_LINKS, "path": target, "line": 999,
                "reason": "stale entry that matches nothing",
            }]}), encoding="utf-8")
            code, output = run_fixture("bad_links", "--exemptions",
                                       str(exempt))
        self.assertEqual(code, 1)
        self.assertIn(CHECK.RULE_EXEMPTION, output)
        self.assertIn("unused exemption", output)

    def test_exemptions_without_reason_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exempt = Path(tmp) / "exemptions.json"
            exempt.write_text(json.dumps({"exemptions": [{
                "rule": CHECK.RULE_LINKS, "path": "x", "line": 1,
            }]}), encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                code = CHECK.main(["--sample", str(FIXTURES / "bad_links"),
                                   "--exemptions", str(exempt)])
        self.assertEqual(code, 2)
        self.assertIn("malformed exemption", stderr.getvalue())


class CanonicalizationTests(unittest.TestCase):
    def test_parser_default_forms(self):
        self.assertEqual(CHECK.canon_parser_default(None), "null")
        self.assertEqual(CHECK.canon_parser_default(True), "true")
        self.assertEqual(CHECK.canon_parser_default(False), "false")
        self.assertEqual(CHECK.canon_parser_default([0]), "[0]")
        self.assertEqual(CHECK.canon_parser_default(5), "5")
        self.assertEqual(CHECK.canon_parser_default("auto"), "auto")
        relative = (CHECK.REPO_ROOT / "samples" / "vision" / "resnet")
        self.assertEqual(
            CHECK.canon_parser_default(str(relative / "model" / "m.bin")),
            "samples/vision/resnet/model/m.bin")
        outside = "/definitely/not/in/the/repo/model.bin"
        self.assertEqual(CHECK.canon_parser_default(outside), outside)

    def test_readme_default_forms(self):
        self.assertEqual(CHECK.canon_readme_default("`null`"), "null")
        self.assertEqual(CHECK.canon_readme_default("NONE"), "null")
        self.assertEqual(CHECK.canon_readme_default("`5`"), "5")
        self.assertEqual(CHECK.canon_readme_default("[0]"), "[0]")


class ReportTests(unittest.TestCase):
    def test_json_report_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            code, _ = run_fixture("bad_links", "--report", str(report_path),
                                  "--format", "json")
            self.assertEqual(code, 1)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["tool"], "sample-contract")
        self.assertEqual(payload["summary"]["violations"], 3)
        self.assertEqual(payload["summary"]["samples"], 1)
        rules = {f["rule"]
                 for sample in payload["samples"]
                 for f in sample["findings"]}
        self.assertEqual(rules, {CHECK.RULE_LINKS})


if __name__ == "__main__":
    unittest.main()
