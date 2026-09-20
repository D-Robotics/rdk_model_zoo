# SPDX-License-Identifier: Apache-2.0
"""Numerical evidence must identify the measured artifact even on a host."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'rdk-model-zoo-validate'
SPEC = importlib.util.spec_from_file_location('evidence_scope_validator', ROOT / 'scripts/validate_evidence.py')
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class EvidenceScopeTests(unittest.TestCase):
    def receipt(self, purpose='accuracy'):
        receipt = json.loads((ROOT / 'assets/verification.template.json').read_text(encoding='utf-8'))
        receipt['target'].update(commit='a' * 40, dirty=False)
        check = receipt['checks'][0]
        check.update(level='host', purpose=purpose, status='passed', reason=None)
        check['environment']['host'] = 'synthetic-host'
        check['execution'] = {
            'argv': ['synthetic-check'], 'cwd': '/synthetic', 'exit_code': 0,
            'started_at': '2026-09-17T00:00:00Z', 'ended_at': '2026-09-17T00:00:01Z',
        }
        check['result'] = {'summary': 'Synthetic test record, not actual measurement', 'acceptance': 'absolute error <= 0.01'}
        check['evidence'] = [{'path': 'synthetic.log', 'sha256': 'b' * 64}]
        return receipt

    def test_host_numeric_checks_require_scope_identity(self):
        for purpose in ('accuracy', 'consistency', 'performance'):
            with self.subTest(purpose=purpose):
                errors = VALIDATOR.validate(self.receipt(purpose))
                for field in ('platform', 'model_variant', 'task', 'runtime', 'model_sha256', 'input_sha256'):
                    self.assertTrue(any('scope.' + field in error for error in errors), (field, errors))

    def test_identified_host_numeric_record_is_structurally_valid(self):
        receipt = self.receipt()
        receipt['checks'][0]['scope'].update(
            platform='host-cpu', model_variant='synthetic-fp32', task='classification',
            runtime='synthetic-reference', model_sha256='c' * 64, input_sha256='d' * 64,
        )
        self.assertEqual(VALIDATOR.validate(receipt), [])

    def test_static_structure_check_does_not_require_model(self):
        receipt = self.receipt('structure')
        receipt['checks'][0]['level'] = 'static'
        self.assertEqual(VALIDATOR.validate(receipt), [])


if __name__ == '__main__':
    unittest.main()
