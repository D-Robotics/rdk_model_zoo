"""Target selection must not turn a preparation option into hardware evidence."""
import unittest
from unittest.mock import patch


class TargetSelectionTests(unittest.TestCase):
    def test_known_identity_and_s100p_refinement(self):
        from samples._shared.platforms import resolve_target
        cases = [('X5', None, 'x5'), ('s100', None, 's100'),
                 ('s100', 'RDK S100P', 's100p'), ('s600', None, 's600')]
        for soc, board, expected in cases:
            with self.subTest(soc=soc, board=board):
                self.assertEqual(resolve_target(soc_name=soc, board_type=board), expected)

    def test_unknown_soc_cannot_be_refined_into_known_board(self):
        from samples._shared.platforms import resolve_target
        for soc in ('s100-unknown', 'prototype', 'x3'):
            with self.subTest(soc=soc), self.assertRaises(ValueError):
                resolve_target(soc_name=soc, board_type='s100p')

    def test_group_is_not_an_execution_target(self):
        from samples._shared.platforms import resolve_target
        with self.assertRaises(ValueError):
            resolve_target('s')

    def test_explicit_preparation_does_not_read_host(self):
        from samples._shared.platforms import resolve_target
        with patch('samples._shared.platforms.detect_target', side_effect=AssertionError('host read')):
            self.assertEqual(resolve_target('s600'), 's600')

    def test_execution_requires_matching_actual_identity(self):
        from samples._shared.platforms import require_execution_target
        for actual in (None, 'x5', 's100p'):
            with self.subTest(actual=actual):
                with patch('samples._shared.platforms.detect_target', return_value=actual):
                    with self.assertRaises(ValueError):
                        require_execution_target('s100')
        with patch('samples._shared.platforms.detect_target', return_value='s100'):
            self.assertEqual(require_execution_target('s100'), 's100')
            self.assertEqual(require_execution_target('auto'), 's100')

    def test_missing_board_info_is_unknown(self):
        from samples._shared.platforms import detect_target
        with patch('pathlib.Path.read_text', side_effect=FileNotFoundError):
            self.assertIsNone(detect_target())

    def test_observed_x5_device_tree_is_fallback_only(self):
        from samples._shared.platforms import detect_target
        def identity(path):
            return 'D-Robotics RDK X5 V1.0' if str(path).replace('\\', '/') == '/proc/device-tree/model' else None
        with patch('samples._shared.platforms._read_identity', side_effect=identity):
            self.assertEqual(detect_target(), 'x5')
        def conflict(path):
            return 'unknown' if str(path).endswith('soc_name') else identity(path)
        with patch('samples._shared.platforms._read_identity', side_effect=conflict):
            self.assertIsNone(detect_target())

    def test_observed_socinfo_x5u_is_exact_alias(self):
        from samples._shared.platforms import detect_target
        for alias in ('X5U', 'X5H', 'X5M'):
            def identity(path):
                return alias if str(path).replace('\\', '/') == '/sys/class/socinfo/soc_name' else None
            with self.subTest(alias=alias), patch('samples._shared.platforms._read_identity', side_effect=identity):
                self.assertEqual(detect_target(), 'x5')
        with patch('samples._shared.platforms._read_identity', side_effect=lambda path: 'X5U-unknown' if str(path).replace('\\', '/') == '/sys/class/socinfo/soc_name' else None):
            self.assertIsNone(detect_target())


if __name__ == '__main__':
    unittest.main()
