"""Target selection must not turn a preparation option into hardware evidence."""
import unittest
from unittest.mock import patch


class TargetSelectionTests(unittest.TestCase):
    def test_known_identity_and_s100p_refinement(self):
        from utils.py_utils.platforms import resolve_target
        cases = [('X5', None, 'x5'), ('s100', None, 's100'),
                 ('s100', 'RDK S100P', 's100p'), ('s600', None, 's600')]
        for soc, board, expected in cases:
            with self.subTest(soc=soc, board=board):
                self.assertEqual(resolve_target(soc_name=soc, board_type=board), expected)

    def test_unknown_soc_cannot_be_refined_into_known_board(self):
        from utils.py_utils.platforms import resolve_target
        for soc in ('s100-unknown', 'prototype', 'x3'):
            with self.subTest(soc=soc), self.assertRaises(ValueError):
                resolve_target(soc_name=soc, board_type='s100p')

    def test_group_is_not_an_execution_target(self):
        from utils.py_utils.platforms import resolve_target
        with self.assertRaises(ValueError):
            resolve_target('s')

    def test_explicit_preparation_does_not_read_host(self):
        from utils.py_utils.platforms import resolve_target
        with patch('utils.py_utils.platforms.detect_target', side_effect=AssertionError('host read')):
            self.assertEqual(resolve_target('s600'), 's600')

    def test_execution_requires_matching_actual_identity(self):
        from utils.py_utils.platforms import require_execution_target
        for actual in (None, 'x5', 's100p'):
            with self.subTest(actual=actual):
                with patch('utils.py_utils.platforms.detect_target', return_value=actual):
                    with self.assertRaises(ValueError):
                        require_execution_target('s100')
        with patch('utils.py_utils.platforms.detect_target', return_value='s100'):
            self.assertEqual(require_execution_target('s100'), 's100')
            self.assertEqual(require_execution_target('auto'), 's100')

    def test_missing_board_info_is_unknown(self):
        from utils.py_utils.platforms import detect_target
        with patch('pathlib.Path.read_text', side_effect=FileNotFoundError):
            self.assertIsNone(detect_target())

    def test_observed_x5_device_tree_is_fallback_only(self):
        from utils.py_utils.platforms import detect_target
        def identity(path):
            return 'D-Robotics RDK X5 V1.0' if str(path).replace('\\', '/') == '/proc/device-tree/model' else None
        with patch('utils.py_utils.platforms._read_identity', side_effect=identity):
            self.assertEqual(detect_target(), 'x5')
        def conflict(path):
            return 'unknown' if str(path).endswith('soc_name') else identity(path)
        with patch('utils.py_utils.platforms._read_identity', side_effect=conflict):
            self.assertIsNone(detect_target())

    def test_observed_socinfo_x5u_is_exact_alias(self):
        from utils.py_utils.platforms import detect_target
        for alias in ('X5U', 'X5H', 'X5M'):
            def identity(path):
                return alias if str(path).replace('\\', '/') == '/sys/class/socinfo/soc_name' else None
            with self.subTest(alias=alias), patch('utils.py_utils.platforms._read_identity', side_effect=identity):
                self.assertEqual(detect_target(), 'x5')
        with patch('utils.py_utils.platforms._read_identity', side_effect=lambda path: 'X5U-unknown' if str(path).replace('\\', '/') == '/sys/class/socinfo/soc_name' else None):
            self.assertIsNone(detect_target())

    def test_null_sysfs_identity_falls_back_to_s100_v1p1_device_tree(self):
        from pathlib import Path
        import tempfile
        from utils.py_utils import platforms

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            soc = root / 'soc_name'
            board = root / 'board_type'
            socinfo = root / 'socinfo'
            tree = root / 'model'
            soc.write_text('(null)\n', encoding='utf-8')
            board.write_text('(null)\n', encoding='utf-8')
            socinfo.write_text('', encoding='utf-8')
            tree.write_text('D-Robotics RDK S100 V1P1\x00', encoding='utf-8')

            with patch.multiple(
                platforms,
                SOC_NAME_PATH=soc,
                BOARD_TYPE_PATH=board,
                SOCINFO_NAME_PATH=socinfo,
                DEVICE_TREE_MODEL_PATH=tree,
            ):
                self.assertEqual(platforms.detect_target(), 's100')

                # Non-null unknown SoC must still fail closed rather than falling back
                soc.write_text('unknown\n', encoding='utf-8')
                self.assertIsNone(platforms.detect_target())

                # (null) without a recognized device-tree model remains unknown
                soc.write_text('(null)\n', encoding='utf-8')
                tree.write_text('Unknown Board\x00', encoding='utf-8')
                self.assertIsNone(platforms.detect_target())

    def test_inspect_soc_name_handles_null_sysfs_via_device_tree_fallback(self):
        from utils.py_utils import inspect as rdk_inspect
        from unittest.mock import mock_open

        with patch('builtins.open', mock_open(read_data='(null)\n')), \
             patch('utils.py_utils.platforms.detect_target', return_value='s100'):
            self.assertEqual(rdk_inspect.get_soc_name_fallback_free(), 's100')
            self.assertEqual(rdk_inspect.get_soc_name(), 's100')
        with patch('builtins.open', mock_open(read_data='(null)\n')), \
             patch('utils.py_utils.platforms.detect_target', return_value=None):
            self.assertEqual(rdk_inspect.get_soc_name_fallback_free(), '')
            self.assertEqual(rdk_inspect.get_soc_name(), 's100')


if __name__ == '__main__':
    unittest.main()
