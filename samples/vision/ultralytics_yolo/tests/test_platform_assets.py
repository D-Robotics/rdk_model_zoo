"""Owner acceptance checks independent of implementation worker tests."""
from pathlib import Path
import sys
import unittest
import json

ROOT=Path(__file__).resolve().parents[4]

class PlatformAcceptance(unittest.TestCase):
    def test_conflicting_input_override_is_rejected(self):
        from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
        from samples.vision.ultralytics_yolo.runtime.python.backend import Nv12InputAdapter,UnsupportedInputError
        with self.assertRaises(UnsupportedInputError):
            Nv12InputAdapter.from_metadata(resolve_platform('s100'),'m',['y','uv'],{'y':(1,640,640,1),'uv':(1,320,320,2)},(224,224))

    def test_uv_layout_is_checked_not_only_element_count(self):
        from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
        from samples.vision.ultralytics_yolo.runtime.python.backend import Nv12InputAdapter,UnsupportedInputError
        with self.assertRaises(UnsupportedInputError):
            Nv12InputAdapter.from_metadata(resolve_platform('s100'),'m',['y','uv'],{'y':(1,640,640,1),'uv':(1,640,320,1)})

    def test_odd_input_geometry_is_rejected(self):
        from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
        from samples.vision.ultralytics_yolo.runtime.python.backend import Nv12InputAdapter,UnsupportedInputError
        with self.assertRaises(UnsupportedInputError):
            Nv12InputAdapter.from_metadata(resolve_platform('x5'),'m',['in'],{'in':(1,3,641,641)})

    def test_x5_v10_keeps_dfl_and_s_keeps_nms_free(self):
        from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
        from samples.vision.ultralytics_yolo.runtime.python.cli import is_nms_free
        self.assertFalse(is_nms_free(resolve_platform('x5'),'yolov10'))
        self.assertTrue(is_nms_free(resolve_platform('s600'),'yolov10'))

    def test_all_advertised_assets_exist_in_manifest_snapshot(self):
        from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
        from samples.vision.ultralytics_yolo.runtime.python.cli import family_registry,model_url,UnsupportedAssetError
        import yaml
        urls=set()
        for manifest in ('x5','s'):
            data=yaml.safe_load((ROOT/f'docs/release/{manifest}/models.yaml').read_text(encoding='utf-8'))
            urls|={a['url'] for m in data['models'] for a in m.get('assets',[]) if a.get('url')}
        for key in ['x5','s100','s100p','s600']:
            profile=resolve_platform(key)
            for family,spec in family_registry(profile).items():
                for task in spec.tasks:
                    for size in spec.task_sizes.get(task,spec.sizes):
                        try:url=model_url(profile,family,task,size)
                        except UnsupportedAssetError:continue
                        with self.subTest(platform=key,family=family,task=task,size=size):
                            self.assertTrue(url in urls, url)

    def test_s100p_board_with_base_soc(self):
        from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
        self.assertEqual(resolve_platform(soc_name='s100',board_type='s100p').key,'s100p')

    def test_explicit_platform_beats_board(self):
        from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
        self.assertEqual(resolve_platform('s100',soc_name='s100',board_type='s100p').key,'s100')

    def test_unknown_soc_is_not_a_p_variant(self):
        from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform,UnsupportedPlatformError
        with self.assertRaises(UnsupportedPlatformError):
            resolve_platform(soc_name='s100-unknown',board_type='prototype')

if __name__=='__main__':unittest.main()
