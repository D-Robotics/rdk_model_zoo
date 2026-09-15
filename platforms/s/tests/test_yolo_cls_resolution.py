"""Check public S-series classification download and runtime paths."""
from pathlib import Path
import sys, subprocess, tempfile, unittest
SAMPLE=Path(__file__).resolve().parents[1]/'samples/vision/ultralytics_yolo'
ROOT=Path(__file__).resolve().parents[3]
class ClassificationResolution(unittest.TestCase):
    def test_python_defaults_match_platform(self):
        for platform,suffix in [('s100','nashe'),('s100p','nashm'),('s600','nashp')]:
            for task in ['cls','detect','seg','pose']:
                r=subprocess.run([sys.executable,str(SAMPLE/'runtime/python/main.py'),'--platform',platform,'--task',task,'--dry-run'],cwd=tempfile.gettempdir(),capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
                size=224 if task=='cls' else 640
                self.assertIn(f'yolo11n_{task}_{suffix}_{size}x{size}_nv12.hbm',r.stdout)
    def test_all_classifier_download_names(self):
        p=ROOT/'samples/vision/ultralytics_yolo/runtime/python'
        sys.path.insert(0,str(p))
        from yolo_assets import model_url
        from yolo_platform import resolve_platform
        for platform,suffix in [('s100','nashe'),('s100p','nashm'),('s600','nashp')]:
            for family in ['yolov8','yolo11']:
                for size in ['n','s','m','l','x']:
                    url=model_url(resolve_platform(platform),family,'cls',size)
                    self.assertTrue(url.endswith(f'{family}{size}_cls_{suffix}_224x224_nv12.hbm'))
                    self.assertIn('/rdk_s600/' if platform=='s600' else '/rdk_s100/',url)
if __name__=='__main__':unittest.main()
