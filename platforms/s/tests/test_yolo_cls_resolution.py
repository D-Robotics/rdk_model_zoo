"""Check public model resolution selection without board runtime dependencies."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'samples/vision/ultralytics_yolo'
BASH = shutil.which('bash') or ('C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else None)
if os.name == 'nt' and Path('C:/Program Files/Git/bin/bash.exe').exists():
    BASH = 'C:/Program Files/Git/bin/bash.exe'

class ClassificationResolution(unittest.TestCase):
    def test_python_defaults_match_platform(self):
        tree = ast.parse((SAMPLE/'runtime/python/main.py').read_text(encoding='utf-8-sig'))
        names = {'MODEL_FILE_PATTERNS', 'DOWNLOAD_URL_BASE'}
        nodes = [n for n in tree.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)) or (isinstance(n, ast.FunctionDef) and n.name in {'get_default_model_path','get_download_url'})]
        namespace = {'os':os}
        exec(compile(ast.Module(body=nodes,type_ignores=[]), '<model path functions>', 'exec'), namespace)
        for march,suffix,size in [('nash-e','nashe',224),('nash-m','nashm',224),('nash-p','nashp',224)]:
            for task in ['cls','detect','seg','pose']:
                expected = size if task=='cls' else 640
                filename=f'yolo11n_{task}_{suffix}_{expected}x{expected}_nv12.hbm'
                self.assertTrue(namespace['get_default_model_path'](task,march,suffix).endswith(filename))
                url=namespace['get_download_url'](task,march,suffix)
                self.assertTrue(url.endswith('/'+march+'/'+filename))
                self.assertIn('/rdk_s600/' if suffix=='nashp' else '/rdk_s100/',url)

    @unittest.skipUnless(BASH, 'bash required')
    def test_download_and_runtime_entry_points(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory)
            model=tmp/'model';runtime=tmp/'runtime/python';bin_dir=tmp/'bin';board=tmp/'board'
            for d in [model,runtime,bin_dir,board]: d.mkdir(parents=True)
            for source,target in [(SAMPLE/'model/download_model.sh',model/'download_model.sh'),(SAMPLE/'runtime/python/run.sh',runtime/'run.sh')]:
                text=source.read_text(encoding='utf-8-sig').replace('/sys/class/boardinfo',board.as_posix())
                target.write_text(text,encoding='utf-8')
            stubs={'wget': '#!/bin/bash\nprintf "MOCK_WGET %s\\n" "$*"\nwhile [[ $# -gt 0 ]]; do if [[ "$1" == "-O" ]]; then shift; touch "$1"; fi; shift; done\n',
                'pip3':'#!/bin/bash\ncase "$2" in numpy) echo "Version: 1.26.4";; opencv-python) echo "Version: 4.11.0.86";; scipy) echo "Version: 1.15.3";; *) exit 1;; esac\n',
                'python3':'#!/bin/bash\nprintf "MOCK_PYTHON %s\\n" "$*"\n'}
            for name,content in stubs.items():
                p=bin_dir/name;p.write_text(content,encoding='utf-8');p.chmod(0o755)
            def run(script,args):
                posix_bin=bin_dir.as_posix()
                if os.name == 'nt': posix_bin='/'+posix_bin[0].lower()+posix_bin[2:]
                cmd='export PATH="'+posix_bin+':$PATH"; bash "$@"'
                return subprocess.run([BASH,'-c',cmd,'test',str(script),*args],cwd=runtime,text=True,capture_output=True,check=True,timeout=30).stdout
            for soc,march,suffix,resolution in [('s100','nash-e','nashe',224),('s100p','nash-m','nashm',224),('s600','nash-p','nashp',224)]:
                (board/'soc_name').write_text(soc);(board/'board_type').write_text(soc)
                for family in ['yolov8','yolo11']:
                    for size in ['n','s','m','l','x']:
                        output=run(model/'download_model.sh',[soc,family,'cls',size])
                        self.assertIn(f'{family}{size}_cls_{suffix}_{resolution}x{resolution}_nv12.hbm',output)
                        self.assertIn('/rdk_s600/' if soc=='s600' else '/rdk_s100/',output)
                for task in ['cls','detect']:
                    output=run(runtime/'run.sh',[task])
                    expected=resolution if task=='cls' else 640
                    self.assertIn(f'yolo11n_{task}_{suffix}_{expected}x{expected}_nv12.hbm',output)
                    self.assertIn('MOCK_PYTHON',output)

if __name__ == '__main__': unittest.main()
