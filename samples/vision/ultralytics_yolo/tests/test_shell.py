"""Execute real shell entry points using this test's Python interpreter."""
from pathlib import Path
import os, re, shlex, shutil, subprocess, sys, tempfile, unittest

S = Path(__file__).resolve().parents[1]
R = S.parents[2]
BASH = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')


@unittest.skipUnless(BASH and Path(BASH).exists(), 'Bash required')
class ShellEntrypoints(unittest.TestCase):
    def test_syntax_and_public_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / 'python3'
            launcher.write_text('#!/bin/sh\nexec ' + shlex.quote(Path(sys.executable).as_posix()) + ' "$@"\n', encoding='utf-8')
            launcher.chmod(0o755)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ['PATH'])
            def run(path, *args):
                result = subprocess.run([BASH, str(path), *args], env=env, cwd=directory,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout
            trees = [S] + [R / f'platforms/{t}/samples/vision/ultralytics_yolo' for t in ('x5', 's')]
            for tree in trees:
                for script in tree.rglob('*.sh'):
                    result = subprocess.run([BASH, '-n', str(script)], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
            x5, s = trees[1:]
            result = run(x5 / 'model/fulldownload.sh', '--dry-run')
            names = set(re.findall(r'\b[yY][oO][lL][oO][^/\s]*\.bin', result))
            self.assertEqual(len(names), 67)
            result = run(s / 'model/download_model.sh', 's600', 'yolov8', 'cls', 'n', '--dry-run')
            self.assertIn('yolov8n_cls_nashp_224x224_nv12.hbm', result)
            for tree, platform in [(S, 's100'), (x5, 'x5'), (s, 's600')]:
                result = run(tree / 'runtime/python/run.sh', 'cls', '--platform', platform, '--dry-run')
                self.assertIn('[dry-run]', result)


if __name__ == '__main__':
    unittest.main()
