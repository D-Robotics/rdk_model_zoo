"""Execute real shell entry points using this test's Python interpreter.

The historical platforms/ shell trees were removed with the migration
closeout (2026-10-01); their transitional entries are reachable through
the pinned commit recorded in docs/migration/2026-09-30-model-examples.md.
"""
from pathlib import Path
import os, re, shlex, shutil, subprocess, sys, tempfile, unittest

S = Path(__file__).resolve().parents[1]
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
            for script in S.rglob('*.sh'):
                result = subprocess.run([BASH, '-n', str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = run(S / 'runtime/python/run.sh', 'cls', '--platform', 's100', '--dry-run')
            self.assertIn('[dry-run]', result)


if __name__ == '__main__':
    unittest.main()
