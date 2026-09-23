"""SDK-free CLI, download delegation, and explicit prompt rejection tests."""
import contextlib, io, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from samples.vision.yoloworld.runtime.python import main
from samples.vision.yoloworld.model import download
SAMPLE=Path(__file__).resolve().parents[1]
class EntryTests(unittest.TestCase):
    def test_help_list_and_explicit_dry_run_do_not_load_sdk(self):
        for args in (('--help',),('--list-models',),('--dry-run','--target','x5')):
            p=subprocess.run([sys.executable,str(SAMPLE/'runtime/python/main.py'),*args],cwd='/tmp',capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
        p=subprocess.run([sys.executable,str(SAMPLE/'runtime/python/main.py'),'--dry-run'],cwd='/tmp',capture_output=True,text=True)
        self.assertEqual(p.returncode,2)
    def test_empty_and_overflow_prompt_rejected(self):
        self.assertEqual(main.main(['--dry-run','--target','x5','--prompts','dog,,cat']),2)
        self.assertEqual(main.main(['--dry-run','--target','x5','--prompts',','.join(['dog']*33)]),2)
    def test_download_uses_manifest_asset_without_network(self):
        with tempfile.TemporaryDirectory() as d, patch.object(download,'download_asset',return_value='d'*64) as fetch, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(download.main(['--target','x5','--output-dir',d]),0)
        self.assertEqual(fetch.call_args.args[0].reference,'x5:yoloworld:yolo_world.bin'); self.assertIn('Observed SHA-256: '+'d'*64,out.getvalue())
if __name__=='__main__': unittest.main()
