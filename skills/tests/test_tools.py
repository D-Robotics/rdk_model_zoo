# SPDX-License-Identifier: Apache-2.0
"""Contract tests; all repositories and model records below are synthetic fixtures."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PACK = Path(__file__).resolve().parents[1]
INSPECT = PACK / 'rdk-model-zoo-repo/scripts/inspect_repo.py'
CATALOG = PACK / 'rdk-model-zoo/scripts/read_catalog.py'
EVIDENCE = PACK / 'rdk-model-zoo-validate/scripts/validate_evidence.py'

class Workspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'repo with spaces'
        self.root.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.write('README.md', '# Synthetic test repository\n')
        self.git('add', 'README.md')
        self.git('commit', '-qm', 'fixture')
    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], check=True, text=True, capture_output=True).stdout.strip()
    def write(self, path, text):
        out = self.root / path
        out.parent.mkdir(parents=True, exist_ok=True)
        # Keep fixture bytes stable on Windows; evidence hashes describe bytes,
        # so implicit CRLF conversion would make the expected digest platform
        # dependent.
        out.write_text(text, encoding='utf-8', newline='\n')
        return out
    def invoke(self, script, *args, code=0):
        self.assertTrue(script.is_file(), f'implementation missing: {script.name}')
        p = subprocess.run([sys.executable, str(script), *map(str,args)], text=True, capture_output=True, timeout=20, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(p.returncode, code, p.stdout+p.stderr)
        try: return json.loads(p.stdout)
        except json.JSONDecodeError: self.fail('tool must emit structured JSON: '+p.stdout+p.stderr)

class InspectTests(Workspace):
    def test_clean_git_identity(self):
        x=self.invoke(INSPECT,'--repo',self.root)
        self.assertEqual(x['head'],self.git('rev-parse','HEAD')); self.assertFalse(x['dirty']); self.assertFalse(x['board_verified'])
    def test_dirty_without_executing_sample(self):
        self.write('samples/vision/demo/main.py','raise RuntimeError("never import me")\n')
        x=self.invoke(INSPECT,'--repo',self.root,'--sample','samples/vision/demo')
        self.assertTrue(x['dirty']); self.assertIn('samples/vision/demo/main.py', x['sample']['files'])
    def test_x3_legacy_candidates(self):
        self.git('checkout','-qb','rdk_x3');self.write('demos/vision/README.md','fixture');self.write('release/models.yaml','schema_version: 1\nmodels: []\n')
        x=self.invoke(INSPECT,'--repo',self.root)
        self.assertIn('demos',x['layout_roots']);self.assertIn('release/models.yaml',x['manifest_candidates'])

    def test_current_manifest_candidate_does_not_infer_target(self):
        self.git('checkout','-qb','feature/target');
        self.write('docs/manifests/models.yaml','''schema_version: 1
release:
  platform: x5
  branch: rdk_x5
models: []
''')
        x=self.invoke(INSPECT,'--repo',self.root)
        self.assertIn('docs/manifests/models.yaml',x['manifest_candidates'])
        self.assertEqual(x['branch_role'],'unknown')
        self.assertFalse(x['is_maintenance_branch'])

    def test_maintenance_branch_is_distinguished_from_target(self):
        self.git('checkout','-qb','rdk_x5')
        x=self.invoke(INSPECT,'--repo',self.root)
        self.assertTrue(x['is_maintenance_branch'])
        self.assertEqual(x['maintenance_branch'],'rdk_x5')
        self.assertEqual(x['branch_role'],'maintenance')
        # Keep the historical branch hint while exposing its maintenance role.
        self.assertEqual(x['platform_hint'],'x5')
    def test_detached_head_is_not_x5(self):
        self.git('checkout','--detach','-q');x=self.invoke(INSPECT,'--repo',self.root)
        self.assertIsNone(x['branch']);self.assertIsNone(x['platform_hint'])
    def test_remote_secrets_redacted(self):
        self.git('remote','add','origin','https://user:secret123@github.com/D-Robotics/rdk_model_zoo.git?token=secret123')
        x=self.invoke(INSPECT,'--repo',self.root)
        self.assertNotIn('secret123', json.dumps(x));self.assertIn('github.com',json.dumps(x['remotes']))
    def test_traversal_rejected(self):
        x=self.invoke(INSPECT,'--repo',self.root,'--sample','../outside',code=2);self.assertFalse(x['ok'])
    def test_symlink_escape_rejected(self):
        outside=Path(self.tmp.name)/'outside';outside.mkdir();(self.root/'escape').symlink_to(outside,target_is_directory=True)
        x=self.invoke(INSPECT,'--repo',self.root,'--sample','escape',code=2);self.assertFalse(x['ok'])
    def test_non_git_is_explicit(self):
        p=Path(self.tmp.name)/'not-git';p.mkdir();x=self.invoke(INSPECT,'--repo',p,code=2);self.assertFalse(x['ok'])

class CatalogTests(Workspace):
    def model(self, path='docs/release/models.yaml'):
        return self.write(path,'''schema_version: 1
release: {platform: x5, tag: fixture-x5-v1.0.0}
models:
  - id: fixture-demo
    name: Fixture Demo
    sample_path: samples/vision/demo
    tasks: [object-detection]
    availability: manual
    assets:
      - {filename: fixture.bin, format: bin, sha256: null}
''')

    def test_current_manifest_layout(self):
        self.model('docs/manifests/models.yaml')
        self.write('docs/manifests/benchmarks.yaml','''schema_version: 1
release: {platform: x5, tag: fixture-x5-v1.0.0}
benchmarks: []
''')
        x=self.invoke(CATALOG,'--repo',self.root,'--model','fixture-demo')
        self.assertEqual(x['model_manifest']['path'],'docs/manifests/models.yaml')
        self.assertEqual(x['benchmark_manifest']['path'],'docs/manifests/benchmarks.yaml')
    def test_preserves_null_and_manual(self):
        self.model();x=self.invoke(CATALOG,'--repo',self.root,'--model','fixture-demo')
        self.assertIsNone(x['models'][0]['assets'][0]['sha256']);self.assertEqual(x['models'][0]['availability'],'manual');self.assertFalse(x['board_verified'])
    def test_legacy_manifest(self):
        self.model('release/models.yaml');x=self.invoke(CATALOG,'--repo',self.root)
        self.assertEqual(x['model_manifest']['path'],'release/models.yaml')
    def test_no_match_is_not_unsupported_platform(self):
        self.model();x=self.invoke(CATALOG,'--repo',self.root,'--model','missing')
        self.assertEqual(x['models'],[]);self.assertTrue(x['ok'])
    def test_benchmark_raw_qualifier_and_source(self):
        self.model();self.write('docs/release/benchmarks.yaml','''schema_version: 1
release: {platform: x5, tag: fixture-x5-v1.0.0}
benchmarks:
  - id: fixture-bench
    sample_id: fixture-demo
    performance: [{metric: throughput, value: 200, unit: fps, qualifier: lower-bound}]
    source: {ref: fixture-ref, path: docs/result.md}
''')
        x=self.invoke(CATALOG,'--repo',self.root)
        self.assertEqual(x['benchmarks'][0]['performance'][0]['qualifier'],'lower-bound');self.assertEqual(x['benchmarks'][0]['source']['ref'],'fixture-ref')
    def test_two_manifests_require_selection(self):
        self.model();self.model('release/models.yaml')
        x=self.invoke(CATALOG,'--repo',self.root,code=2);self.assertIn('ambiguous',x['reason'])
        y=self.invoke(CATALOG,'--repo',self.root,'--manifest','release/models.yaml');self.assertTrue(y['ok'])

    def test_all_supported_manifest_layouts_require_selection(self):
        self.model('docs/manifests/models.yaml')
        self.model('docs/release/models.yaml')
        self.model('release/models.yaml')
        x=self.invoke(CATALOG,'--repo',self.root,code=2)
        self.assertIn('ambiguous',x['reason'])
        self.assertIn('--manifest',x['reason'])

    def test_redefined_yaml_anchors_are_read_without_inference(self):
        self.model('docs/manifests/models.yaml')
        self.write('docs/manifests/benchmarks.yaml','''schema_version: 1
release: {platform: x5, tag: fixture-x5-v1.0.0}
benchmarks:
  - id: first
    sample_id: fixture-demo
    environment: &scope {hardware: RDK X5}
    source: &source {ref: fixture-ref, path: docs/result.md}
  - id: before-rebind
    sample_id: fixture-demo
    environment: *scope
    source: *source
  - id: second
    sample_id: fixture-demo
    environment: &scope {hardware: RDK S100}
    source: *source
  - id: third
    sample_id: fixture-demo
    environment: *scope
    source: *source
''')
        x=self.invoke(CATALOG,'--repo',self.root)
        self.assertEqual([row['id'] for row in x['benchmarks']],['first','before-rebind','second','third'])
        self.assertEqual(x['benchmarks'][1]['environment']['hardware'],'RDK X5')
        self.assertEqual(x['benchmarks'][3]['environment']['hardware'],'RDK S100')

    def test_s_release_keeps_explicit_subhardware_scope(self):
        self.write('docs/manifests/models.yaml','''schema_version: 1
release:
  platform: s
  tag: fixture-s-v1.0.0
  compatibility: {hardware: RDK S100/S100P/S600}
models:
  - id: fixture-s
    name: Fixture S
''')
        self.write('docs/manifests/benchmarks.yaml','''schema_version: 1
release: {platform: s, tag: fixture-s-v1.0.0}
benchmarks:
  - id: fixture-s100
    sample_id: fixture-s
    environment: {hardware: RDK S100}
''')
        x=self.invoke(CATALOG,'--repo',self.root)
        self.assertEqual(x['benchmarks'][0]['environment']['hardware'],'RDK S100')
        self.assertFalse(x['board_verified'])
        self.assertTrue(any('S release group' in warning for warning in x['warnings']))
    def test_missing_manifest_not_fabricated(self):
        x=self.invoke(CATALOG,'--repo',self.root,code=2);self.assertEqual(x['reason'],'manifest-not-found')
    def test_yaml_object_execution_rejected(self):
        self.write('docs/release/models.yaml','!!python/object/apply:os.system ["echo unsafe"]')
        x=self.invoke(CATALOG,'--repo',self.root,code=2);self.assertFalse(x['ok'])
    def test_duplicate_model_ids_rejected(self):
        self.write('docs/release/models.yaml','schema_version: 1\nmodels: [{id: same}, {id: same}]\n')
        x=self.invoke(CATALOG,'--repo',self.root,code=2);self.assertIn('duplicate',x['reason'])
    def test_unknown_schema_rejected(self):
        self.write('docs/release/models.yaml','schema_version: 99\nmodels: []\n')
        x=self.invoke(CATALOG,'--repo',self.root,code=2);self.assertIn('schema',x['reason'])
    def test_symlink_manifest_rejected(self):
        outside=Path(self.tmp.name)/'outside.yaml';outside.write_text('schema_version: 1\nmodels: []\n')
        p=self.root/'docs/release';p.mkdir(parents=True);(p/'models.yaml').symlink_to(outside)
        x=self.invoke(CATALOG,'--repo',self.root,code=2);self.assertFalse(x['ok'])
    def test_cyclic_yaml_rejected(self):
        self.write('docs/release/models.yaml','schema_version: 1\nmodels: &m [*m]\n')
        x=self.invoke(CATALOG,'--repo',self.root,code=2);self.assertFalse(x['ok'])

class EvidenceTests(Workspace):
    def receipt(self):
        return {'schema_version':1,'kind':'verification','target':{'repository':'D-Robotics/rdk_model_zoo','commit':self.git('rev-parse','HEAD'),'dirty':False,'patch_sha256':None,'sample_path':'samples/vision/demo'},'checks':[{'id':'fixture-check','required':True,'level':'host','purpose':'smoke','status':'not-run','scope':{'platform':None,'model_variant':None,'task':None,'runtime':None,'model_sha256':None,'input_sha256':None},'environment':{'host':None,'board':None,'os':None,'runtime_version':None},'execution':None,'result':None,'evidence':[],'reason':'No execution in this fixture.'}]}
    def run_receipt(self,r,*args,code=0):
        path=self.write('receipt.json',json.dumps(r));return self.invoke(EVIDENCE,path,*args,code=code)
    def passed(self):
        r=self.receipt();c=r['checks'][0];c.update(status='passed',reason=None,environment={'host':'synthetic-host','board':None,'os':'fixture-os','runtime_version':None},execution={'argv':['python3','fixture.py'],'cwd':'/fixture','exit_code':0,'started_at':'2026-09-10T00:00:00Z','ended_at':'2026-09-10T00:00:01Z'},result={'summary':'synthetic fixture success','acceptance':'exit zero'},evidence=[{'path':'logs/fixture.log','sha256':hashlib.sha256(b'fixture\n').hexdigest()}]);return r
    def test_not_run_valid_without_invented_evidence(self):
        x=self.run_receipt(self.receipt());self.assertTrue(x['valid']);self.assertFalse(x['board_verified'])
    def test_empty_checks_rejected(self):
        r=self.receipt();r['checks']=[];x=self.run_receipt(r,code=2);self.assertFalse(x['valid'])
    def test_pass_without_log_rejected(self):
        r=self.passed();r['checks'][0]['evidence']=[];x=self.run_receipt(r,code=2);self.assertFalse(x['valid'])
    def test_pass_nonzero_exit_rejected(self):
        r=self.passed();r['checks'][0]['execution']['exit_code']=1;self.run_receipt(r,code=2)
    def test_board_pass_missing_board_identity_rejected(self):
        r=self.passed();r['checks'][0]['level']='board';self.run_receipt(r,code=2)
    def test_dirty_pass_requires_patch_digest(self):
        r=self.passed();r['target']['dirty']=True;self.run_receipt(r,code=2)
    def test_hash_checked_when_root_supplied(self):
        r=self.passed();self.write('logs/fixture.log','fixture\n');x=self.run_receipt(r,'--evidence-root',self.root)
        self.assertTrue(x['evidence_hashes_checked']);self.assertFalse(x['board_verified'])
    def test_hash_mismatch_rejected(self):
        r=self.passed();self.write('logs/fixture.log','tampered\n');self.run_receipt(r,'--evidence-root',self.root,code=2)
    def test_log_path_escape_rejected(self):
        r=self.passed();r['checks'][0]['evidence'][0]['path']='../secret';self.run_receipt(r,code=2)
    def test_duplicate_check_ids_rejected(self):
        r=self.receipt();r['checks'].append(dict(r['checks'][0]));self.run_receipt(r,code=2)
    def test_not_run_cannot_contain_execution(self):
        r=self.passed();r['checks'][0]['status']='not-run';r['checks'][0]['reason']='missing board';self.run_receipt(r,code=2)
    def test_accuracy_needs_acceptance(self):
        r=self.passed();r['checks'][0]['purpose']='accuracy';r['checks'][0]['result']['acceptance']=None;self.run_receipt(r,code=2)
    def test_pass_is_not_runtime_certification(self):
        r=self.passed();x=self.run_receipt(r);self.assertFalse(x['evidence_hashes_checked']);self.assertFalse(x['board_verified'])

if __name__ == '__main__': unittest.main()
