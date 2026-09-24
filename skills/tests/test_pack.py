# SPDX-License-Identifier: Apache-2.0
"""Structural tests and generator isolation; these are not Agent behavior tests."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
VALIDATOR=ROOT/'tools/validate_pack.py'
SYNC=ROOT/'tools/sync_references.py'
class PackTests(unittest.TestCase):
    def run_tool(self,script,*args,code=0):
        self.assertTrue(script.is_file(),'implementation missing: '+script.name)
        p=subprocess.run([sys.executable,str(script),*map(str,args)],capture_output=True,text=True,encoding='utf-8',timeout=20)
        self.assertEqual(p.returncode,code,p.stdout+p.stderr)
        return json.loads(p.stdout)
    def test_all_seven_skills_validate(self):
        x=self.run_tool(VALIDATOR,'--pack-root',ROOT)
        self.assertEqual(x['skills_checked'],7);self.assertTrue(x['valid'])
    def test_shared_files_are_current(self):
        x=self.run_tool(SYNC,'--pack-root',ROOT);self.assertTrue(x['valid']);self.assertFalse(x['applied'])
    def test_behavior_definitions_not_results(self):
        x=self.run_tool(VALIDATOR,'--pack-root',ROOT);self.assertGreaterEqual(x['eval_cases'],70);self.assertFalse(x['behavior_evaluated'])
    def test_platform_workspace_and_router_handoffs_match_hub(self):
        context=(ROOT/'_shared/context-policy.md').read_text(encoding='utf-8')
        self.assertIn('`.drobotics-x5/`',context)
        self.assertIn('`.drobotics-s/`',context)
        self.assertNotIn('`.drobotics/`',context)
        self.assertNotIn('`.horizon/`',context)
        handoff=(ROOT/'_shared/toolchain-handoff.md').read_text(encoding='utf-8')
        entry=(ROOT/'rdk-model-zoo/SKILL.md').read_text(encoding='utf-8')
        self.assertIn('drobotics-router',handoff)
        self.assertNotIn('horizon-router',handoff)
        self.assertIn('drobotics-router',entry)
        self.assertNotIn('horizon-router',entry)
        contract=(ROOT/'rdk-model-zoo-release/references/release-contract.md').read_text(encoding='utf-8')
        self.assertIn('`.drobotics-x5/`',contract)
        self.assertIn('`.drobotics-s/`',contract)
        self.assertNotIn('`.drobotics/`',contract)
        self.assertNotIn('`.horizon/`',contract)
    def test_released_pack_and_member_versions_are_consistent(self):
        manifest=json.loads((ROOT/'pack.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['version'],'1.0.1')
        self.assertEqual(manifest['release_state'],'released')
        self.assertEqual((ROOT/'VERSION').read_text(encoding='utf-8').strip(),manifest['version'])
        self.assertEqual({row['name']:row['version'] for row in manifest['skills']}, {
            'rdk-model-zoo':'1.1.1',
            'rdk-model-zoo-repo':'1.0.1',
            'rdk-model-zoo-integrate':'1.0.1',
            'rdk-model-zoo-develop':'1.0.1',
            'rdk-model-zoo-validate':'1.0.1',
            'rdk-model-zoo-review':'1.0.1',
            'rdk-model-zoo-release':'1.0.1',
        })
        readme=(ROOT/'README.md').read_text(encoding='utf-8')
        self.assertIn('Pack 1.0.1 已发布',readme)
        self.assertIn('Skill 版本为 1.1.1',readme)
        self.assertIn('其余六个 Skill 为 1.0.1',readme)
        changelog=(ROOT/'CHANGELOG.md').read_text(encoding='utf-8')
        self.assertIn('## Pack 1.0.1',changelog)
        for row in manifest['skills']:
            skill_root=ROOT/row['name']
            text=(skill_root/'SKILL.md').read_text(encoding='utf-8')
            frontmatter=re.match(r'^---\n(.*?)\n---\n',text,re.S)
            self.assertIsNotNone(frontmatter,row['name'])
            version=re.search(r'^version: ["\']?([^"\'\n]+)["\']?$',frontmatter[1],re.M)
            self.assertIsNotNone(version,row['name'])
            self.assertEqual(version[1],row['version'])
            card=(skill_root/'skill-card.md').read_text(encoding='utf-8')
            self.assertIn(f'| Skill 版本 | {row["version"]} |',card)
            self.assertIn(f'| Pack 版本 | {manifest["version"]} |',card)
    def test_single_flat_install_is_self_contained(self):
        with tempfile.TemporaryDirectory() as td:
            for source in ROOT.iterdir():
                if not (source/'SKILL.md').is_file():continue
                dest=Path(td)/source.name;shutil.copytree(source,dest)
                x=self.run_tool(VALIDATOR,'--skill-root',dest);self.assertTrue(x['valid'])
    def test_broken_local_reference_detected(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'rdk-model-zoo';shutil.copytree(ROOT/dest.name,dest)
            (dest/'references/context-policy.md').unlink()
            x=self.run_tool(VALIDATOR,'--skill-root',dest,code=2);self.assertFalse(x['valid'])
    def test_cross_skill_file_dependency_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'rdk-model-zoo';shutil.copytree(ROOT/dest.name,dest)
            with (dest/'SKILL.md').open('a',encoding='utf-8') as f:f.write('\n[unsafe](../another/secret.md)\n')
            x=self.run_tool(VALIDATOR,'--skill-root',dest,code=2);self.assertIn('escapes',str(x['errors']))
    def test_generated_reference_drift_is_reported_without_writing(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'skills';shutil.copytree(ROOT,dest)
            p=dest/'rdk-model-zoo/references/context-policy.md';p.write_text('changed\n')
            x=self.run_tool(SYNC,'--pack-root',dest,code=2);self.assertFalse(x['valid']);self.assertEqual(p.read_text(),'changed\n')
    def test_sync_refuses_overwriting_unmanaged_file(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'skills';shutil.copytree(ROOT,dest)
            p=dest/'rdk-model-zoo/references/context-policy.md';p.write_text('manual user file\n')
            x=self.run_tool(SYNC,'--pack-root',dest,'--apply',code=2);self.assertFalse(x['valid']);self.assertEqual(p.read_text(),'manual user file\n')
    def test_generator_repairs_managed_reference(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'skills';shutil.copytree(ROOT,dest)
            p=dest/'rdk-model-zoo/references/context-policy.md';old=p.read_text(encoding='utf-8');p.write_text(old+'\nextra\n',encoding='utf-8')
            self.run_tool(SYNC,'--pack-root',dest,'--apply');self.assertEqual(p.read_text(encoding='utf-8'),old)
if __name__=='__main__':unittest.main()
