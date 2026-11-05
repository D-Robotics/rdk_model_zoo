"""A rebuilt ASR download and runtime selection must stay one complete package."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from samples.speech.paraformer.runtime.python.cli import converted_profiles, resolve_selections
from samples.speech.paraformer.runtime.python.pipeline import ParaformerPipeline, bind_model

SAMPLE=Path(__file__).resolve().parents[1]


class ConvertedProfileTests(unittest.TestCase):
    def setUp(self):
        self.variant,=converted_profiles('s100')

    def test_full_new_group_and_actual_collected_stage_metadata_bind(self):
        old=resolve_selections('s100');new=resolve_selections('s100',variant=self.variant)
        self.assertEqual(tuple(s.stage for s in new),('encoder','predictor','decoder'))
        self.assertTrue(all(a.asset.reference!=b.asset.reference for a,b in zip(old,new)))
        profile=json.loads((SAMPLE/'conversion/converted_targets_20261104.json').read_text())
        for selection in new:
            component=profile['components'][selection.stage]
            self.assertEqual(selection.asset.sha256,component['sha256'])
            facts={'model_name':component['actual_model_name']}
            for single,plural in [('input','inputs'),('output','outputs')]:
                facts.update({single+'_names':[v['name']for v in component[plural]],
                              single+'_shapes':{v['name']:v['shape']for v in component[plural]},
                              single+'_dtypes':{v['name']:v['dtype']for v in component[plural]}})
            bound=bind_model(selection,facts)
            self.assertEqual(bound.model_name,component['actual_model_name'])
        ids={s.stage:s.asset.reference for s in new}
        self.assertEqual(resolve_selections('s100',asset_ids=ids),new)
        ids['decoder']=old[2].asset.reference
        with self.assertRaises(ValueError):resolve_selections('s100',asset_ids=ids)

    def test_mixed_generation_is_rejected_before_runtime_construction(self):
        old=resolve_selections('s100');new=list(resolve_selections('s100',variant=self.variant))
        new[2]=old[2];calls=[]
        def factory(path):calls.append(path);raise AssertionError('No SDK call expected')
        with self.assertRaisesRegex(ValueError,'one complete publication'):
            ParaformerPipeline.from_models(new,[f'token{i}'for i in range(8404)],runtime_factory=factory)
        self.assertEqual(calls,[])

    def test_six_file_preview_uses_new_archive_and_renames_matching_tokens(self):
        with tempfile.TemporaryDirectory()as directory:
            dest=Path(directory)/'new'
            result=subprocess.run([sys.executable,str(SAMPLE/'model/download.py'),'--variant',self.variant,
                                   '--dry-run','--output-dir',str(dest)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            lines=result.stdout.splitlines();self.assertEqual(len(lines),6)
            remote=[line for line in lines if 'https://'in line]
            self.assertEqual(len(remote),4)
            self.assertTrue(all('/downloads/rdk_model_zoo_web/rdk_s100/paraformer/'in line for line in remote))
            self.assertFalse(any('bundle.tar.gz'in line for line in lines))
            token,=[line for line in remote if 'tokens8404'in line]
            self.assertTrue(token.endswith(str(dest/'s100/tokens.json')))
            self.assertFalse(dest.exists())


if __name__=='__main__':unittest.main()
