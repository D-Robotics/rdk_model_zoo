"""New featuremap specs must retain their exact artifact, rank and output ABI."""
from pathlib import Path
import copy, json, unittest
from types import SimpleNamespace
import numpy as np
from utils.py_utils.converted_featuremap import Runner, select, bind
from utils.py_utils.runtime_meta import MetadataMismatchError

ROOT=Path(__file__).resolve().parents[3]


class ConvertedFeaturemapTest(unittest.TestCase):
    def profiles(self):
        rows=[]
        for domain,sample in [('vision',s)for s in ['vit','fcos','lanenet','unet','3dresnet','siglip']]+[('speech','asr')]:
            folder=ROOT/'samples'/domain/sample;path=folder/'model/converted_featuremap.json'
            if path.exists():
                rows.extend((folder,p)for p in json.loads(path.read_text())['profiles'])
        self.assertTrue(rows)
        return rows

    def test_published_identity_and_observed_all_head_abi(self):
        for folder,p in self.profiles():
            with self.subTest(sample=folder.name):
                selection=select(folder,p['target'],p['variant'],asset_id=p['asset_id'])
                meta={'model_name':p['model_name'],'model_names':[p['model_name']]}
                for side in ['input','output']:
                    rows=p['io'][side+'s']
                    meta.update({side+'_names':[r['name']for r in rows],
                        side+'_shapes':{r['name']:r['shape']for r in rows},
                        side+'_dtypes':{r['name']:r['dtype']for r in rows}})
                self.assertEqual(bind(selection,meta).model_name,p['model_name'])
                for mutation in ['model','rank','dtype','head']:
                    wrong=copy.deepcopy(meta)
                    if mutation=='model':wrong['model_name']='another-graph';wrong['model_names']=['another-graph']
                    if mutation=='rank':wrong['input_shapes'][wrong['input_names'][0]]=wrong['input_shapes'][wrong['input_names'][0]][1:]
                    if mutation=='dtype':wrong['output_dtypes'][wrong['output_names'][0]]='float16'
                    if mutation=='head':wrong['output_names']=wrong['output_names'][:-1]
                    with self.subTest(mutation=mutation),self.assertRaises((MetadataMismatchError,ValueError)):
                        bind(selection,wrong)

    def test_custom_path_and_cross_model_selection_cannot_inherit_a_profile(self):
        for folder,p in self.profiles():
            with self.subTest(sample=folder.name):
                with self.assertRaises(ValueError):select(folder,p['target'],p['variant'],model_path='arbitrary.hbm')
                with self.assertRaises(ValueError):select(folder,p['target'],p['variant'],asset_id='s:vit:s100/another.hbm')
                with self.assertRaises(ValueError):select(folder,'s600',p['variant'])

    def test_multi_input_masks_cannot_be_omitted_or_silently_cast(self):
        runner=Runner.__new__(Runner)
        runner.selection=SimpleNamespace(profile={'io':{'inputs':[
            {'name':'waveform','shape':[1,4],'dtype':'float32'},
            {'name':'attention_mask','shape':[1,4],'dtype':'float32'}]}})
        runner.backend='board';runner.runner=lambda feed:feed
        feed={'waveform':np.zeros((1,4),np.float32),'attention_mask':np.ones((1,4),np.float32)}
        self.assertIs(runner(feed)['waveform'],feed['waveform'])
        wrong_cases=[feed['waveform'],{'waveform':feed['waveform']},
                     {**feed,'attention_mask':feed['attention_mask'].astype(np.int64)},
                     {**feed,'attention_mask':np.ones((4,),np.float32)},
                     {**feed,'waveform':np.full((1,4),np.nan,np.float32)}]
        for wrong in wrong_cases:
            with self.assertRaises(ValueError):runner(wrong)


if __name__=='__main__':unittest.main()
