"""New featuremap specs must retain their exact artifact, rank and output ABI."""
from pathlib import Path
import copy, json, unittest
from utils.py_utils.converted_featuremap import select, bind
from utils.py_utils.runtime_meta import MetadataMismatchError

ROOT=Path(__file__).resolve().parents[3]


class ConvertedFeaturemapTest(unittest.TestCase):
    def profiles(self):
        rows=[]
        for sample in ['vit','fcos','lanenet','unet','3dresnet']:
            folder=ROOT/'samples/vision'/sample;path=folder/'model/converted_featuremap.json'
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


if __name__=='__main__':unittest.main()
