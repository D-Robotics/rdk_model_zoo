"""A detector bundle alone cannot certify a tracking task."""
import copy, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from component_package import validate_tracking_reference


class TrackingReleaseTest(unittest.TestCase):
    def setUp(self):
        self.row={'model':{'source':'fasterrcnn','task':'track'},'artifact':{
            'components':[{'role':r,'sha256':r+'-hbm'}for r in ['parta','partb']],
            'bundle':{'members':[{'filename':'tracking.json','sha256':'a'*64}]}},
            'provenance':{'source_checkpoint_sha256':'b'*64,'components':{
                r:{'onnx_sha256':r+'-onnx'}for r in ['parta','partb']}},'validation':{
                'host_tracking':{'status':'passed','evidence_file':'tracking.json','evidence_sha256':'a'*64,
                    'component_sha256':{r:r+'-hbm'for r in ['parta','partb']}}}}
        self.host={'status':'passed','source_checkpoint_sha256':'b'*64,
            'onnx_sha256':{r:r+'-onnx'for r in ['parta','partb']},'frame_count':2,
            'stable_adjacent_ids':True,'all_five_detector_outputs_preserved':True,
            'frames':[{'frame':i,'outputs':['rpn_obj','rpn_deltas','feat0','cls_packed','reg_packed'],
                       'tracks':[{'id':1}]}for i in [1,2]]}

    def test_connected_tracking_proof_and_both_detector_identities_pass(self):
        validate_tracking_reference(self.row,self.host)

    def test_detector_only_or_another_detector_is_rejected(self):
        original=copy.deepcopy(self.row)
        for field,value in [('evidence_sha256','0'*64),('component_sha256',{}),('status','not-run')]:
            self.row=copy.deepcopy(original);self.row['validation']['host_tracking'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_tracking_reference(self.row)
        self.row=original;self.host['onnx_sha256']['partb']='other-graph'
        with self.assertRaises(ValueError):validate_tracking_reference(self.row,self.host)

    def test_reordered_frames_or_missing_head_is_rejected(self):
        self.host['frames'].reverse()
        with self.assertRaises(ValueError):validate_tracking_reference(self.row,self.host)
        self.host['frames'].reverse();self.host['frames'][0]['outputs'].pop()
        with self.assertRaises(ValueError):validate_tracking_reference(self.row,self.host)


if __name__=='__main__':unittest.main()
