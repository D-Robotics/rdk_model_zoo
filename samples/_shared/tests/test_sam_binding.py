"""Offline metadata and dual-runtime boundaries for both SAM consumers."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from samples._shared.assets import Asset
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError
from samples._shared.sam_binding import (resolve_selection, list_available_assets,
    bind_model, bind_stage, validate_tensors)
from samples._shared.sam_runner import RuntimeModelRunner


def metadata(sample,stage,target='s100',dtype='float32',box_shape=None,space=128):
    inp='batched_images' if sample=='efficient_sam' else 'normalized_images'
    inputs={inp:(1,3,512,512)} if stage=='encoder' else {'image_embeddings':(1,256,32,32)}
    if stage=='decoder' and sample=='mobile_sam':
        inputs['boxes']=box_shape or ((1,4,1,1) if target=='x5' else (1,4))
    outputs={'image_embeddings':(1,256,32,32)} if stage=='encoder' else {
        'low_res_masks':(1,3,space,space), 'iou_predictions':(1,3,1,1) if target=='x5' else (1,3)}
    return RuntimeMetadata.from_mapping(dict(model_name=stage,input_names=tuple(inputs),
        input_shapes=inputs,input_dtypes={n:'float32' for n in inputs},
        output_names=tuple(outputs),output_shapes=outputs,output_dtypes={n:dtype for n in outputs},
        output_quants={n:dict(scale=0.1,zero_point=4) for n in outputs}))


class FakeRuntime:
    def __init__(self,meta):
        for field in ('model_names','input_names','input_shapes','input_dtypes','output_names','output_shapes','output_dtypes','output_quants'):
            setattr(self,field,getattr(meta,field))
        self.name=meta.model_name
        self.calls=[]
        self.schedules=[]
        self.outputs={n:np.full(s,3,dtype=meta.output_dtypes[n]) for n,s in meta.output_shapes.items()}
    def run(self,inputs):
        self.calls.append(inputs)
        return {self.name:self.outputs}
    def set_scheduling_params(self,priority=None,bpu_cores=None):
        """Native protocol per X5 board evidence 2026-09-24: Mapping arguments only.

        The real signature is priority: Mapping[str, SupportsInt] and
        bpu_cores: Mapping[str, Sequence[SupportsInt]]; anything else is an
        incompatible-arguments TypeError exactly like the scalar board failure.
        """
        for name,value in (('priority',priority),('bpu_cores',bpu_cores)):
            if value is None:continue
            if not isinstance(value,dict) or not value:
                raise TypeError(f'set_scheduling_params(): incompatible function arguments: {name} must be a nonempty Mapping')
            for model,entry in value.items():
                valid=(type(entry) is int) if name=='priority' else (
                    isinstance(entry,(list,tuple)) and bool(entry) and all(type(core) is int for core in entry))
                if not valid:
                    raise TypeError(f'set_scheduling_params(): incompatible function arguments: {name}[{model!r}]')
        self.schedules.append({k:v for k,v in (('priority',priority),('bpu_cores',bpu_cores)) if v is not None})


class SAMBindingTests(unittest.TestCase):
    def test_all_eight_pairs_and_external_paths(self):
        for sample in ('efficient_sam','mobile_sam'):
            with patch('samples._shared.platforms.detect_target',side_effect=AssertionError('no board reads')):
                self.assertEqual(len(list_available_assets(sample)),8)
                for target in ('x5','s100','s100p','s600'):
                    s=resolve_selection(sample,target)
                    self.assertEqual(len(list_available_assets(sample,target)),2)
                    self.assertIn('encoder',s.encoder_asset.filename)
                    self.assertIn('decoder',s.decoder_asset.filename)
                    self.assertEqual(s.encoder_model_path.name,Path(s.encoder_asset.filename).name)
                    override=resolve_selection(sample,target,encoder_asset_id=s.encoder_asset.reference,
                        encoder_model_path='/tmp/custom-encoder.bin')
                    self.assertEqual(override.encoder_model_path,Path('/tmp/custom-encoder.bin'))
                    with self.assertRaises(ValueError):
                        resolve_selection(sample,target,encoder_model_path='/tmp/custom-encoder.bin')
                    with self.assertRaises(ValueError):
                        resolve_selection(sample,target,encoder_asset_id=s.decoder_asset.reference)
                    foreign=resolve_selection(sample,'s600' if target!='s600' else 's100')
                    with self.assertRaises(ValueError):
                        resolve_selection(sample,target,decoder_asset_id=foreign.decoder_asset.reference)
        with self.assertRaises(ValueError): list_available_assets('typo')
        with self.assertRaises(ValueError): list_available_assets('mobile_sam','x3')

    def test_binding_rechecks_publication_facts(self):
        s=resolve_selection('efficient_sam','s100')
        bad=replace(s,encoder_asset=replace(s.encoder_asset,url='https://invalid.example/a'))
        with self.assertRaisesRegex(ValueError,'publication|manifest'):
            bind_model(bad,metadata(s.sample,'encoder'),metadata(s.sample,'decoder'))

    def test_bound_metadata_is_an_immutable_snapshot(self):
        source=metadata('efficient_sam','encoder')
        b=bind_stage('efficient_sam','s100','encoder',source)
        with self.assertRaises(TypeError): b.metadata.input_shapes['batched_images']=(2,)
        with self.assertRaises(TypeError): b.metadata.output_dtypes['image_embeddings']='int8'
        with self.assertRaises(TypeError): b.metadata.output_quants['image_embeddings']['scale']=9
        source.input_shapes['batched_images']=(9,)
        source.output_quants['image_embeddings']['scale']=9
        self.assertEqual(b.metadata.input_shapes['batched_images'],(1,3,512,512))
        self.assertEqual(b.metadata.output_quants['image_embeddings']['scale'],0.1)

    def test_native_dtype_protocols_and_target_box_shapes(self):
        for sample in ('efficient_sam','mobile_sam'):
            for target in ('x5','s100','s100p','s600'):
                for dtype in ('float32','float16','int8','uint8','int16','int32'):
                    s=resolve_selection(sample,target)
                    b=bind_model(s,metadata(sample,'encoder',target,dtype),metadata(sample,'decoder',target,dtype))
                    self.assertEqual(b.decoder.metadata.output_dtypes['low_res_masks'],dtype)
                    self.assertEqual(b.encoder.metadata.output_quants['image_embeddings']['scale'],0.1)
        for shape in ((1,4),(1,4,1,1)):
            bind_stage('mobile_sam','x5','decoder',metadata('mobile_sam','decoder','x5',box_shape=shape))
        with self.assertRaises(ValueError):
            bind_stage('mobile_sam','s100','decoder',metadata('mobile_sam','decoder',box_shape=(1,4,1,1)))
        b=bind_stage('efficient_sam','s600','decoder',metadata('efficient_sam','decoder','s600',space=256))
        self.assertEqual(b.metadata.output_shapes['low_res_masks'],(1,3,256,256))
        with self.assertRaises(ValueError):
            bind_stage('efficient_sam','x5','decoder',metadata('efficient_sam','decoder','x5',space=256))

    def test_unknown_ambiguous_or_invalid_metadata_rejected(self):
        sample='mobile_sam'; good=metadata(sample,'decoder')
        bads=[replace(good,input_names=('image_embeddings',)),
            replace(good,input_dtypes={**good.input_dtypes,'boxes':'int32'}),
            replace(good,output_dtypes={**good.output_dtypes,'low_res_masks':'mystery'}),
            replace(good,output_names=('low_res_masks','low_res_masks')),
            replace(good,output_shapes={**good.output_shapes,'low_res_masks':(2,3,128,128)}),
            replace(good,output_shapes={**good.output_shapes,'low_res_masks':(1,4,128,128)}),
            replace(good,output_shapes={**good.output_shapes,'low_res_masks':(1,3,0,128)}),
            replace(good,output_shapes={**good.output_shapes,'iou_predictions':(3,)}),
            replace(good,input_shapes={**good.input_shapes,'boxes':(4,)}),
            replace(good,model_names=('decoder','extra'))]
        for bad in bads:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError): bind_stage(sample,'s100','decoder',bad)

    def test_tensor_validation_is_pure_strict_and_finite(self):
        b=bind_stage('efficient_sam','s100','decoder',metadata('efficient_sam','decoder',dtype='int16'))
        raw={'low_res_masks':np.arange(3*128*128,dtype=np.int16).reshape(1,3,128,128),
             'iou_predictions':np.array([[4,8,2]],dtype=np.int16)}
        result=validate_tensors(b,raw,outputs=True)
        for name in raw: self.assertIs(result[name],raw[name])
        with self.assertRaises(ValueError): validate_tensors(b,{**raw,'extra':np.zeros(1)},outputs=True)
        with self.assertRaises(ValueError): validate_tensors(b,{**raw,'iou_predictions':raw['iou_predictions'].astype(np.float32)},outputs=True)
        with self.assertRaises(ValueError): validate_tensors(b,{'image_embeddings':np.zeros((1,256,32,32),dtype=np.float64)})
        v=np.zeros((1,256,32,32),dtype=np.float32); v.flat[0]=np.nan
        with self.assertRaises(ValueError): validate_tensors(b,{'image_embeddings':v})

    def test_lazy_identity_gate_precedes_factory(self):
        s=resolve_selection('efficient_sam','s100')
        with patch('samples._shared.platforms.require_execution_target',side_effect=ValueError('Target mismatch')) as gate, patch('samples._shared.sam_runner._default_runtime_factory') as factory:
            r=RuntimeModelRunner(s)
            self.assertFalse(r.loaded)
            with self.assertRaisesRegex(ValueError,'Target mismatch'): r.load()
            gate.assert_called_once_with('s100'); factory.assert_not_called()

    def make_runner(self,target='s100',sample='efficient_sam',bad_decoder=False):
        s=resolve_selection(sample,target)
        e=FakeRuntime(metadata(sample,'encoder',target,dtype='int16'))
        d=FakeRuntime(metadata(sample,'decoder',target,dtype='int16'))
        if bad_decoder: d.input_shapes={'image_embeddings':(1,3)}
        fixtures={str(s.encoder_model_path):e,str(s.decoder_model_path):d}
        return RuntimeModelRunner(s,runtime_factory=fixtures.__getitem__),e,d

    def test_both_models_bind_before_execution_and_failure_is_named(self):
        r,e,d=self.make_runner(bad_decoder=True)
        with self.assertRaisesRegex(ValueError,'decoder'): r.load()
        self.assertFalse(r.loaded); self.assertEqual(e.calls,[]); self.assertEqual(d.calls,[])
        with self.assertRaises(RuntimeError): r.encoder({'batched_images':np.zeros(1)})

    def test_container_adaptation_and_raw_outputs_unchanged(self):
        for target in ('x5','s100','s100p','s600'):
            r,e,d=self.make_runner(target)
            b=r.load(); self.assertIs(r.load(),b)
            inp={'batched_images':np.zeros((1,3,512,512),dtype=np.float32)}
            outputs=r.encoder(inp)
            self.assertIs(outputs['image_embeddings'],e.outputs['image_embeddings'])
            self.assertEqual(outputs['image_embeddings'].flat[0],3)
            self.assertEqual(set(e.calls[0]),set(inp) if target=='x5' else {'encoder'})
            if target=='x5': self.assertIs(e.calls[0]['batched_images'],inp['batched_images'])
            else: self.assertIs(e.calls[0]['encoder']['batched_images'],inp['batched_images'])
            with patch.object(d,'run',side_effect=RuntimeError('hardware fault')):
                with self.assertRaisesRegex(RuntimeError,'decoder.*hardware fault'):
                    r.decoder({'image_embeddings':np.zeros((1,256,32,32),dtype=np.float32)})

    def test_scheduling_is_applied_to_both_and_rejects_invalid_requests(self):
        for priority in (0,7):
            for target in ('x5','s100'):
                r,e,d=self.make_runner(target)
                r.set_scheduling_params(priority=priority,bpu_cores=None if target=='x5' else [0])
                expected={'priority':{'encoder':priority}} if target=='x5' else {'priority':{'encoder':priority},'bpu_cores':{'encoder':[0]}}
                self.assertEqual(e.schedules,[expected])
                self.assertEqual(d.schedules,[{'priority':{'decoder':priority}}] if target=='x5'
                                 else [{'priority':{'decoder':priority},'bpu_cores':{'decoder':[0]}}])
                for kwargs in ({'priority':-1},{'priority':256},{'bpu_cores':[]},{'bpu_cores':[-1]}):
                    with self.assertRaises(ValueError): r.set_scheduling_params(**kwargs)
                self.assertEqual(len(e.schedules),1)
        r,e,d=self.make_runner('x5')
        with self.assertRaises(ValueError): r.set_scheduling_params(bpu_cores=[0])

    def test_scalar_priority_never_reaches_the_native_protocol(self):
        """X5 board evidence 2026-09-24: native set_scheduling_params takes Mappings only."""
        for target in ('x5','s100'):
            r,e,d=self.make_runner(target)
            r.set_scheduling_params(priority=5)
            self.assertEqual(e.schedules,[{'priority':{'encoder':5}}])
            self.assertEqual(d.schedules,[{'priority':{'decoder':5}}])

    def test_runtime_output_must_match_bound_metadata_not_reshape(self):
        r,e,d=self.make_runner(); r.load()
        e.outputs['image_embeddings']=e.outputs['image_embeddings'].reshape(-1)
        with self.assertRaisesRegex(ValueError,'encoder'):
            r.encoder({'batched_images':np.zeros((1,3,512,512),dtype=np.float32)})
        r,e,d=self.make_runner(); r.load()
        with patch.object(e,'run',return_value=e.outputs):
            with self.assertRaisesRegex(ValueError,'encoder'):
                r.encoder({'batched_images':np.zeros((1,3,512,512),dtype=np.float32)})

if __name__=='__main__': unittest.main()
