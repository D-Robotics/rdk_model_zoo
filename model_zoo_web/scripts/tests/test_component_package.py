"""A downloadable pair must be complete and tied to both native OE reports."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from component_package import validate_components, validate_bundle_files
from stage_converted import stage
from upload_archive_assets import validate_package


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ComponentPackageTests(unittest.TestCase):
    def setUp(self):
        self.files = {'encoder.hbm': b'encoder fixture', 'decoder.hbm': b'decoder fixture',
                      'encoder.html': b'<!doctype html><html>encoder OE</html>',
                      'decoder.html': b'<!doctype html><html>decoder OE</html>', 'LICENSE.txt': b'fixture license'}
        self.files['host-smoke.json'] = json.dumps({'status':'passed','checkpoint_sha256':'d'*64,
            'encoder_onnx_sha256':'a'*64,'decoder_onnx_sha256':'a'*64,
            'cases':[{'input':name,'comparisons':{'connected':{'passed':True}}}
                     for name in ('first.jpg','second.jpg','third.jpg','seed0-random')]}).encode()
        components = [{'role':r, 'filename':r+'.hbm', 'platform':'s100', 'format':'HBM',
                       'url':f'https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s100/mobile_sam/{r}.hbm',
                       'sha256':sha(self.files[r+'.hbm']), 'size_bytes':len(self.files[r+'.hbm'])}
                      for r in ('encoder','decoder')]
        tensor = [{'name':'test', 'shape':[1,256,32,32], 'dtype':'float32'}]
        self.files['deployment.json'] = json.dumps({'schema_version':1, 'target':'s100',
                                                    'kind':'encoder-decoder', 'components':components}).encode()
        for component in components:
            self.files[component['role']+'.oe.json'] = json.dumps({
                'provenance':{'artifact_sha256':component['sha256'],'onnx_sha256':'a'*64,'config_sha256':'b'*64},
                'target':{'platform':'s100'},'io':{'inputs':tensor,'outputs':tensor}}).encode()
        self.row = {'model':{'source':'mobile_sam'}, 'target':{'platform':'s100'},
                    'artifact':{'filename':'complete.tar.gz', 'sha256':'f'*64, 'components':components,
                                'bundle':{'kind':'encoder-decoder', 'descriptor':'deployment.json',
                                          'members':[{'filename':n, 'sha256':sha(v), 'size_bytes':len(v)}
                                                     for n,v in self.files.items() if not n.endswith('.hbm')]}},
                    'provenance':{'source_checkpoint_sha256':'d'*64,'components':{r:{'onnx_sha256':'a'*64, 'config_sha256':'b'*64}
                                                for r in ('encoder','decoder')}},
                    'conversion':{'components':{r:{'status':'passed', 'artifact_sha256':c['sha256'],
                                                   'inputs':tensor, 'outputs':tensor,
                                                   'oe_report_filename':r+'.html',
                                                   'oe_data_filename':r+'.oe.json',
                                                   'oe_data_sha256':sha(self.files[r+'.oe.json']),
                                                   'oe_report_sha256':sha(self.files[r+'.html'])}
                                                for r,c in zip(('encoder','decoder'),components)}},
                    'validation':{'host_pipeline':{'status':'passed', 'stages':['encoder','decoder'],
                                                   'evidence_file':'host-smoke.json','evidence_sha256':sha(self.files['host-smoke.json']),
                                                   'component_sha256':{c['role']:c['sha256'] for c in components}}}}
        self.oe = {'provenance':{'artifact_sha256':'f'*64},
                   'components':{r:{**self.row['provenance']['components'][r],
                                    'artifact_sha256':c['sha256'],
                                    'oe_data_sha256':sha(self.files[r+'.oe.json']),
                                    'oe_report_sha256':sha(self.files[r+'.html'])}
                                 for r,c in zip(('encoder','decoder'),components)}}
        self.row['conversion']['outputs'] = tensor
        self.html = ('<!doctype html><script id="component-reports" type="application/json">'+json.dumps(
            {r:base64.b64encode(self.files[r+'.html']).decode() for r in ('encoder','decoder')})+'</script>').encode()

    def archive(self, files=None):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w:gz') as tar:
            for name,data in (self.files if files is None else files).items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info,io.BytesIO(data))
        return output.getvalue()

    def test_complete_pair_and_both_original_reports_are_accepted(self):
        validate_bundle_files(self.row,self.archive(),self.html,self.oe)

    def test_stage_and_archive_keep_the_complete_pair_inventory(self):
        row,oe = copy.deepcopy(self.row),copy.deepcopy(self.oe)
        data = self.archive()
        digest = sha(data)
        row.update(status='converted',metadata_revision=1)
        row['artifact'].update(sha256=digest,size_bytes=len(data),
            url='https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s100/mobile_sam/complete.tar.gz')
        row['validation'].update(artifact_sha256=digest,conversion='passed',host_smoke='passed',
            **{k:'not-run' for k in ('board_smoke','accuracy','runtime_1','runtime_2','cpp_1','cpp_2')})
        oe['provenance']['artifact_sha256'] = digest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model,html,oe_path = root/'complete.tar.gz',root/'report.html',root/'oe.json'
            model.write_bytes(data)
            html.write_bytes(self.html)
            oe_path.write_text(json.dumps(oe))
            stage(row,model,html,oe_path,root/'package')
            validate_package(root/'package')
            checks = (root/'package/SHA256SUMS').read_text()
            for name in ('encoder.hbm','decoder.hbm','complete.tar.gz'):
                self.assertIn(name,checks)

    def test_missing_replaced_or_unlisted_component_is_rejected(self):
        for change in ('missing','replaced','unlisted'):
            files = dict(self.files)
            if change == 'missing': files.pop('decoder.hbm')
            if change == 'replaced': files['decoder.hbm'] = b'replaced bytes'
            if change == 'unlisted': files['another.bin'] = b'extra'
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_bundle_files(self.row,self.archive(files),self.html,self.oe)

    def test_swapped_target_or_duplicate_roles_are_rejected(self):
        for field,value in [('platform','s600'),('role','encoder'),('sha256','d'*64)]:
            row = copy.deepcopy(self.row)
            row['artifact']['components'][1][field] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_components(row,platform='s100',source='mobile_sam',oe=self.oe)

    def test_old_component_oe_cannot_certify_a_new_pair(self):
        oe = copy.deepcopy(self.oe)
        oe['components']['decoder']['artifact_sha256'] = 'd'*64
        with self.assertRaises(ValueError):
            validate_bundle_files(self.row,self.archive(),self.html,oe)
        old_html = ('<script id="component-reports">'+json.dumps(
            {r:base64.b64encode(self.files['encoder.html']).decode() for r in ('encoder','decoder')})+'</script>').encode()
        with self.assertRaises(ValueError):
            validate_bundle_files(self.row,self.archive(),old_html,self.oe)
        files = dict(self.files)
        native = json.loads(files['decoder.oe.json'])
        native['provenance']['config_sha256'] = 'f'*64
        files['decoder.oe.json'] = json.dumps(native).encode()
        row,oe = copy.deepcopy(self.row),copy.deepcopy(self.oe)
        digest = sha(files['decoder.oe.json'])
        next(x for x in row['artifact']['bundle']['members'] if x['filename']=='decoder.oe.json').update(
            sha256=digest,size_bytes=len(files['decoder.oe.json']))
        row['conversion']['components']['decoder']['oe_data_sha256'] = digest
        oe['components']['decoder']['oe_data_sha256'] = digest
        with self.assertRaises(ValueError):
            validate_bundle_files(row,self.archive(files),self.html,oe)

    def test_disconnected_smokes_or_one_component_provenance_are_rejected(self):
        for section,field,value in [('validation','host_pipeline',{}),
                                    ('provenance','onnx_sha256','a'*64),
                                    ('conversion','components',{'encoder':self.row['conversion']['components']['encoder']})]:
            row = copy.deepcopy(self.row)
            row[section][field] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_components(row,platform='s100',source='mobile_sam')

    def test_host_report_digest_and_source_graph_identity_are_bound(self):
        row = copy.deepcopy(self.row)
        row['validation']['host_pipeline']['evidence_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            validate_bundle_files(row,self.archive(),self.html,self.oe)
        for field,value in [('checkpoint_sha256','0'*64),('decoder_onnx_sha256','0'*64),('status','failed')]:
            files = dict(self.files)
            host = json.loads(files['host-smoke.json'])
            host[field] = value
            files['host-smoke.json'] = json.dumps(host).encode()
            row = copy.deepcopy(self.row)
            digest = sha(files['host-smoke.json'])
            row['validation']['host_pipeline']['evidence_sha256'] = digest
            next(x for x in row['artifact']['bundle']['members'] if x['filename']=='host-smoke.json').update(
                sha256=digest,size_bytes=len(files['host-smoke.json']))
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_bundle_files(row,self.archive(files),self.html,self.oe)

    def test_native_pair_task_cannot_be_registered_as_one_hbm(self):
        row = {'artifact':{},'provenance':{},'conversion':{}}
        with self.assertRaises(ValueError):
            validate_components(row,platform='s100',source='efficient_sam')
        self.assertIsNone(validate_components(row,platform='s100',source='convnext'))


if __name__ == '__main__':
    unittest.main()
