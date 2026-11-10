import copy
import importlib.util
import unittest
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / 'build_catalog.py'
SPEC = importlib.util.spec_from_file_location('converted_catalog', SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


class ConvertedCatalogTest(unittest.TestCase):
    def setUp(self):
        self.path = catalog.DATA_ROOT/'vision/ultralytics_yolo/yolo26/detect.yaml'
        self.record = yaml.safe_load(self.path.read_text())
        self.record['variants'] = [self.record['variants'][0]]
        variant = self.record['variants'][0]
        variant['platforms'] = [variant['platforms'][0]]
        self.original_performance = copy.deepcopy(variant['platforms'][0]['performance'])
        row = variant['platforms'][0]
        for key in ('accuracy','performance','released_at'): row.pop(key,None)
        row.update(status='converted',converted_at='2026-11-02',metadata_revision=1,
            conversion={'outputs':[{'name':'output','dtype':'float32','shape':[1,1000]}]},
            validation={'artifact_sha256':row['artifact']['sha256'],'conversion':'passed','host_smoke':'passed',
                        **{key:'not-run' for key in ('board_smoke','accuracy','runtime_1','runtime_2','cpp_1','cpp_2')}})
        row['provenance'].update(source_checkpoint_sha256='c'*64,checkpoint_revision='d'*40,
            source_weight_url='https://huggingface.co/example/model/resolve/'+'d'*40+'/model.safetensors',
            toolchain_image_id='sha256:'+'e'*64,onnx_sha256='a'*64,config_sha256='b'*64,
            repository_commit='94df7461027547906bbfb12016356f89123c0e74',
            toolchain_image='existing-test-image:v1',build_id='converted-test-r01',precision_policy='int8')
        filename=row['artifact']['url'].rsplit('/',1)[1]
        base='https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/assets/rdk_x5/ultralytics_yolo/'+Path(filename).stem+'/v1'
        row['artifact'].update(release_manifest_url=base+'/release.json',checksums_url=base+'/SHA256SUMS')
        row['reports'].update(oe_conversion_url=base+'/oe_report.html',oe_data_url=base+'/oe_report_data.json')
        self.row=row

    def validate(self):
        return catalog.validate_record(self.record,self.path)

    def test_conversion_can_be_published_without_board_metrics_or_flop_measurements(self):
        self.record['variants'][0].pop('model')
        result=self.validate()
        row=result['variants'][0]['platforms'][0]
        self.assertNotIn('accuracy',row)
        self.assertNotIn('performance',row)
        self.assertEqual(row['validation']['runtime_1'],'not-run')

    def test_promotion_to_released_still_requires_full_measurements(self):
        self.row.update(status='released',released_at='2026-11-02')
        with self.assertRaises(catalog.CatalogError): self.validate()

    def test_completed_accuracy_can_publish_a_failed_verdict_without_performance(self):
        self.row['accuracy'] = {
            'dataset': 'COCO val2017', 'task': 'object-detection', 'images': 5000,
            'float_onnx': {'map_50_95': 0.5, 'map_50': 0.7, 'map_75': 0.6},
            'runtime': {'map_50_95': 0.4, 'map_50': 0.6, 'map_75': 0.5},
            'acceptance': {
                'execution': 'completed', 'status': 'fail', 'campaign': 'fixture',
                'job_id': 'fixture-x5', 'baseline_key': 'fixture-baseline', 'policy_revision': 2,
                'thresholds': {'max_ap_drop': 0.03, 'min_ap_retention': 0.9},
                'report_url': 'https://example.org/evaluation/',
            },
        }
        self.row['validation']['accuracy'] = 'failed'
        self.row['validation']['evidence'] = {'accuracy': {
            'artifact_sha256': self.row['artifact']['sha256'], 'url': 'https://example.org/evaluation/data.json'}}
        self.validate()
        self.assertNotIn('performance', self.row)
        self.row['validation']['accuracy'] = 'passed'
        with self.assertRaises(catalog.CatalogError): self.validate()
        self.row['validation']['accuracy'] = 'failed'
        self.row['accuracy']['acceptance']['status'] = 'pass'
        with self.assertRaises(catalog.CatalogError): self.validate()
        self.row['accuracy']['acceptance']['status'] = 'fail'
        self.row['validation']['evidence']['accuracy']['artifact_sha256'] = '0' * 64
        with self.assertRaises(catalog.CatalogError): self.validate()

    def test_classification_gate_is_inclusive_and_checks_top1_and_top5(self):
        accuracy = {
            'dataset': 'ImageNetV2 MatchedFrequency', 'task': 'image-classification', 'images': 10000,
            'float_onnx': {'top1': 0.7, 'top5': 0.9}, 'runtime': {'top1': 0.6, 'top5': 0.8},
            'acceptance': {'execution': 'completed', 'status': 'pass', 'campaign': 'fixture',
                'job_id': 'fixture-x5', 'baseline_key': 'fixture-baseline', 'policy_revision': 2,
                'thresholds': {'max_top1_drop': 0.1, 'max_top5_drop': 0.1},
                'report_url': 'https://example.org/evaluation/'},
        }
        catalog.validate_accuracy(accuracy, 'cls', 'fixture.accuracy')
        for metric in ('top1', 'top5'):
            with self.subTest(metric=metric):
                invalid = copy.deepcopy(accuracy)
                invalid['runtime'][metric] -= 0.0001
                with self.assertRaises(catalog.CatalogError):
                    catalog.validate_accuracy(invalid, 'cls', 'fixture.accuracy')
                invalid['acceptance']['status'] = 'fail'
                catalog.validate_accuracy(invalid, 'cls', 'fixture.accuracy')

    def test_conversion_validation_cannot_describe_a_different_binary_or_missing_sources(self):
        for section,field,value in [('validation','artifact_sha256','0'*64),
                                    ('validation','conversion','not-run'),
                                    ('provenance','onnx_sha256',None),
                                    ('provenance','checkpoint_revision','main')]:
            original=self.row[section].get(field)
            self.row[section][field]=value
            with self.subTest(field=field),self.assertRaises(catalog.CatalogError): self.validate()
            self.row[section][field]=original

    def test_individual_runtime_measurements_need_a_matching_passed_check_and_evidence(self):
        self.row['performance']=copy.deepcopy(self.original_performance)
        self.row['performance']['measurements']=[self.original_performance['measurements'][0]]
        self.row['performance']['end_to_end']=[]
        with self.assertRaises(catalog.CatalogError): self.validate()
        self.row['validation']['runtime_1']='passed'
        with self.assertRaises(catalog.CatalogError): self.validate()
        self.row['validation']['evidence']={'runtime_1':{'artifact_sha256':self.row['artifact']['sha256'],
            'url':self.row['artifact']['release_manifest_url']}}
        self.validate()
        self.assertEqual(self.row['validation']['runtime_2'],'not-run')

    def test_measured_performance_keeps_unknown_frequency_and_needs_artifact_evidence(self):
        self.row['performance'] = copy.deepcopy(self.original_performance)
        performance = self.row['performance']
        end_to_end = performance['end_to_end']
        conditions = [end_to_end] if isinstance(end_to_end, dict) else end_to_end
        checks = [f"runtime_{m['threads']}" for m in performance['measurements']]
        checks += [f"cpp_{m['pipeline_streams']}" for m in conditions]
        for check in checks:
            self.row['validation'][check] = 'measured'
            self.row['validation'].setdefault('evidence', {})[check] = {
                'artifact_sha256': self.row['artifact']['sha256'],
                'url': 'https://example.org/performance.json',
            }
        for condition in conditions:
            condition.update(bpu_frequency_mhz=None, measurement_status='measured', environment_status='incomplete')
        self.validate()
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_end_to_end_record(conditions[0], 'released.end_to_end')
        conditions[0].pop('environment_status')
        with self.assertRaises(catalog.CatalogError): self.validate()
        conditions[0]['environment_status'] = 'incomplete'
        self.row['validation']['evidence'][checks[0]]['artifact_sha256'] = '0' * 64
        with self.assertRaises(catalog.CatalogError): self.validate()

    def test_artifact_license_can_differ_without_rewriting_existing_family_license(self):
        original=copy.deepcopy(self.record['license'])
        self.row['license']={'name':'MIT','url':'https://opensource.org/license/mit','scope':'model'}
        self.validate()
        self.assertEqual(self.record['license'],original)
        self.row['license']['scope']='sample-code'
        with self.assertRaises(catalog.CatalogError):self.validate()

    def test_direct_vendor_checkpoint_uses_its_exact_sha_as_an_immutable_version(self):
        self.row['provenance']['checkpoint_revision']='sha256:'+self.row['provenance']['source_checkpoint_sha256']
        self.validate()
        self.row['provenance']['checkpoint_revision']='sha256:'+'f'*64
        with self.assertRaises(catalog.CatalogError):self.validate()

    def test_speech_features_keep_three_ntc_dimensions_and_vision_keeps_nchw(self):
        self.record['domain'] = 'speech'
        self.path = catalog.DATA_ROOT/'speech/ultralytics_yolo/yolo26/detect.yaml'
        source = self.record['variants'][0]['input']['source']
        source.update(format='FEATUREMAP', layout='NTC', shape=[1,400,560])
        self.record['variants'][0]['input'].update(width=560,height=400)
        self.validate()
        source['shape'] = [1,1,400,560]
        with self.assertRaisesRegex(catalog.CatalogError, 'three NTC'):
            self.validate()
        source['shape'] = [1,400,560]
        self.record['domain'] = 'vision'
        self.path = catalog.DATA_ROOT/'vision/ultralytics_yolo/yolo26/detect.yaml'
        with self.assertRaisesRegex(catalog.CatalogError, 'four dimensions'):
            self.validate()

    def test_metadata_version_and_model_license_cannot_be_bypassed(self):
        self.row['metadata_revision']=2
        with self.assertRaises(catalog.CatalogError): self.validate()
        self.row['metadata_revision']=1
        self.record['license']['scope']='sample-code'
        with self.assertRaises(catalog.CatalogError): self.validate()

    def test_speech_waveform_keeps_its_real_two_dimensional_contract(self):
        self.record['domain']='speech'
        self.path=catalog.DATA_ROOT/'speech/ultralytics_yolo/yolo26/detect.yaml'
        source=self.record['variants'][0]['input']['source']
        source.update(format='FEATUREMAP',layout='NT',shape=[1,30000])
        self.record['variants'][0]['input'].update(width=30000,height=1)
        self.validate()
        source['shape']=[1,1,1,30000]
        with self.assertRaisesRegex(catalog.CatalogError,'two NT waveform'):
            self.validate()
        source['shape']=[1,30000];self.record['domain']='vision'
        self.path=catalog.DATA_ROOT/'vision/ultralytics_yolo/yolo26/detect.yaml'
        with self.assertRaisesRegex(catalog.CatalogError,'four dimensions'):
            self.validate()


if __name__=='__main__':
    unittest.main()
