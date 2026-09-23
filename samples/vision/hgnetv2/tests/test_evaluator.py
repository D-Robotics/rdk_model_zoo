"""HGNetV2 dataset evaluation must stay SDK-free and report real coverage."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[4]

class EvaluatorTests(unittest.TestCase):
    def test_help_without_board_sdk(self):
        result = subprocess.run([sys.executable, str(ROOT/'samples/vision/hgnetv2/evaluator/eval.py'), '--help'], cwd='/tmp', text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--val-csv', result.stdout)

    def test_csv_nested_paths_and_invalid_or_conflicting_labels(self):
        from samples.vision.hgnetv2.evaluator.eval import load_ground_truth_csv
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp)/'gt.csv'
            csv.write_text('image:file,category\na\\b.jpg,2\n')
            self.assertEqual(load_ground_truth_csv(csv), {'a/b.jpg':2})
            for data in ('a.jpg,1000\n','a.jpg,nope\n','a.jpg,1\na.jpg,2\n'):
                csv.write_text(data)
                with self.assertRaises(ValueError):load_ground_truth_csv(csv)

    def test_evaluation_denominator_and_partial_coverage(self):
        from samples.vision.hgnetv2.evaluator.eval import evaluate_images, collect_images_with_relative_paths
        from samples._shared.classification import ClassificationResult
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'nested').mkdir()
            for name in ('a.jpg','b.jpg','unmatched.jpg'):
                cv2.imwrite(str(root/'nested'/name), np.zeros((3,4,3),np.uint8))
            (root/'nested/broken.jpg').write_bytes(b'not an image')
            images=collect_images_with_relative_paths(root)
            result=ClassificationResult(np.array([1,2]),np.array([.8,.2]),('1','2'))
            summary=evaluate_images(images,{'nested/a.jpg':1,'nested/b.jpg':2,'nested/broken.jpg':3},lambda image:result,top_k=2)
            self.assertEqual(summary['total_images_scanned'],4)
            self.assertEqual(summary['matched_to_gt'],3)
            self.assertEqual(summary['successful_inferences'],2)
            self.assertEqual(summary['failed_images'],1)
            self.assertEqual(summary['unmatched_images'],1)
            self.assertEqual(summary['top1_acc'],.5)
            self.assertEqual(summary['topk_acc'],1.)
            self.assertNotIn('top5_acc',summary)
            self.assertEqual(summary['status'],'partial')

    def test_empty_evaluation_is_not_a_zero_accuracy_success(self):
        from samples.vision.hgnetv2.evaluator.eval import evaluate_images
        summary=evaluate_images([],{},lambda _:None,top_k=5)
        self.assertEqual(summary['status'],'no-results')
        self.assertIsNone(summary['top1_acc'])
        self.assertIsNone(summary['top5_acc'])

    def test_evaluator_default_resize_preserves_source(self):
        from samples.vision.hgnetv2.evaluator.eval import build_parser
        args=build_parser().parse_args(['--image-path','images','--val-csv','truth.csv'])
        self.assertEqual(args.resize_type,0)
        self.assertEqual(args.top_k,5)
        self.assertEqual(args.limit,0)

    def test_main_writes_metrics_using_the_real_task_with_injected_runtime(self):
        from unittest.mock import patch
        from samples.vision.hgnetv2.evaluator.eval import main
        from samples.vision.hgnetv2.runtime.python.model_binding import bind_model
        from samples.vision.hgnetv2.runtime.python import model_runner
        import contextlib
        import io
        import json

        class HostRunner:
            def __init__(self, selection):
                self.selection = selection
            def load(self):
                return bind_model(self.selection, {'model_name':'host',
                    'input_names':['data'],'input_shapes':{'data':(1,3,224,224)},
                    'input_dtypes':{'data':'U8'},'output_names':['scores'],
                    'output_shapes':{'scores':(1,1000)},'output_dtypes':{'scores':'F32'}})
            def set_scheduling_params(self, **kwargs):
                pass
            def __call__(self, tensors):
                scores = np.zeros((1,1000),dtype=np.float32)
                scores[0,999] = 10
                return {'scores':scores}

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=root/'images';images.mkdir()
            cv2.imwrite(str(images/'one.jpg'),np.zeros((5,7,3),np.uint8))
            truth=root/'truth.csv';truth.write_text('one.jpg,999\n')
            output=root/'results.json'
            with patch.object(model_runner,'RuntimeModelRunner',HostRunner), contextlib.redirect_stdout(io.StringIO()):
                rc=main(['--target','x5','--image-path',str(images),'--val-csv',str(truth),'--json-save-path',str(output)])
            self.assertEqual(rc,0)
            result=json.loads(output.read_text())
            self.assertEqual(result['top1_acc'],1)
            self.assertEqual(result['top5_acc'],1)
            self.assertEqual(result['successful_inferences'],1)
            self.assertEqual(result['config']['resize_type'],0)
