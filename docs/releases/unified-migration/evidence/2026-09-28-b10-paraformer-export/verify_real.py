"""Real-weight source decoder and complete Torch/ORT pipeline comparisons."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import torch
from samples.speech.paraformer.conversion.export import load_model, compare, signature
from samples.speech.paraformer.conversion.torch_stages import build_stages
from samples.speech.paraformer.runtime.python.pipeline import ParaformerPipeline, TensorNames
from samples.speech.paraformer.runtime.python.model_binding import CONTEXT
from samples._shared.assets import sha256_file

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=Path, required=True)
parser.add_argument('--export-dir', type=Path, required=True)
parser.add_argument('--disable-optimizations', action='store_true')
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(191009)
files = {name: args.model_dir / name for name in ('config.yaml', 'model.pt')}
stages = build_stages(load_model(files))
vocabulary = json.loads((args.model_dir / 'tokens.json').read_text())
names = TensorNames('speech', CONTEXT, CONTEXT, '/predictor/Add_output_0',
    '/predictor/Concat_5_output_0', CONTEXT, 'token_num', 'bias_embed', 'onnx::Shape_8609', 'logits')
options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
options.log_severity_level = 3
if args.disable_optimizations:
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sessions = {stage: ort.InferenceSession(str(args.export_dir / f'{stage}.onnx'),
    sess_options=options, providers=['CPUExecutionProvider']) for stage in stages}

def torch_runner(stage):
    def run(feed):
        module = stages[stage]
        inputs = [torch.from_numpy(feed[name]) for name, _, _ in signature(stage, 'input')]
        outputs = module(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        return {spec[0]: array.numpy() for spec, array in zip(signature(stage, 'output'), outputs, strict=True)}
    return run

def ort_runner(stage):
    def run(feed):
        session = sessions[stage]
        return dict(zip([o.name for o in session.get_outputs()], session.run(None, feed), strict=True))
    return run

pipeline_torch = ParaformerPipeline(*(torch_runner(s) for s in stages), names, vocabulary)
pipeline_ort = ParaformerPipeline(*(ort_runner(s) for s in stages), names, vocabulary)
result = {'weights_sha256': sha256_file(files['model.pt']), 'source_decoder': [], 'pipeline': [],
          'scope': 'Two example utterances, not dataset CER or board validation',
          'ort_optimizations': 'disabled' if args.disable_optimizations else 'all'}
manifest = ROOT / 'docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-cli/run-5_sjkdzc/features/prepared-manifest.json'
with torch.inference_mode():
    context = torch.randn(1, 400, 512)
    bias = torch.zeros(1, 1, 512)
    # Independent reference: original FunASR decoder, then the archived
    # deployment Range pass. Only its fixed /tmp path is redirected.
    class SourceDecoder(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
            self.register_buffer("lengths", torch.tensor([400], dtype=torch.int32))
        def forward(self, context, count, bias, acoustic):
            logits, _ = self.decoder(context, self.lengths, acoustic, count, bias)
            return torch.log_softmax(logits, dim=-1)
    with tempfile.TemporaryDirectory(dir=args.export_dir) as directory:
        directory = Path(directory)
        raw, folded = directory / "source.onnx", directory / "source-folded.onnx"
        torch.onnx.export(SourceDecoder(stages['decoder'].decoder).eval(),
            (context, torch.tensor([100], dtype=torch.int32), bias, torch.zeros(1,100,512)),
            str(raw), input_names=['context', 'count', 'bias', 'acoustic'],
            output_names=['logits'], opset_version=15, dynamo=False)
        source_path = ROOT / 'platforms/s/samples/speech/paraformer/conversion/05_fold_range.py'
        original_source = source_path.read_text()
        modified = original_source.replace('"/tmp/dp_range_probe.onnx"', repr(str(directory / 'probe.onnx')))
        assert modified != original_source
        namespace = {'__name__': 'archived_range_reference'}
        exec(compile(modified, str(source_path), 'exec'), namespace)
        namespace['main'](str(raw), str(folded))
        reference_session = ort.InferenceSession(str(folded), sess_options=options,
            providers=['CPUExecutionProvider'])
        result['source_range_sha256'] = sha256_file(source_path)
        result['source_reference_note'] = 'Upstream decoder export plus archived fixed-100 Range pass; only temporary probe path redirected'
        for count in (0, 1, 17, 100):
            acoustic = torch.zeros(1, 100, 512)
            acoustic[:, :count] = torch.randn(1, count, 512)
            length = torch.tensor([count], dtype=torch.int32)
            if count == 100:
                source_torch, _ = stages['decoder'].decoder(context, torch.tensor([400], dtype=torch.int32),
                    acoustic, length, bias)
                source_torch = torch.log_softmax(source_torch, dim=-1).numpy()
                result['torch_source_count100'] = compare([source_torch],
                    [stages['decoder'](context, length, bias, acoustic).numpy()])
            reference = reference_session.run(None, {'context': context.numpy(), 'count': length.numpy(),
                'bias': bias.numpy(), 'acoustic': acoustic.numpy()})
            torch_actual = stages['decoder'](context, length, bias, acoustic).numpy()
            actual = sessions['decoder'].run(None, {CONTEXT: context.numpy(), 'token_num': length.numpy(),
                'bias_embed': bias.numpy(), 'onnx::Shape_8609': acoustic.numpy()})[0]
            print('new torch vs ort', float(np.max(np.abs(actual-torch_actual))), flush=True)
            print('checking source count', count, 'maxdiff', float(np.max(np.abs(reference[0]-actual))), flush=True)
            differences = compare(reference, [actual])
            result['source_decoder'].append({'count': count, 'full_padded_output_max_abs_diff': differences,
                'new_torch_vs_ort_max_abs_diff': float(np.max(np.abs(actual-torch_actual)))})
    for record in json.loads(manifest.read_text()):
        path = manifest.parent / record['feature_file']
        assert sha256_file(path) == record['feature_sha256']
        value = np.load(path, allow_pickle=False)
        a = pipeline_torch.predict(value, record['feat_length'])
        b = pipeline_ort.predict(value, record['feat_length'])
        assert a.token_ids == b.token_ids and a.text == b.text and a.token_count == b.token_count
        result['pipeline'].append({'utt_id': record['utt_id'], 'feature_sha256': record['feature_sha256'],
            'reference_text': record['text'], 'torch': asdict(a), 'onnxruntime': asdict(b), 'ids_equal': True})
Path(__file__).with_name('real-comparison-noopt.json' if args.disable_optimizations else 'real-comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
