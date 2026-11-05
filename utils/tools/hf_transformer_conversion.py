# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Frozen trained Wav2Vec2/SigLIP inputs and complete FP32 source exports."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import subprocess
import numpy as np

FACTS = {
    'siglip': {
        'repository': 'google/siglip-base-patch16-224',
        'revision': '7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed',
        'filename': 'model.safetensors',
        'checkpoint_sha256': '2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8',
        'config_sha256': 'cd85b3d28829722820bcb89a2cfbb4160e55fd359249a3044da724166a8d9688',
        'preprocessor_sha256': 'd11ccb80f15d358a11bdb070e92e2d889005874b7db15823d5f10d9b2533b14a',
        'variant': 'base-patch16-224-bicubic-dual-r02',
        'calibration_count': 200, 'atol': .006, 'rtol': .001,
        'inputs': {'_input_0': [1, 3, 224, 224]},
        'outputs': {'pooler_output': [1, 768], 'last_hidden_state': [1, 196, 768]},
    },
    'asr': {
        'repository': 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn',
        'revision': '99ccb2737be22b8bb50dcfcc39ad4d567fb90cfd',
        'filename': 'pytorch_model.bin',
        'checkpoint_sha256': 'de031fd4b29e0c0667e5346450fadfe1326c89936b888b59c4ede608db763ee4',
        'config_sha256': '9af98614e322a5300c75605227d93c98af0c8f78754aba1dfc279166688afddd',
        'preprocessor_sha256': 'c403ce09975b90dff0dd8302c42d422e9de1f166cd7772df23490069893cb0cf',
        'variant': 'xlsr53-zh3503-30k-mask-r02',
        'calibration_count': 50, 'atol': .002, 'rtol': .001,
        'inputs': {'waveform': [1, 30000], 'attention_mask': [1, 30000]},
        'outputs': {'logits': [1, 93, 3503]},
    },
}


def identity(path):
    """Identify one frozen file without retaining a checkpoint in memory."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest(),
            'size_bytes': path.stat().st_size}


def prepare(kind, source_dir, source_manifest, output, purpose='calibration'):
    """Prepare all real tensors and compare each with the pinned HF processor."""
    import transformers
    from transformers import SiglipImageProcessor, Wav2Vec2FeatureExtractor
    if transformers.__version__ != '4.51.3':
        raise ValueError('The processor implementation is pinned to 4.51.3')
    facts = FACTS[kind]
    source_dir, source_manifest, output = map(Path, (source_dir, source_manifest, output))
    if identity(source_dir / 'preprocessor_config.json')['sha256'] != facts['preprocessor_sha256']:
        raise ValueError('The pinned upstream preprocessing configuration differs')
    source = json.loads(source_manifest.read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    if kind == 'siglip':
        from samples.vision.siglip.runtime.python.public_preprocess import prepare_image
        from PIL import Image
        producer = prepare_image
        processor = SiglipImageProcessor.from_pretrained(str(source_dir), local_files_only=True)
        originals = source['images']
        for index, row in enumerate(originals):
            path = Path(row['source'])
            source_id = identity(path)
            if row.get('source_sha256') != source_id['sha256']:
                raise ValueError('Frozen image identity differs')
            value = prepare_image(path)
            with Image.open(path) as image:
                expected = processor(images=image.convert('RGB'), return_tensors='np')['pixel_values']
            if not np.array_equal(value, expected):
                raise ValueError('Native image preparation differs from upstream')
            directory = output / 'inputs' / '_input_0'
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f'{index:04d}.npy'
            np.save(target, value)
            rows.append({'source': source_id, 'inputs': {'_input_0': identity(target)}})
    else:
        from samples.speech.asr.runtime.python.public_preprocess import prepare_waveform
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        from math import gcd
        producer = prepare_waveform
        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(source_dir), local_files_only=True)
        for index, row in enumerate(source['records']):
            path = Path(row['wav_path'])
            source_id = identity(path)
            if row.get('wav_sha256') != source_id['sha256']:
                raise ValueError('Frozen waveform identity differs')
            rate, wave = wavfile.read(path)
            if purpose == 'calibration':
                if rate != 16000 or wave.dtype != np.float32 or wave.ndim != 1:
                    raise ValueError('Calibration requires frozen 16kHz mono F32 WAV')
            else:
                if wave.dtype == np.int16:
                    wave = wave.astype(np.float32) / 32768
                elif wave.dtype == np.int32:
                    wave = wave.astype(np.float32) / 2147483648
                else:
                    wave = wave.astype(np.float32)
                if wave.ndim == 2:
                    wave = wave.mean(axis=1, dtype=np.float32)
                if rate != 16000:
                    factor = gcd(rate, 16000)
                    wave = resample_poly(wave, 16000 // factor, rate // factor).astype(np.float32)
            start = row.get('start_sample', 0)
            length = row.get('valid_samples', min(30000, len(wave) - start))
            raw = wave[start:start + length].copy()
            values = prepare_waveform(raw)
            expected = processor(raw, sampling_rate=16000, padding='max_length',
                                 max_length=30000, truncation=True,
                                 return_attention_mask=True, return_tensors='np')
            if (not np.array_equal(values['waveform'], expected['input_values'])
                    or not np.array_equal(values['attention_mask'], expected['attention_mask'])):
                raise ValueError('Native waveform/mask differs from upstream')
            prepared = {}
            for name, value in values.items():
                directory = output / 'inputs' / name
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f'{index:04d}.npy'
                np.save(target, value)
                prepared[name] = identity(target)
            rows.append({'source': source_id, 'start_sample': start,
                         'valid_samples': len(raw), 'inputs': prepared})
    expected_count = facts['calibration_count'] if purpose == 'calibration' else 8
    if len(rows) != expected_count:
        raise ValueError('The frozen calibration count differs')
    report = {'sample': kind, 'variant': facts['variant'], 'purpose': purpose, 'count': len(rows),
              'source_manifest': identity(source_manifest), 'records': rows,
              'input_specs': facts['inputs'], 'official_preprocessing_byte_exact': True,
              'preprocessor_config': identity(source_dir / 'preprocessor_config.json'),
              'native_preprocessor': identity(inspect.getfile(producer)),
              'preparation_tool': identity(__file__),
              'upstream_processor': identity(inspect.getfile(type(processor))),
              'transformers': transformers.__version__, 'board': 'not-run'}
    (output / 'calibration-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def compare(actual, expected, facts):
    """Check every finite raw F32 output under the owner-approved fixed gate."""
    if tuple(actual) != tuple(facts['outputs']) or tuple(expected) != tuple(facts['outputs']):
        raise ValueError('A complete output head is missing or out of order')
    result = {}
    for name, shape in facts['outputs'].items():
        value, reference = actual[name], expected[name]
        if (value.dtype != np.float32 or reference.dtype != np.float32
                or value.shape != reference.shape or list(value.shape) != shape
                or not np.isfinite(value).all() or not np.isfinite(reference).all()):
            raise ValueError('Raw output shape/dtype/finite contract differs')
        delta = np.abs(value.astype(np.float64) - reference.astype(np.float64))
        scale = np.abs(reference.astype(np.float64))
        result[name] = {
            'shape': shape, 'dtype': 'float32', 'elements': value.size,
            'max_abs': float(delta.max()),
            'violations': int(np.count_nonzero(delta > facts['atol'] + facts['rtol'] * scale)),
            'old_gate_violations': int(np.count_nonzero(delta > 1e-4 + 1e-3 * scale)),
            'bit_exact': bool(np.array_equal(value, reference)),
        }
    return result


def export(kind, source_dir, calibration_manifest, output, validation_manifest=None):
    """Strictly load trained weights, export complete heads, and verify all inputs."""
    import torch
    import transformers
    import onnx
    import onnxruntime as ort
    from transformers import SiglipConfig, SiglipVisionModel, Wav2Vec2ForCTC
    facts = FACTS[kind]
    source_dir, calibration_manifest, output = map(Path, (source_dir, calibration_manifest, output))
    checkpoint = identity(source_dir / facts['filename'])
    config_id = identity(source_dir / 'config.json')
    if checkpoint['sha256'] != facts['checkpoint_sha256'] or config_id['sha256'] != facts['config_sha256']:
        raise ValueError('Source checkpoint/config differs from the frozen source')
    if transformers.__version__ != '4.51.3':
        raise ValueError('The source implementation is pinned to transformers 4.51.3')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if kind == 'siglip':
        from safetensors.torch import load_file
        config = SiglipConfig.from_json_file(str(source_dir / 'config.json')).vision_config
        config._attn_implementation = 'eager'
        if (config.image_size, config.patch_size, config.hidden_size) != (224, 16, 768):
            raise ValueError('Wrong vision architecture')
        model = SiglipVisionModel(config).eval().float()
        state = load_file(str(source_dir / facts['filename']), device='cpu')
        model.load_state_dict({k: v for k, v in state.items() if k.startswith('vision_model.')}, strict=True)
        del state
        def original(feed):
            value = model(torch.from_numpy(feed['_input_0']), return_dict=True)
            return {'pooler_output': value.pooler_output.numpy().copy(),
                    'last_hidden_state': value.last_hidden_state.numpy().copy()}
        class Complete(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.model = backbone

            def forward(self, pixels):
                value = self.model(pixels, return_dict=True)
                return value.pooler_output, value.last_hidden_state
    else:
        vocabulary = identity(source_dir / 'vocab.json')
        if vocabulary['sha256'] != '33fea3444869c2cd2433f59da079b04ce91515f946d21fe1b0ff3825398bcec7':
            raise ValueError('The complete 3503-token vocabulary differs')
        model, loading = Wav2Vec2ForCTC.from_pretrained(
            str(source_dir), local_files_only=True, output_loading_info=True,
            attn_implementation='eager')
        if any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise ValueError('The complete trained source state did not load strictly')
        model = model.eval().float()
        if model.config.vocab_size != 3503 or model.config.pad_token_id != 0:
            raise ValueError('Wrong CTC vocabulary or blank ID')
        def original(feed):
            value = model(torch.from_numpy(feed['waveform']),
                          attention_mask=torch.from_numpy(feed['attention_mask']).long())
            return {'logits': value.logits.numpy().copy()}
        class Complete(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.model = backbone

            def forward(self, waveform, attention_mask):
                return self.model(waveform, attention_mask=attention_mask.long(), return_dict=False)[0]
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite trained parameter: ' + name)
    parameter_count = sum(p.numel() for p in model.parameters())
    manifest = json.loads(calibration_manifest.read_text())
    if (manifest['sample'] != kind or manifest['purpose'] != 'calibration'
            or manifest['count'] != facts['calibration_count']
            or not manifest['official_preprocessing_byte_exact']):
        raise ValueError('Wrong or unverified calibration manifest')
    cases = []
    prepared_rows = [('real', manifest['records'])]
    if validation_manifest is not None:
        validation = json.loads(Path(validation_manifest).read_text())
        if (validation['sample'] != kind or validation['purpose'] != 'validation'
                or validation['count'] != 8 or not validation['official_preprocessing_byte_exact']):
            raise ValueError('Wrong independent validation inputs')
        prepared_rows.append(('heldout', validation['records']))
    for prefix, records in prepared_rows:
        for index, row in enumerate(records):
            feed = {}
            for name, shape in facts['inputs'].items():
                source = row['inputs'][name]
                if identity(source['path'])['sha256'] != source['sha256']:
                    raise ValueError('Calibration tensor identity differs')
                value = np.load(source['path'], allow_pickle=False)
                if value.dtype != np.float32 or list(value.shape) != shape or not np.isfinite(value).all():
                    raise ValueError('Wrong calibration tensor boundary')
                feed[name] = value
            cases.append((f'{prefix}-{index:04d}', feed))
    random = np.random.default_rng(42 if kind == 'siglip' else 0)
    feed = ({'_input_0': random.uniform(-1, 1, (1, 3, 224, 224)).astype(np.float32)}
            if kind == 'siglip' else
            {'waveform': random.normal(size=(1, 30000)).astype(np.float32),
             'attention_mask': np.ones((1, 30000), np.float32)})
    cases.append(('deterministic-random', feed))
    output.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        references = [original(feed) for _, feed in cases]
    folded = []
    if kind == 'asr':
        for name, module in model.named_modules():
            if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
                before = module.weight.detach().clone()
                torch.nn.utils.parametrize.remove_parametrizations(module, 'weight', leave_parametrized=True)
                torch.testing.assert_close(before, module.weight, atol=0, rtol=0)
                folded.append(name)
    wrapper = Complete(model).eval()
    wrapper_checks = []
    with torch.inference_mode():
        for (name, feed), reference in zip(cases, references):
            raw = wrapper(*(torch.from_numpy(feed[k]) for k in facts['inputs']))
            values = raw if isinstance(raw, tuple) else (raw,)
            actual = {key: v.numpy().copy() for key, v in zip(facts['outputs'], values)}
            wrapper_checks.append({'case': name, 'outputs': compare(actual, reference, facts)})
        onnx_path = output / 'model.onnx'
        torch.onnx.export(wrapper, tuple(torch.from_numpy(cases[0][1][k]) for k in facts['inputs']),
                          str(onnx_path), input_names=list(facts['inputs']),
                          output_names=list(facts['outputs']), opset_version=17,
                          dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])
    for observed, expected in [(session.get_inputs(), facts['inputs']), (session.get_outputs(), facts['outputs'])]:
        if [(x.name, x.shape, x.type) for x in observed] != [(k, v, 'tensor(float)') for k, v in expected.items()]:
            raise ValueError('ONNX complete public ABI differs')
    report = {'sample': kind, 'variant': facts['variant'], 'status': 'running',
              'native_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                  cwd=Path(__file__).resolve().parents[2], text=True).strip(),
              'source': {'repository': 'https://huggingface.co/' + facts['repository'],
                         'revision': facts['revision'], 'checkpoint': checkpoint, 'config': config_id,
                         'weight_url': f"https://huggingface.co/{facts['repository']}/resolve/{facts['revision']}/{facts['filename']}",
                         'license': 'Apache-2.0', 'precision': 'float32'},
              'parameter_count': parameter_count, 'onnx': identity(onnx_path),
              'calibration_manifest': identity(calibration_manifest),
              'validation_manifest': identity(validation_manifest) if validation_manifest else None,
              'exporter': identity(__file__),
              'input_specs': facts['inputs'], 'output_specs': facts['outputs'],
              'gate': {'revision': 'owner-approved-20261105-guide-v2.21',
                       'atol': facts['atol'], 'rtol': facts['rtol'], 'all_elements_required': True},
              'folded_weight_norm': folded, 'wrapper_checks': wrapper_checks, 'checks': [],
              'versions': {'torch': torch.__version__, 'transformers': transformers.__version__,
                           'onnx': onnx.__version__, 'onnxruntime': ort.__version__},
              'execution': {'provider': 'CPUExecutionProvider', 'mkldnn': False, 'threads': 2},
              'board': 'not-run', 'dataset_accuracy': 'not-run'}
    for (name, feed), reference in zip(cases, references):
        values = session.run(None, feed)
        actual = {key: value for key, value in zip(facts['outputs'], values)}
        report['checks'].append({'case': name, 'outputs': compare(actual, reference, facts)})
        (output / 'export.json').write_text(json.dumps(report, indent=2) + '\n')
    failed = any(x['violations'] for row in report['checks'] + wrapper_checks for x in row['outputs'].values())
    report['status'] = 'failed' if failed else 'passed'
    (output / 'export.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(kind, operation, argv=None):
    """Run explicit server-side preparation/export; never download model files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    if operation == 'prepare':
        parser.add_argument('--source-manifest', type=Path, required=True)
        parser.add_argument('--purpose', choices=['calibration', 'validation'], default='calibration')
        args = parser.parse_args(argv)
        report = prepare(kind, args.source_dir, args.source_manifest, args.output, args.purpose)
        print(json.dumps({'sample': kind, 'calibration_count': report['count'],
                          'official_preprocessing_byte_exact': True}))
        return 0
    parser.add_argument('--calibration-manifest', type=Path, required=True)
    parser.add_argument('--validation-manifest', type=Path)
    args = parser.parse_args(argv)
    report = export(kind, args.source_dir, args.calibration_manifest, args.output, args.validation_manifest)
    print(json.dumps({'sample': kind, 'status': report['status'], 'report': str(args.output / 'export.json')}))
    return int(report['status'] != 'passed')
