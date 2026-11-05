#!/usr/bin/env python3
"""Export the pinned OpenAI ViT-B/32 pair and check the native matching path."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys


def identity(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new attempt directory')
    checkpoint = identity(args.checkpoint)
    if checkpoint['sha256'] != '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af':
        raise ValueError('checkpoint differs from the official ViT-B/32 SHA-256')
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from samples.vision.clip.runtime.python.matching import prepare_inputs
    from samples.vision.clip.runtime.python.simple_tokenizer import PromptTokenizer

    upstream = {}
    for name, expected in {
        'model': '9902cbe5ee90a1da2aa3e6f043e8a23dc1f8831193b963785c9af03d5c7bef2c',
        'simple_tokenizer': 'd1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad',
    }.items():
        path = args.source_dir / 'clip' / (name + '.py')
        if identity(path)['sha256'] != expected:
            raise ValueError(f'upstream {name} differs from the frozen commit')
        spec = importlib.util.spec_from_file_location('openai_clip_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        upstream[name] = module

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The official checkpoint is a frozen TorchScript archive. Build the eager
    # source implementation from its exact state, rather than exporting JIT code.
    archive = torch.jit.load(str(args.checkpoint), map_location='cpu').eval()
    model = upstream['model'].build_model(archive.state_dict()).float().eval()
    del archive
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).reshape(1, 3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).reshape(1, 3, 1, 1)

    class ImageEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = model.visual
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        def forward(self, image):
            # Native inference supplies RGB [0,1]; normalize exactly once here.
            return self.visual((image - self.mean) / self.std).float()

    class TextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, tokens):
            return self.model.encode_text(tokens).float()

    image_encoder, text_encoder = ImageEncoder().eval(), TextEncoder().eval()
    prompts = ['a diagram', 'a dog', 'a photo of a bird', 'a red car', '一个人', 'café']
    tokenizer = PromptTokenizer()
    tokens = tokenizer(prompts)
    official = upstream['simple_tokenizer'].SimpleTokenizer()
    official_tokens = np.zeros_like(tokens)
    for index, text in enumerate(prompts):
        row = [official.encoder['<|startoftext|>'], *official.encode(text), official.encoder['<|endoftext|>']]
        official_tokens[index, :len(row)] = row
    np.testing.assert_array_equal(tokens, official_tokens)
    native_bpe = Path(__file__).resolve().parents[1] / 'runtime/python/bpe_simple_vocab_16e6.txt.gz'
    official_bpe = args.source_dir / 'clip/bpe_simple_vocab_16e6.txt.gz'
    if identity(native_bpe)['sha256'] != identity(official_bpe)['sha256']:
        raise ValueError('native and official BPE vocabularies differ')
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if len(files) != 200:
        raise ValueError('exactly 200 frozen real calibration images are required')
    args.output.mkdir(parents=True)
    cal = args.output / 'calibration/image'
    cal.mkdir(parents=True)
    images, manifest = [], []
    for index, path in enumerate(files):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'cannot decode {path}')
        x = prepare_inputs(image, prompts, tokenizer).tensors['image']
        dest = cal / f'{index:04}.npy'
        np.save(dest, x, allow_pickle=False)
        manifest.append({'image': identity(path), 'tensor': identity(dest)})
        if index in (0, 52, 143):
            images.append((path.name, x))
    images.append(('random-seed0', np.random.default_rng(0).uniform(0, 1, (1, 3, 224, 224)).astype(np.float32)))
    manifest_path = args.output / 'calibration/manifest.json'
    manifest_path.write_text(json.dumps({'count': 200, 'producer': identity(__file__),
        'preprocess': 'native OpenCV cubic rounded-short224 center crop RGB float32 /255; normalization in image graph',
        'files': manifest}, indent=2) + '\n')
    graph_paths = {'image': args.output / 'image_encoder.onnx', 'text': args.output / 'text_encoder.onnx'}
    with torch.inference_mode():
        torch.onnx.export(image_encoder, torch.from_numpy(images[0][1]), str(graph_paths['image']),
            input_names=['image'], output_names=['image_features'], opset_version=args.opset,
            do_constant_folding=True, dynamo=False)
        torch.onnx.export(text_encoder, torch.from_numpy(tokens[:2]), str(graph_paths['text']),
            input_names=['tokens'], output_names=['text_features'], opset_version=args.opset,
            dynamic_axes={'tokens': {0: 'prompts'}, 'text_features': {0: 'prompts'}},
            do_constant_folding=True, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    sessions = {}
    for role, path in graph_paths.items():
        onnx.checker.check_model(str(path))
        sessions[role] = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    report = {'schema_version': 1, 'status': 'checking', 'checkpoint': checkpoint,
        'source_commit': 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6', 'exporter': identity(__file__),
        'graphs': {role: identity(path) for role, path in graph_paths.items()},
        'bpe': identity(native_bpe), 'tokenizer_matches_official': True,
        'calibration_manifest': identity(manifest_path), 'atol': 1e-4, 'rtol': 1e-3,
        'versions': {'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__},
        'cases': [], 'board': 'not-run', 'dataset_accuracy': 'not-run'}
    def check(actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
        return float(np.max(np.abs(actual - expected)))
    with torch.inference_mode():
        for index, (name, image) in enumerate(images):
            selected_tokens = tokens[:(1, 2, 6, 3)[index]]
            input_image = torch.from_numpy(image)
            input_tokens = torch.from_numpy(selected_tokens)
            ref_image = model.encode_image((input_image - mean) / std).numpy()
            ref_text = model.encode_text(input_tokens).numpy()
            check(image_encoder(input_image).numpy(), ref_image)
            check(text_encoder(input_tokens).numpy(), ref_text)
            got_image = sessions['image'].run(None, {'image': image})[0]
            got_text = sessions['text'].run(None, {'tokens': selected_tokens})[0]
            row = {'name': name, 'prompt_count': len(selected_tokens),
                'image_max_abs': check(got_image, ref_image), 'text_max_abs': check(got_text, ref_text)}
            def cosine(a, b):
                return (a / np.linalg.norm(a, axis=-1, keepdims=True)) @ (b / np.linalg.norm(b, axis=-1, keepdims=True)).T
            row['cosine_max_abs'] = check(cosine(got_image, got_text), cosine(ref_image, ref_text))
            row['status'] = 'passed'
            report['cases'].append(row)
    report['status'] = 'passed'
    (args.output / 'host-smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'cases': report['cases']}), flush=True)


if __name__ == '__main__':
    main()
