"""Prepare SDK masks, reference tokens and local RPC resources on the host."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoTokenizer
from llm_compression.models.generate_utils import get_causal_mask, get_causal_mask_chunks


def sha256(path):
    """Return a streaming file digest without loading model files into memory."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    """Validate reference inputs and export a board-local evaluation bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=Path, default=Path('MiniCPM5-2B'), help='Original pinned HF checkpoint')
    parser.add_argument('--test-data', type=Path, default=Path('datasets/wikitext2-test/test-00000-of-00001.parquet'), help='Verified WikiText2 TEST parquet')
    parser.add_argument('--output-dir', type=Path, default=Path('ppl-bundle'), help='New directory for generated board resources')
    args = parser.parse_args()
    weights = args.model_path/'model-00000-of-00001.safetensors'
    assert sha256(weights) == '14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
    assert sha256(args.test_data) == '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
    with safe_open(str(weights), framework='pt') as model:
        embedding = model.get_tensor('model.embed_tokens.weight').to(torch.float16).numpy()
        assert hashlib.sha256(embedding.tobytes()).hexdigest() == '5e720b334ed68a4949481837d004be9afdee9ef08c32cceb005d215b5dd82918'
    del embedding
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    dataset = load_dataset('parquet', data_files={'test': str(args.test_data)}, split='test')
    ids = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt').input_ids.numpy()
    assert ids.shape == (1, 288009)
    masks = get_causal_mask_chunks(get_causal_mask(torch.ones((1,2048),dtype=torch.int32),4096)
                                  .squeeze(1).to(torch.bfloat16),4096,256)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir/'input_ids.npy', ids)
    np.save(args.output_dir/'masks.npy', torch.stack(masks).float().numpy())
    distribution = importlib.metadata.distribution('hbm-infer')
    assert distribution.version == '3.15.3', distribution.version
    package = Path(distribution.locate_file('hbm_infer'))
    protocol = args.output_dir/'rpc_protocol'
    protocol.mkdir()
    (protocol/'__init__.py').touch()
    # These SDK-owned files are copied only from the user's installed SDK.
    shutil.copy2(package/'frame_pb2.py', protocol/'frame_pb2.py')
    shutil.copytree(package/'server_linux', args.output_dir/'sdk-runtime')
    manifest = {str(p.relative_to(args.output_dir)): sha256(p)
                for p in args.output_dir.rglob('*') if p.is_file()}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('EVALUATION_BUNDLE_READY', args.output_dir)


if __name__ == '__main__':
    main()
