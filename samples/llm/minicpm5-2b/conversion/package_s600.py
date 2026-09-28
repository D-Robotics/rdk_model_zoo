"""Verify S600 HBM metadata and create a board deployment directory."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    """Validate command-line inputs and create the requested deployment files."""
    parser = argparse.ArgumentParser()
    parser.add_argument('hbm', type=Path)
    parser.add_argument('tokenizer', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    from hbdk4.compiler.hbm import Hbm
    model = Hbm(str(args.hbm))
    assert model.march_name.lower().replace('_', '-') == 'nash-p', model.march_name
    assert model.desc is not None, 'HBM metadata is absent'
    metadata = json.loads(model.desc.string_data)
    assert metadata['config']['head_dim'] == 128
    assert metadata['horizon']['head_dim'] == 128
    assert metadata['config']['num_hidden_layers'] == 42
    assert metadata['config']['vocab_size'] == 130560
    assert metadata['horizon']['prefill_chunk_size'] == 256
    assert metadata['horizon']['prefill_cache_len'] == 4096
    assert set(metadata['generation_config']['eos_token_id']) == {1, 130073}
    # Preserve the compiler-generated graph and metadata, adding the separately
    # stored HF chat template to the runtime's tokenizer metadata.
    tokenizer_config = json.loads((args.tokenizer/'tokenizer_config.json').read_text())
    metadata['tokenizer_config']['chat_template'] = tokenizer_config['chat_template']
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / args.hbm.name
    if destination.exists():
        raise FileExistsError(destination)
    model.staged_desc = json.dumps(metadata, ensure_ascii=False)
    model.save_by_staged_info(str(destination))
    final = Hbm(str(destination))
    assert json.loads(final.desc.string_data) == metadata
    for source in args.tokenizer.iterdir():
        if source.is_file(): shutil.copy2(source, args.output/source.name)
    embed_name = metadata['horizon']['embed_weight_name']
    shutil.copy2(args.hbm.parent/embed_name, args.output/embed_name)
    config = {'work_dir':str(args.output.resolve()), 'lm_model_file':destination.name,
              'embed_weight_name':embed_name, 'runtime_type':'LLM',
              'max_batch_size':1, 'max_conv_cache_num':0,
              'backends':{'prefill':[1,2,3,4], 'decode':[1,2,3,4]}}
    (args.output/'runtime_config.json').write_text(json.dumps(config, indent=2))
    (args.output/'hbm_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    checksums = {}
    for source in sorted(args.output.iterdir()):
        if source.is_file():
            digest = hashlib.sha256()
            with source.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
            checksums[source.name] = digest.hexdigest()
    (args.output/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in checksums.items()))
    print('S600_PACKAGE_METADATA_PASS', destination, flush=True)


if __name__ == '__main__':
    main()
