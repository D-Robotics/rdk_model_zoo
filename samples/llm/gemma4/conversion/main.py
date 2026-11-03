#!/usr/bin/env python3
"""Render and invoke the SDK's released Gemma4 recipe, without adapting SDK code."""
import argparse
import copy
import json
from pathlib import Path
import subprocess


SOURCE_ID = 'google/gemma-4-E2B-it-qat-q4_0-unquantized'
SOURCE_REVISION = '6befbaca7398925921802abd1f277b495b78b738'


def render(sdk, model, wikitext, mmstar, output):
    import yaml  # Supplied by the conversion SDK; inference does not import it.
    sdk, model, wikitext, mmstar, output = [Path(p).resolve() for p in (sdk, model, wikitext, mmstar, output)]
    if output.exists():
        raise ValueError('Use a fresh output directory; existing conversion runs are not overwritten')
    recipe = sdk / 'llm_compression/lightcompress/configs/released_models/gemma4_e2b/lm_w4_kv_w8'
    configs = []
    for index, name in enumerate(('step_1_autoclip.yml', 'step_2_gptq.yml')):
        config = yaml.safe_load((recipe / name).read_text())
        config['model']['path'] = str(model if index == 0 else output / 'artifacts/step1_autoclip/transformed_model')
        config['calib']['path'] = str(wikitext)
        for evaluation in config['eval']:
            evaluation['path'] = str(wikitext)
        config['save']['save_path'] = str(output / ('artifacts/step1_autoclip' if index == 0 else 'artifacts/step2_gptq'))
        configs.append((name, config))
    hpp = yaml.safe_load((sdk / 'llm_compression/configs/gemma4.yml').read_text())
    hpp['model'].update(path=str(model), march='nash-p', model_list=['lm'], max_new_tokens=64,
                        fake_quant_path=str(output / 'artifacts/step2_gptq/fake_quant_model'))
    hpp['model']['text_config'].update(max_kvcache_len=4096, max_lm_input_len=512)
    artifacts = output / 'artifacts/hpp'
    hpp['calibration'].update(dataset_type='mmstar', data_path=str(mmstar), calibration_step=30,
                              log_path=str(artifacts / 'qconfig'), calib_ckpt_save_path=str(artifacts / 'calibration'))
    hpp['evaluation'] = dict(dataset_type='ppl', dataset_name='wikitext2', data_path=str(wikitext),
                             seq_len=2048, bs=1, calib_ckpt_load_path=str(artifacts / 'calibration'),
                             result_path=str(artifacts / 'fakequant_eval'))
    hpp['compile'].update(hbm_save_path=str(artifacts / 'hbm'), calib_ckpt_load_path=str(artifacts / 'calibration'),
                           cache_path=str(artifacts / 'compiler_cache'), core_num=4, jobs=32,
                           embed_dtype='fp16', return_all_logits=True)
    floating = copy.deepcopy(hpp)
    floating['model'].pop('fake_quant_path', None)
    floating['evaluation'].pop('calib_ckpt_load_path', None)
    floating['evaluation']['result_path'] = str(artifacts / 'float_eval')
    configs.extend([('gemma4_text_s600.yml', hpp), ('gemma4_text_float.yml', floating)])
    (output / 'configs').mkdir(parents=True)
    for name, value in configs:
        (output / 'configs' / name).write_text(yaml.safe_dump(value, sort_keys=False))
    (output / 'plan.json').write_text(json.dumps({'sdk_root': str(sdk), 'source_model': SOURCE_ID,
        'source_revision': SOURCE_REVISION, 'target': 's600', 'board_tests': 'not-run'}, indent=2) + '\n')
    return output


def command(work, stage, cuda):
    work = Path(work).resolve()
    sdk = Path(json.loads((work / 'plan.json').read_text())['sdk_root'])
    if stage in ('autoclip', 'gptq'):
        name = 'step_1_autoclip.yml' if stage == 'autoclip' else 'step_2_gptq.yml'
        argv = ['bash', str(sdk / 'llm_compression/lightcompress/scripts/run_llmc_serial.sh'),
                '--cuda_devices', str(cuda), str(work / 'configs' / name)]
    else:
        script = {'float': 'torch_eval.sh', 'calib': 'calib.sh', 'eval': 'torch_eval.sh', 'compile': 'compile.sh'}[stage]
        name = 'gemma4_text_float.yml' if stage == 'float' else 'gemma4_text_s600.yml'
        argv = ['bash', str(sdk / 'llm_compression/scripts' / script), '--config_path', str(work / 'configs' / name)]
    return sdk, argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare = sub.add_parser('prepare')
    for key in ('sdk-root', 'model-path', 'wikitext-dir', 'mmstar', 'output'):
        prepare.add_argument('--' + key, type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--work-dir', type=Path, required=True)
    run.add_argument('--stage', choices=['autoclip', 'gptq', 'float', 'calib', 'eval', 'compile'], required=True)
    run.add_argument('--cuda-device', default='0')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(render(args.sdk_root, args.model_path, args.wikitext_dir, args.mmstar, args.output))
    else:
        cwd, argv = command(args.work_dir, args.stage, args.cuda_device)
        raise SystemExit(subprocess.run(argv, cwd=cwd, check=False).returncode)


if __name__ == '__main__':
    main()
