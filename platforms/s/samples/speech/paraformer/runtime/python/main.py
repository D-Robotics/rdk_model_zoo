"""Command-line entry point for Paraformer manifest inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from paraformer import Paraformer, ParaformerConfig, ParaformerFrontend, ParaformerFrontendConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed Paraformer runtime arguments.
    """
    parser = argparse.ArgumentParser(description='Run Paraformer S100 on a WAV manifest.')
    parser.add_argument('--manifest', required=True, help='JSON list with utt_id and optional reference text.')
    parser.add_argument('--audio-dir', required=True, help='Directory containing <utt_id>.wav files.')
    parser.add_argument('--encoder-model-path', default='/opt/hobot/model/s100/basic/paraformer/paraformer_large_encoder_400x560_s100.hbm')
    parser.add_argument('--predictor-model-path', default='/opt/hobot/model/s100/basic/paraformer/paraformer_large_predictor_400x512_s100.hbm')
    parser.add_argument('--decoder-model-path', default='/opt/hobot/model/s100/basic/paraformer/paraformer_large_decoder_400x512_s100.hbm')
    parser.add_argument('--tokens-path', default='/opt/hobot/model/s100/basic/paraformer/tokens.json')
    parser.add_argument('--cmvn-path', default='/opt/hobot/model/s100/basic/paraformer/am.mvn')
    parser.add_argument('--preprocess-only', action='store_true', help='Generate temporary features for C++ runtime.')
    parser.add_argument('--output-dir', help='Output directory for --preprocess-only features.')
    parser.add_argument('--max-utts', type=int, default=0, help='Maximum manifest items; zero processes all.')
    return parser.parse_args()


def main() -> None:
    """Run the preserved Paraformer pipeline and print stage timing summaries."""
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if args.max_utts:
        manifest = manifest[:args.max_utts]
    if args.preprocess_only:
        if not args.output_dir:
            raise ValueError('--output-dir is required with --preprocess-only.')
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frontend = ParaformerFrontend(ParaformerFrontendConfig(cmvn_path=args.cmvn_path))
        for item in manifest:
            audio_file = Path(args.audio_dir) / f"{item['utt_id']}.wav"
            features, feature_length = frontend.pre_process(audio_file)
            np.save(output_dir / f"{item['utt_id']}.npy", features)
            item['feat_length'] = feature_length
        Path(args.manifest).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
        return
    model = Paraformer(ParaformerConfig(
        encoder_model_path=args.encoder_model_path,
        predictor_model_path=args.predictor_model_path,
        decoder_model_path=args.decoder_model_path,
        tokens_path=args.tokens_path,
        cmvn_path=args.cmvn_path,
    ))
    timings = {'frontend_ms': 0.0, 'encoder_ms': 0.0, 'predictor_ms': 0.0, 'cif_ms': 0.0, 'decoder_ms': 0.0}
    processed = 0
    started = perf_counter()
    for item in manifest:
        audio_file = Path(args.audio_dir) / f"{item['utt_id']}.wav"
        if not audio_file.exists():
            print(f"Skip missing WAV: {audio_file}")
            continue
        frontend_begin = perf_counter()
        features, feature_length = model.pre_process(audio_file)
        frontend_ms = (perf_counter() - frontend_begin) * 1000
        logits, token_num, item_timings = model.forward(features, feature_length)
        item_timings['frontend_ms'] = frontend_ms
        text = model.post_process(logits, token_num)
        processed += 1
        for key, value in item_timings.items():
            timings[key] += value
        print(f"[{processed}] {item['utt_id']}: {text}")
    if not processed:
        raise RuntimeError('No manifest WAV files were processed.')
    frontend_total = timings['frontend_ms']
    hbm_pipeline_total = sum(value for key, value in timings.items() if key != 'frontend_ms')
    print(f'Processed {processed} utterances in {(perf_counter() - started) * 1000 / processed:.2f} ms/utt wall-clock')
    print('Average stages (ms): ' + ' '.join(f'{key}={value / processed:.2f}' for key, value in timings.items()))
    print(f'Average HBM pipeline (ms): {hbm_pipeline_total / processed:.2f}')
    print(f'Average end-to-end (ms): {(frontend_total + hbm_pipeline_total) / processed:.2f}')


if __name__ == '__main__':
    main()
