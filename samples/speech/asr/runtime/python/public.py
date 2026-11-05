# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pinned Chinese XLSR53 CTC recognition with independent masked windows."""
from pathlib import Path
from math import gcd
import argparse
import json
import numpy as np
from utils.py_utils.assets import sha256_file
from utils.py_utils.converted_featuremap import Runner, select
from .asr import decode_logits
from .cli import load_vocabulary
from .public_preprocess import SAMPLES, feature_frames, prepare_waveform

SAMPLE = Path(__file__).resolve().parents[2]
VARIANT = 'xlsr53-zh3503-30k-mask-r02'


def audio_windows(path):
    """Read bounded real audio blocks and explicitly resample each to 16kHz."""
    import soundfile as sf
    from scipy.signal import resample_poly
    with sf.SoundFile(path) as stream:
        rate = int(stream.samplerate)
        if rate <= 0 or stream.frames <= 0:
            raise ValueError('Expected a nonempty audio file with a valid sample rate')
        # Floor keeps the resampled block within the fixed graph window.
        size = max(1, SAMPLES * rate // 16000)
        start = 0
        while True:
            raw = stream.read(size, dtype='float32', always_2d=True)
            if not len(raw): break
            if not np.isfinite(raw).all(): raise ValueError('Nonfinite audio')
            mono = raw.mean(axis=1)
            if rate != 16000:
                factor = gcd(rate, 16000)
                mono = resample_poly(mono, 16000//factor, rate//factor).astype(np.float32)
            if not 0 < len(mono) <= SAMPLES:
                raise ValueError('Resampled chunk exceeds the fixed source contract')
            yield mono, {'source_start': start, 'source_samples': len(raw), 'source_rate': rate}
            start += len(raw)


def transcribe(path, runner, vocabulary, *, raw_dir=None):
    """Decode all valid frames, retaining complete raw logits when requested."""
    records = []
    for index, (mono, source) in enumerate(audio_windows(path)):
        frames = feature_frames(len(mono))
        row = {**source, 'index': index, 'valid_samples': len(mono), 'valid_frames': frames}
        if frames == 0:
            # No receptive field can fit, but these source samples are accounted for.
            row.update(text='', inference='no-valid-feature-frame')
        else:
            outputs = runner(prepare_waveform(mono))
            if set(outputs) != {'logits'}:
                raise ValueError('Expected the complete CTC logits head')
            logits = outputs['logits']
            if (logits.shape != (1, 93, 3503) or logits.dtype != np.float32
                    or not np.isfinite(logits).all()):
                raise ValueError('Expected all finite F32 [1,93,3503] raw logits')
            row.update(text=decode_logits(logits[:, :frames], vocabulary, mode='ctc'), inference='completed')
            if raw_dir is not None:
                raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / f'chunk-{index:06d}.npz'
                np.savez(raw_path, logits=logits)
                row['raw_logits'] = str(raw_path)
        records.append(row)
    return {'text': ''.join(r['text'] for r in records), 'chunks': records,
            'source_samples': sum(r['source_samples'] for r in records),
            'scope': 'Independent 30000-sample windows; no overlapping context or language-model decoding'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=['s100'], default='s100')
    p.add_argument('--variant', default=VARIANT)
    p.add_argument('--asset-id'); p.add_argument('--model-path', type=Path)
    p.add_argument('--audio-file', type=Path, default=SAMPLE/'test_data/chi_sound.wav')
    p.add_argument('--vocab-file', type=Path, default=SAMPLE/'test_data/vocab.json')
    p.add_argument('--backend', choices=['board', 'source-onnx'], default='board')
    p.add_argument('--source-onnx', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/asr-public'))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    selection = select(SAMPLE, a.target, a.variant, asset_id=a.asset_id, model_path=a.model_path)
    if a.dry_run:
        print(json.dumps({'asset_id': selection.asset.reference, 'profile': selection.profile})); return 0
    report = transcribe(a.audio_file, Runner(selection, backend=a.backend, source_onnx=a.source_onnx),
                        load_vocabulary(a.vocab_file), raw_dir=a.output_dir/'raw')
    report.update(backend=a.backend, asset_id=selection.asset.reference,
                  audio_sha256=sha256_file(a.audio_file), variant=a.variant)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir/'transcript.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
