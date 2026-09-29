# B10 Paraformer real CPU frontend — source equivalence

Status: real host frontend implemented and verified; complete sample, native C++,
conversion/evaluator integration and whole-branch review remain open. No HBM, SDK,
board, OE, dataset CER or model-latency execution is claimed.

## Preserved features and corrected side effect

`frontend.py` wraps the actual FunASR WavFrontend using source settings: 16 kHz,
80 mel bins, hamming/25 ms/10 ms, LFR 7/6, fixed CMVN and seed 191009. File reading
stays outside. Inputs are finite float32 mono or multichannel samples, averaged
like the source, without implicit resampling or waveform normalization. Short
windows follow FunASR's own behavior, including the verified 10 ms case; no
arbitrary 25 ms minimum is imposed.

The source reset Torch's global RNG before each call. The new adapter scopes only
the CPU RNG, restores it on success/error, and serializes its own calls. It does
not seed GPU generators or promise coordination with unrelated threads consuming
global RNG. An injected backend error after random-number consumption verifies
exception restoration. All normal feature checks use actual dependencies, not a
fake frontend or substituted numerical implementation.

The fixed first-400-frame behavior is preserved, with explicit original length,
valid length and truncation in `PreparedFeatures`. Thirty seconds produced 500
frames; only the first 400 are returned and marked truncated. No chunking/VAD,
streaming, timestamp or hotword behavior is invented. The caller must surface
truncation when the complete CLI is integrated.

## Actual environment and comparisons

A separate Python 3.12.14 macOS arm64 environment was installed outside the tracked
repository. Source direct versions were available: Torch/torchaudio 2.6.0,
FunASR 1.3.14, SoundFile 0.14.0, NumPy 1.26.4 and protobuf 4.23.0. The full observed
[dependency list](evidence/2026-09-28-b10-paraformer-frontend/host-requirements.txt)
and install log are retained. This is not a board environment lockfile.
FunASR reports ffmpeg absent; SoundFile loads the arrays and ffmpeg is not needed.

The original frontend classes are compiled unchanged from byte-checked pinned
S source, excluding the unrelated eager board-runtime import. Seven cases match
**byte for byte**, including repeated calls:

| Input | Original → valid frames | Truncated |
| --- | --- | --- |
| BAC009S0724W0121 | 71 → 71 | no |
| BAC009S0724W0168 | 78 → 78 | no |
| derived stereo | 71 → 71 | no |
| 0.5 s silence | 8 → 8 | no |
| 30 s derived audio | 500 → 400 | yes |
| 25 ms single window | 1 → 1 | no |
| 10 ms short window | 1 → 1 | no |

Both WAVs and the reference manifest are copied unchanged into unified test_data.
The verifier checks WAV bytes against the pinned source, and document checks also
verify the manifest. Waveform unit tests cover owned mono/stereo preparation and
invalid rate/dtype/empty/nonfinite geometry without loading optional frontend deps.
The full sample suite has 26 tests; shared regression has 156 tests.

[Machine-readable real results](evidence/2026-09-28-b10-paraformer-frontend/real-summary.json)
include source/audio/feature hashes, RNG results and package versions. From the
repository root, using the documented real frontend environment:

```bash
python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-frontend/verify_real.py
```

## Documentation and remaining work

Runtime and model guides now describe real frontend usage, dependency setup,
input/output contracts, truncation, RNG scope and meaningful validation limits.
Test-data guides retain actual reference text and source identity, provide host
verification commands and do not copy the obsolete source `run.sh` promise.
Host examples are actually executed with the required environment. Dependency
installation commands are not repeatedly replayed by documentation verification;
the retained installation record establishes what was installed. Prior evidence
verifiers are limited to their own original README command blocks so newer setup
instructions cannot unexpectedly install dependencies during a historical recheck.

Remaining: complete safe audio/manifest CLI and reporting, C++ frontend bridge and
native source migration, conversion/calibration scripts sharing CIF, evaluator,
and remaining root/conversion/evaluator bilingual documentation. Full branch
acceptance and H0–H9 remain open; board tests remain deferred by user instruction.
