# B10 source capability and contract audit

Base: `c268a18d`. Source audit only; canonical B10 implementation, host acceptance
and independent review remain pending. No board, SDK or robot was accessed.
H0–H9 remains the controlling full scope.

## Sources and completeness

The [reproducible audit](evidence/2026-09-28-b10-source/audit.py) and
[results](evidence/2026-09-28-b10-source/audit.json) inventory 132 files against
X5 `ac115717197920355fc390bb04299b20e6436864` and S
`380e1a2bf42041af54be6f34935e50197cfadff9`:

| Sample | Files | Capabilities to preserve | Publication scope |
| --- | ---: | --- | --- |
| HIMLoco | 52 | Python/C++ offline fused policy, export, calibration, Mapper, JIT/ONNX/action comparison, 21 held-out inputs | One X5 BIN, publisher SHA available |
| ASR | 22 | Python/C++ chunked audio, resampling, normalization, token decoding, bundled vocabulary/audio and historical plots | S100 and S600; no S100P asset |
| KWS | 15 | Python MDTC audio/fbank/max-score pipeline, bundled wake-word audio, historical perf/functional records | S100 only |
| Paraformer | 43 | Python/C++ encoder → predictor → CPU CIF → decoder, FunASR frontend, 11 conversion scripts plus CIF, three configs, WAV manifest | S100: three HBM, downloaded tokens, two bundled config/statistics files |

Every snapshot file matches the pinned source bytes. HIMLoco's 21 input binaries
already exist in Git and each matches the manifest's 1080-byte length and SHA.
An initial progress message incorrectly inferred their absence from a search
that omitted binary files; this audit corrects that statement. No restoration or
source mutation was necessary. The four actual WAV headers and hashes are also
recorded. These are source/input facts, not runtime results.

## Findings that constrain the migration

### KWS

- `THRES=60000` is 3.75 seconds at 16 kHz, not the source comment's 60 seconds.
  The bundled clip is mono PCM16, 40000 samples at 16 kHz (2.5 seconds).
- The wrapper docstring claims S600, while both the active manifest and sample
  support table only publish S100. Target selection must reject S600/S100P/X5.
- Source preprocessing loads audio and invokes PaddleAudio fbank in the inference
  file. Move audio I/O/frontend helpers out; preserve the actual PaddleAudio
  feature contract rather than assume a compatible replacement. Source prose
  says `[1,1,T,80]`, but code merely adds one dimension to the frontend result;
  determine actual shape through frontend execution and SDK metadata validation.
- Source postprocessing uses max of model output, without computing sigmoid.
  Preserve max-score semantics, validate output precision/range and avoid a
  second sigmoid. `output_quants` is read but unused: metadata must determine
  whether the runtime already supplies floats before treating values as scores.
- Conversion has no executable recipe. Describe the missing source/checkpoint/
  compiler prerequisites honestly; do not invent a reproducible conversion.
  Historical 1.176 ms / 830.875 FPS and approximately 0.985 sample score remain
  labeled source measurements. A score is not dataset precision/recall.

### ASR

- The source claims greedy CTC, but both implementations concatenate every
  argmax token and remove `<pad>` without repeat collapse. Executing the unchanged
  Python method on IDs `[1,1,0,1,2,2]` produces `aaabb`; standard blank-0 CTC would
  produce `aab`. Migration must make decoding policy explicit and test repeats
  both adjacent and separated by blank. Any correction is a disclosed behavior
  change, not falsely labeled byte-equivalent source migration.
- Python uses SciPy Fourier resampling; native uses libsamplerate sinc. Their
  waveforms are not inherently equal. Python normalization divides by
  `sqrt(var+1e-5)`; C++ divides by standard deviation when greater than 1e-9 and
  otherwise leaves the samples untouched. Constant nonzero input therefore
  differs as well. The canonical contract and cross-language tolerance need an
  explicit decision and actual fixtures, not a generic equivalence claim.
- Input is independently chunked to 30000 samples at 16 kHz, with source-rate
  reads rounded up and final padding. Preserve and document chunk boundaries;
  it is not evidence of a stateful streaming acoustic model.
- Conversion README is a placeholder; retain this limitation. Preserve source
  accuracy/performance figures with original context and separate new metrics.

### Paraformer

- The unchanged standalone CIF raises `IndexError` for all-zero `[1,401]`
  alphas and `[1,401,512]` hidden states with `real_T=400`. The runtime duplicates
  the same zero-fire indexing. Fix the empty result to zero acoustic embeddings
  and zero tokens, with a regression and preserved nonempty numerical behavior.
- CIF must mask padding after the real LFR frame count. Move its pure numerical
  implementation out of the model wrapper and reuse it for runtime/calibration;
  preserve encoder/predictor/CIF/decoder order. The three models cannot be
  flattened into one raw inference call without changing the deployment.
- Source `set_scheduling_params` silently discards both arguments. The canonical
  runner must apply supported settings or reject unsupported requests explicitly.
- The frontend sets the global Torch seed for fbank dithering. Keep reproducible
  frontend behavior without unintended cross-request/global RNG side effects;
  test the seed boundary. Preserve fixed `[1,400,560]` features, valid-frame count
  and the source limit of 100 decoder tokens, explaining truncation.
- Native starts from WAV through an explicit Python frontend bridge. Keep the
  customer WAV/manifest flow; do not present temporary NPY files as the only
  supported public input. Source wrappers implicitly create environments,
  download and build; canonical preparation must be explicit.

### HIMLoco

- Preserve the fused `[1,270]` float observation history → `[1,12]` float action
  contract, source-indexed 21 inputs, complete conversion and evaluator scripts.
  Source historical runtime timing scopes differ between Python and C++ and
  must stay separately labeled; no new hardware performance is established.
- Move metadata/dtype/shape, scheduling, timing, file I/O and report helpers out
  of the task module. Keep construction/preprocess/forward/postprocess/predict
  readable and reusable; retain original file/shape/checksum strictness.
- This sample remains offline inference and numerical evaluation. No robot
  command transport or live actuation is required or introduced.

## Implementation order and acceptance

Continue KWS, ASR, Paraformer and HIMLoco with common model/identity/asset runners
where their tensor contracts truly match. Provide complete bilingual root,
model, runtime, conversion, evaluator and test-data explanations; native
instructions where an original native implementation exists. Keep source
references, licenses, original measurements and fixed inputs. For each sample,
execute host numerical/error-path tests, read-only commands, README examples and
links; record unavailable real SDK/OE/dataset checks separately. No board results
are inferred from fixtures. This audit starts B10; it does not close any sample.
