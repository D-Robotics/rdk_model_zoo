# HIMLoco author completion audit and sample navigation correction

Base: `f96533a326285088c64f2c933abcdd887b652530`.
This is an author audit, not an independent review or a board acceptance record.

## Scope accounting

| Source capability | Unified implementation / documentation | Evidence and boundary |
| --- | --- | --- |
| 270-to-12 fused Go2 policy | Python policy stages, named shared runner; C++ policy and libdnn adapter | 21 source observation comparisons; host ownership/stage checks; SDK doubles, not hardware |
| Source-indexed offline runtime | Python main/application; native main/cli_io/application | Successful/failed reports, source indices, exact byte lengths, hashes, output preservation and metadata; 23 current HIMLoco tests |
| Model preparation | model/download.py + download_model.sh | Explicit shared hash/atomic preparation; previews checked; actual download not-run |
| Native build and library use | CMake, launcher.py/run.sh; separate policy and SDK types | CMake Release core/CTest checked; source SDK APIs compiled against explicit fixture headers, real vendor SDK not-run |
| Export, calibration, Mapper | Three conversion scripts retained byte-for-byte | Trusted source scheme; bilingual commands/options/outputs reorganized, no recipe execution per user instruction |
| Numerical evaluation and input preparation | Four evaluator scripts retained byte-for-byte | Original metrics, thresholds, commands and historical tables retained; no new model-evaluation claim |
| Test observations and provenance | 21 BINs plus runtime-input-manifest.json | Source bytes retained; both runtimes consume the same prepared float contract |
| Documentation / license / provenance | Seven bilingual levels: root, model, Python, C++, conversion, evaluator, test_data | Source models, geometry, environments, historical data, commands, parameters, outputs and limitations retained |

Detailed implementation/test evidence remains in the linked core, Python CLI,
conversion/evaluator, C++ core, SDK and native CLI records in the migration ledger.
Existing 23-test and CMake records bind the current code: this audit changes only
README/navigation and accounting. No redundant numerical or toolchain rerun was
performed.

## Findings corrected in this audit

1. Root and sample indexes omitted Paraformer and HIMLoco and still described
   robotics as wholly pending. Both languages now index all 49 current sample
   roots (45 vision, three speech, one robotics), with conversion/evaluation links.
2. HIMLoco Python troubleshooting still said native runtime/evaluation migration
   was in progress. It now links the implemented guides. The sample introduction
   states provided functionality and actual verification scope instead of repeating
   internal intermediate acceptance prose.

[Navigation evidence](evidence/2026-09-28-b10-himloco-author-audit/navigation.json)
checks every relative file target in the four top-level/index guides and all 14
HIMLoco guides, and checks exact bilingual index membership against the actual
sample directories. [Sample contract](evidence/2026-09-28-b10-himloco-author-audit/contract.json)
reports zero violations, one ordinary CLI policy skip and zero exemptions.

## Disposition

HIMLoco Mapping/Refactor/Docs are author-complete; Host passes the explicitly
recorded scope. Review remains not-run and Closed=no pending the final fresh
whole-branch review. Board execution and real SDK ABI/model behavior are unverified.
Quantization recipe reruns are explicitly outside completion requirements.
H0–H9 is not closed: B11 LLM/VLA, cross-repository resources/shared review and the
final comprehensive verification/independent review remain required work.
