# Shared dataset README independent review — 2026-09-28

Status: **changes-required**, bounded H8 dataset documentation package. Author implementation remains unaccepted; H8 stays open. Reviewer: Codex; implementer: local Claude Code + GLM.

## Findings

### DATASET-R1 — DOTA annotation identity is misstated (P2)

Both `datasets/dotav1/README{,_cn}.md` call the bundled class-list order the official annotation order and assign global dataset category IDs 1–15. The root dataset guides repeat the one-based-ID claim. The [official annotation specification](https://captain-whu.github.io/DOTA/dataset.html) describes eight coordinates, category and difficulty; native DOTA labels use category names. A COCO conversion may choose numeric IDs, but the mapping belongs to that conversion, not a universal DOTA contract. Retain the useful warning that the two local class orders differ, name each actual file order, and require an explicit mapping for any converted dataset. Also replace “mislabels every category”: `plane` at index 0 is shared by both files (evidence records this counterexample). Correct author evidence/report claims alongside both language pairs.

### DATASET-R2 — COCO acquisition example is not covered by ignore rules (P2)

Both COCO guides suggest invoking `bash download_full_coco.sh` from `datasets/coco`, producing `datasets/coco/coco_full/`, but explain ignore handling as if only running elsewhere needs special care. Current `.gitignore` excludes the direct `datasets/coco/val2017/*` / `annotations/*` layout, not this script's nested layout. `git check-ignore datasets/coco/coco_full/train2017/example.jpg` returns 1, while the direct val2017 comparison returns 0. Correct README instructions without modifying the script or downloading data: either use an explicit out-of-checkout working directory or explain the local exclusion needed before running the documented command. Root guide must not imply downloads are automatically excluded.

### DATASET-R3 — Label-guide navigation and format boundaries (P2)

`datasets/PascalVOC/README.md` language switch points 简体中文 to `./README.md`, looping back to English; point it to the CN file. The ImageNet guide says WordNet synset words are embedded in display strings and broadly equates both label formats wherever `--label-file` appears. The classification evaluator's same-named argument instead expects ordered `n########` synset IDs (see its dataset section); the dict display-name file cannot be passed there. Clearly distinguish runtime display names, human-readable synonyms, and evaluator synset IDs/ground truth, including separate flag semantics. Avoid guaranteeing that any model must classify the bundled zebra correctly; describe it as a smoke input with an expected class checked against each model's evidence.

## Evidence and closure

`evidence/2026-09-28-datasets-independent-review/findings.json` records reviewed document hashes, DOTA ordering counterexample, exact ignore results and the official source. These checks do not run download or quantization recipes. Closure requires corrected bilingual documents and author claims, independent reread of the changed semantics, language-link and command/anchor checks. No broad H8 acceptance follows from this bounded review.

## Rereview follow-up

R1 DOTA identity and R3 language/display-label distinction are corrected in both languages. R2 now correctly states the nested COCO output is not ignored and recommends out-of-checkout acquisition. One command detail remains: the optional `echo ... >> .git/info/exclude` is relative to cwd and assumes `.git` is a directory, whereas the guide previously selects `datasets/coco` and managed worktrees have a `.git` file. Use `git rev-parse --git-path info/exclude` to resolve the real local exclude file from the checkout, or omit the optional command and retain the recommended out-of-checkout path. A read-only query from `datasets/coco` in this actual worktree resolves the primary repository `.git/info/exclude`; no exclusion was modified. Two-file follow-up assigned to Claude Code + GLM before acceptance.

## Final independent disposition

**DATASET-R1/R2/R3 closed; bounded shared dataset README package accepted.** The final COCO inline example uses the Git-resolved exclude path in both languages; it was inspected without changing ignore files or downloading data. Semantic reread confirms native DOTA category names versus conversion-owned IDs, local 15-class ordering distinction, runtime display labels versus evaluator synset ground truth, fixed PF vocabulary, actual download cwd/output behavior and source provenance.

Fresh static checks (`final-checks.json`) cover all 16 README local links/explicit anchors, six bilingual structures, executable dataset-guide command parity, and the affected YOLO/YOLOE sample checkers (zero violations; pre-existing 2/1 policy skips). `resource-and-command-check.json` proves all 11 non-README dataset resources remain byte-identical to X5 pin ac11571 and all evaluator fenced commands are unchanged (7 blocks per Ultralytics language, 4 per YOLOE language). No code/scripts/labels/images/manifest assets were changed. No actual acquisition, inference, board or quantization recipe was run. This closes the dataset-doc sub-item, not all H8.
