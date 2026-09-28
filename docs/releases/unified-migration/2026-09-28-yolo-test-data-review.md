# Ultralytics test-data documentation gap — Codex review

H2-DATA-R1: test_data contains 11 tracked images/label tables but no README in
either language. The root explains selected images and links to runtime/evaluator,
but directory-local readers cannot distinguish inputs, display labels and
historical result illustrations. The user's requested systematic subdirectory
documentation requires that distinction, especially for an Agent consuming this
folder directly. H2 stays open pending a bounded documentation correction.

Add bilingual test_data guides, link them from root directory navigation, describe
actual file roles, image loading expectations, custom-image CLI inputs and output
locations, and separate per-runtime display labels from evaluation annotations
or ImageNet synset ground truth. Preserve DOTA's actual class-name ordering (not
invented numeric category IDs). Label source figures historical, never fresh
refactor evidence. Use source provenance already recorded in independent depth
reviews and pinned source paths. Keep all existing files byte-identical and all
existing commands unchanged. No downloads, inference or quantization execution.
