# Review fixes DOC-B1B2-R1 / DOC-B1B2-N1 (2026-09-28, remediation record)

Scope of this fix: exactly four prose hunks (2 captions + 2 background
sentences) in mobilenetv3 README.md/README_cn.md and efficientformerv2
README.md/README_cn.md, plus this record and the author report update.
No commands, tables, matrix rows, image files, or other samples touched.

## DOC-B1B2-R1 — MobileNetV3 SE gate position (both languages)

Figure re-inspected before fixing: `samples/vision/mobilenetv3/test_data/MobileNetV3_architecture.png`
(sha256 `bc978181…`, unchanged). The SE branch in the figure is
Pool → FC ReLU → FC hard-σ → ⊗, and the ⊗ arrow lands on the slab after
"NL, Dwise 3×3" — i.e., the gate modulates the expanded channels before
the final "NL, 1×1" projection, not the 1×1 output.

Before (EN caption, excerpt): "…a global pool plus
FC-ReLU / FC-hard-sigmoid gates the 1×1 output, and the non-linearity
(NL) is chosen per layer."

After (EN caption, excerpt): "…after the NL depthwise 3×3, a global pool
plus FC-ReLU / FC-hard-sigmoid gate modulates the expanded channels, and
the gated result passes through the final NL 1×1 projection (the
non-linearity is chosen per layer)."

Before (CN caption, excerpt): "…全局池化加 FC-ReLU / FC-hard-sigmoid 门控
1×1 输出，非线性（NL）按层选择。"

After (CN caption, excerpt): "…在 NL depthwise 3×3 之后，全局池化加
FC-ReLU / FC-hard-sigmoid 门控作用于扩展通道，门控结果再经最后的 NL 1×1
投影输出（非线性按层选择）。"

## DOC-B1B2-N1 — EfficientFormerV2 overgeneralization (both languages)

Before (EN): "…unified feed-forward networks, improved MHSA
(talking-head attention with locality), and attention at higher
resolutions with cheaper downsampling cut the cost that kept earlier
hybrids off mobile devices ([paper](…), [snap-research/EfficientFormer](…))."

After (EN): "…unified feed-forward networks, improved MHSA
(talking-head attention with locality), and attention at higher
resolutions with cheaper downsampling reduce the attention and
downsampling overhead relative to the EfficientFormer baseline while
keeping MobileNet-level size and speed ([paper](…),
[snap-research/EfficientFormer](…))."

Before (CN): "…统一 FFN、改进的 MHSA（带 locality 的 talking-head 注意
力），以及在高分辨率上的注意力与更低开销的下采样，削掉了早期混合架构无法
登上移动设备的成本（[论文](…)、[snap-research/EfficientFormer](…)）。"

After (CN): "…统一 FFN、改进的 MHSA（带 locality 的 talking-head 注意
力），以及在高分辨率上的注意力与更低开销的下采样，相对 EfficientFormer
基线降低了注意力与下采样开销，同时保持 MobileNet 量级的尺寸和速度
（[论文](…)、[snap-research/EfficientFormer](…)）。"

The overgeneralizing claim ("earlier hybrids could not run on mobile
devices") is gone; the same-package EfficientFormer root README already
carries mobile-device (iPhone 12/CoreML) measurements, so the replacement
does not contradict maintained content.

## Verification for this fix

- `tools/sample_contract/check.py --sample` for mobilenetv3 and
  efficientformerv2: 0 violations each (output appended to
  `checker-after-review-fix.txt`).
- `git diff --check`: rc=0.
- Fenced code blocks in the four touched files remain identical to HEAD
  (only prose captions/sentences changed; verified by the same
  fenced-block extraction as `command-blocks-and-diff-evidence.txt`).
- Note on diffing: this whole package is uncommitted, so the HEAD→worktree
  diff for these two samples contains no removed lines at all — both
  corrections edited text that this same package had added. The exact
  before/after of the corrected sentences is therefore recorded verbatim
  above rather than read from `git diff`.
