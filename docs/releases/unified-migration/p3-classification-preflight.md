# P3 classification preflight — limited source inspection

Date: 2026-09-16. This is a preparation note, not implementation acceptance.
The inspected legacy files are unchanged from
`cd74a2b241075bb21036d8d0855d0403f8e8c963`. No EfficientNet or MobileNet
artifact metadata, board inference, accuracy or performance was collected
for this note. P1 ResNet18 results do not certify these model families.

The existing manifests also encode different EfficientNet families: X5 has
`EfficientNet_B2/B3/B4_224x224_nv12.bin`, while S has EfficientNet Lite0–4 at
224/240/260/300/380 input sizes for S100 and S600. Same sample directory name
does not mean matching architecture, resolution or interchangeable weights.
MobileNetV1 has one X5 asset and separate S100/S600 assets; neither inspected
manifest has an S100P row for these two sample IDs.

## First source differences to preserve

| Source | Relevant symbols and behavior | Consequence for migration |
| --- | --- | --- |
| [X5 EfficientNet](../../../platforms/x5/samples/vision/efficientnet/runtime/python/efficientnet.py) | `EfficientNetConfig.resize_type=1`; `pre_process` uses LINEAR for direct resize and packed NV12; `post_process` applies SciPy softmax, then `np.argsort(prob)[-topk:][::-1]`; returns IDs, scores and labels | Reuse actual classification stages only with explicit interpolation, score and return policies |
| [S EfficientNet](../../../platforms/s/samples/vision/efficientnet/runtime/python/efficientnet.py) | `EfficientNetConfig.resize_type=1`; `pre_process` supplies Y/UV; `post_process` calls `visualize.get_topk_predictions` | Its helper uses stable exponent normalization and `np.argsort(-probabilities)[:topk]`; tied classes and `topk=None` semantics need tests before merging |
| [X5 MobileNetV1](../../../platforms/x5/samples/vision/mobilenetv1/runtime/python/mobilenetv1.py) | `MobileNetV1Config.resize_type=0`; `pre_process` explicitly uses LINEAR; `post_process` directly ranks the squeezed output without softmax; returns IDs, scores and labels | Do not apply the ResNet18 pilot's `legacy_softmax` policy to this sample |
| [S MobileNetV1](../../../platforms/s/samples/vision/mobilenetv1/runtime/python/mobilenetv1.py) | `MobileNetV1Config.resize_type=1`; explicit `pre_process` resize override persists to `cfg.resize_type`; `post_process` ranks raw output with a default top-k of 5 and returns `(id, score)` pairs | Preserve target defaults, stateful legacy override and public return format through a deliberate compatibility mapping |

S's [resize helper](../../../platforms/s/utils/py_utils/preprocess.py)
defaults direct resize to NEAREST, while its letterbox path calls OpenCV's
default LINEAR interpolation. This is the same kind of distinction found
during P1, but must be checked independently for each consumer. The
[classification presentation helper](../../../platforms/s/utils/py_utils/visualize.py)
also owns mathematical softmax/top-k behavior in the old source; migration
must move that behavior to a task/protocol module rather than keep algorithm
ownership inside drawing utilities.

Source comments call MobileNetV1 output post-softmax probabilities. That is a
source-level statement, not artifact graph verification. The safe initial
contract is the observed legacy **identity score policy** until exact model
metadata and numerical/graph evidence are captured. Likewise, EfficientNet's
host softmax must not be inferred solely from its filename or tensor name.

## Bounded next steps

1. Select exact existing manifest rows and query their compiled input/output
   contracts on each target with an available artifact. Keep absent target
   assets separate from unknown hardware identity.
2. Capture unmodified source outputs and preprocessing for fixed textured,
   non-square inputs, with default and explicitly overridden resize modes.
   Include tie and top-k boundary fixtures as host tests.
3. Map public constructors, preprocessing, runtime calls, score policies,
   label loading, outputs, CLI and old import paths before implementation.
   P1 ResNet code is a reference boundary, not a universal implementation to
   copy or blindly parameterize.
4. Retain conversion, evaluation, native-language and dataset capabilities
   from the [complete inventory](x5-s-migration-map.md). Complete each sample
   with its own documentation, compatibility and validation evidence.

This note covers only these four Python source files and their two named S
helpers. Other classification families, ResNet50/152, YOLO classification and
video-action classification still require their own source/protocol review.
