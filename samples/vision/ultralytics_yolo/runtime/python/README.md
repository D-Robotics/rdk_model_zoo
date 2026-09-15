# Python runtime

Run `python main.py --help` for all parameters. `--platform x5|s100|s100p|s600` chooses the artifact/input profile; `--family`, `--task detect|seg|pose|cls` and `--model-size` select a published combination. Paths resolve relative to this sample, independently of the working directory. User-supplied image/model/output paths are relative to the caller.

The canonical classes are YoloDetect, YoloSeg, YoloPose and YoloCls; pass `platform=resolve_platform(...)` in their config. S YOLOv10 uses YoloV10Detect. Pose returns `(boxes, scores, class_ids, xy, confidence)`; the legacy X5 class retains its three-result tuple. `--classes-num`, comma-separated `--strides` and segmentation `--mc` remain explicit configuration. Changing these cannot make an incompatible compiled output compatible.
