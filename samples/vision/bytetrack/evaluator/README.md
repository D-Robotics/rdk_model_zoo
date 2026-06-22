English | [简体中文](./README_cn.md)

# ByteTrack Model Evaluation Guide

This directory records ByteTrack evaluation metrics, performance records, result checks, and tuning notes.

## MOT Metrics

ByteTrack is usually evaluated with MOT (Multiple Object Tracking) metrics, such as:

- `MOTA`: combines false negatives, false positives, and ID switches.
- `IDF1`: measures identity preservation.
- `FPS`: measures tracking system throughput.

The ByteTrack paper reports `80.3 MOTA` and `77.3 IDF1` on the MOT17 test set, with about `30 FPS` on a V100 GPU.

## Tracker Performance Record

On RDK S100, the ByteTrack tracker update costs about `2.37 ms` on average. Overall throughput mainly depends on the upstream YOLO detector.

## Result Check

Running `runtime/python/run.sh` generates `result.mp4`. Result validation should confirm:

- Pedestrians are tracked stably in the video.
- Each pedestrian box has a unique ID.
- The output video is not empty, and the frame count is the same as or close to the input video.
- Track IDs should not break frequently during short occlusions.

Reference MOT effects:

![MOT17-01-SDP](../test_data/readme_img/MOT17-01-SDP.gif)
![MOT17-07-SDP](../test_data/readme_img/MOT17-07-SDP.gif)

## Parameter Tuning

- `--score-thres`: YOLO detection confidence threshold, default `0.25`. Lower it if too few boxes are detected.
- `--track-thresh`: ByteTrack tracking threshold, default `0.3`. A high value filters out more low-score boxes.
- `--match-thresh`: IoU matching strictness, controlled by tracker configuration.
- `--track-buffer`: lost-track buffer length, controlled by tracker configuration.

For multi-class tracking, either maintain one `BYTETracker` per class or extend `STrack` to carry `class_id` and handle class information in `BYTETracker.update()`.

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
