# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""One-time host preparation of the hardware background asset.

The bundled ``bg_video.mp4`` is MPEG-4 Part 2, which the RDK video decoder
does not support.  This script transcodes it once, on any host with FFmpeg,
into the hardware-decodable H.264 elementary stream the runtime reads, and
writes the ``bg_video.json`` sidecar next to it.  Every sidecar field is
measured from the files on disk — the frame count comes from
``ffprobe -count_frames`` on the produced stream and the SHA-256 from
hashing it — never from assumptions about the input.

The original ``bg_video.mp4`` is only ever read.  The script needs nothing
beyond the Python standard library plus ``ffmpeg``/``ffprobe``.

Run it from the repository root (or anywhere, with explicit paths):

    python3 samples/vision/lite_avatar/test_data/prepare_background.py \
        --data samples/vision/lite_avatar/test_data/preload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

#: Default data directory: the preload directory this script lives next to.
DEFAULT_DATA = Path(__file__).resolve().parent / "preload"


def _probe(ffprobe, media, entries, *, count_frames=False):
    command = [ffprobe, "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        command.append("-count_frames")
    command += [
        "-show_entries", f"stream={entries}", "-of", "json", str(media),
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(
            f"ffprobe failed on {media}: {result.stderr.strip()}"
        )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise SystemExit(f"No video stream found in {media}")
    return streams[0]


def _fps(stream, media):
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not rate or rate in ("0/0", "N/A"):
        raise SystemExit(f"Could not determine the frame rate of {media}")
    value = Fraction(rate)
    if value <= 0:
        raise SystemExit(f"Invalid frame rate {rate!r} for {media}")
    return value


def prepare(source, data_dir, *, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    """Transcode ``source`` and write ``bg_video.h264`` + ``bg_video.json``.

    Returns the sidecar dictionary (also written to disk).
    """

    source = Path(source)
    data_dir = Path(data_dir)
    if not source.is_file():
        raise SystemExit(f"Background source not found: {source}")
    if shutil.which(ffmpeg) is None or shutil.which(ffprobe) is None:
        raise SystemExit("This script needs ffmpeg and ffprobe on PATH")

    # Geometry and cadence of the source drive the transcode; the sidecar's
    # fields are then re-measured on the produced stream below.
    src = _probe(ffprobe, source, "width,height,avg_frame_rate")
    width, height = int(src["width"]), int(src["height"])
    if width % 2 or height % 2:
        raise SystemExit(
            f"Background source is {width}x{height}; even dimensions are "
            "required (the hardware codec codes NV12)"
        )
    fps = round(float(_fps(src, source)))
    if fps < 1:
        raise SystemExit(f"Background frame rate rounds to {fps}; need >= 1")
    # The decoder's NV12 luma stride is align-16, so the prepared stream is
    # coded at the next 16-pixel multiple (the bundled asset's 890 -> 896).
    # The encoder's separate 8-pixel width contract is applied at encode
    # time and cropped again on remux.
    coded_width = (width + 15) // 16 * 16

    elementary = data_dir / "bg_video.h264"
    sidecar_path = data_dir / "bg_video.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
        "-map", "0:v:0",
    ]
    if coded_width != width:
        command += ["-vf", f"pad={coded_width}:{height}:0:0:black"]
    command += [
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
        "-pix_fmt", "yuv420p", "-x264-params", "bframes=0:ref=3",
        "-crf", "14", "-r", str(fps), "-an",
        "-colorspace", "smpte170m", "-color_primaries", "smpte170m",
        "-color_trc", "smpte170m", "-color_range", "tv",
        "-f", "h264", str(elementary),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg transcode failed: {result.stderr.strip()}")

    # Measure the produced stream itself: the sidecar describes what is on
    # disk, not what the command intended.
    out = _probe(
        ffprobe, elementary,
        "codec_name,width,height,avg_frame_rate,nb_read_frames",
        count_frames=True,
    )
    if out["codec_name"] != "h264":
        raise SystemExit(f"Produced stream is {out['codec_name']}, not H.264")
    got_geometry = (int(out["width"]), int(out["height"]))
    if got_geometry != (coded_width, height):
        raise SystemExit(
            f"Produced stream is {got_geometry[0]}x{got_geometry[1]}, "
            f"expected {coded_width}x{height}"
        )
    frames = int(out["nb_read_frames"])
    if frames < 1:
        raise SystemExit("Produced stream holds no decodable frames")
    digest = hashlib.sha256(elementary.read_bytes()).hexdigest()
    # fps is the cadence measured on the source and enforced by ``-r``:
    # a raw H.264 elementary stream carries no container timing, and
    # ffprobe reports a placeholder there (measured 25/1 for streams
    # produced at 30, including the sample's real asset).
    sidecar = {
        "codec": "h264",
        "width": width,
        "coded_width": coded_width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "sha256": digest,
    }
    sidecar_path.write_text(
        json.dumps(sidecar, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"prepared {elementary}\n"
        f"  visible {width}x{height} in coded {coded_width}x{height}, "
        f"{sidecar['fps']} fps, {frames} frames\n"
        f"  sidecar {sidecar_path} (sha256 {digest})"
    )
    return sidecar


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare the hardware-decodable background asset "
        "(one-time host transcode; the original .mp4 is never modified)"
    )
    parser.add_argument(
        "source", nargs="?", default=None,
        help="background source video (default: <data>/bg_video.mp4)",
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA),
        help="avatar data directory to prepare (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    data_dir = Path(args.data)
    source = Path(args.source) if args.source else data_dir / "bg_video.mp4"
    prepare(source, data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
