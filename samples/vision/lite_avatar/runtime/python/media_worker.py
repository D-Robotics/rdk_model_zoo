# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""S-series hardware media worker: VPU encode subprocess and codec helpers.

Two responsibilities in one file (the sample's fifth runtime module):

* Helpers the renderer imports lazily on the board: the limited-range
  BT.601 BGR<->NV12 conversions and :func:`decode_background`, which reads
  the prepared background asset through the ``hobot_vio`` Decoder in file
  mode.  File mode is the SDK's own feeder: it reports the frame count on
  S100/S100P and loops the file after the last frame, so decoding stops
  after exactly the sidecar's frame count (verified against the file's
  SHA-256) and never relies on the loop for real frames.
* The ``__main__`` encoder-worker protocol behind the CLI's
  :class:`~samples.vision.lite_avatar.runtime.python.cli.HardwareVideoWriter`:
  frames arrive as indexes into a shared-memory ring, each consumed slot is
  acknowledged with ``ACK <index>`` so the renderer never overwrites a
  frame the encoder still owns, and FFmpeg remuxes the elementary stream
  with video copied (never re-encoded) after ``END``.

Color-space contract: limited-range BT.601 in both directions (the integer
matrices are the standard fixed-point ones and were measured against the
previous OpenCV ``COLOR_BGR2YUV_I420`` pipeline — same limited-range
convention, quantified in the sample docs).  The remux signals
``video_full_range_flag=0:matrix_coefficients=5`` and crops the replicated
pad columns used to reach the encoder's 8-pixel width multiple.

This module deliberately avoids OpenCV: image math is small NumPy integer
operations pinned by the host tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

#: One-time host-side preparation of the background asset (documented in the
#: sample README; the runtime never performs software video transcoding).
HOST_PREP_COMMAND = (
    "ffmpeg -i bg_video.mp4 -map 0:v:0 -vf pad=896:1920:0:0:black "
    "-c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p "
    "-x264-params bframes=0:ref=3 -crf 14 -r 30 -an "
    "-colorspace smpte170m -color_primaries smpte170m -color_trc smpte170m "
    "-color_range tv -f h264 bg_video.h264"
)

#: Soft per-frame wall-clock budget for the whole background decode.  It is
#: an elapsed deadline checked *between* the SDK's synchronous calls — a
#: single native call is not preempted (the SDK publishes no hard per-call
#: timeout).  The encoder side runs in the worker subprocess, where the
#: parent's ACK/CLOSE timeouts are the enforced bound.
CALL_TIMEOUT_S = 10.0


class BackgroundFormatError(ValueError):
    """Background asset is missing, legacy or inconsistent with its sidecar."""


# ======================================================================
# Limited-range BT.601 BGR <-> NV12 (integer; pinned by host tests).
# ======================================================================

def bgr_to_nv12(frame_bgr):
    """BGR uint8 ``[h,w,3]`` (even h, w) -> tight NV12 bytes.

    Y = (66R + 129G + 25B + 16*256 + 128) >> 8, clipped to [16,235];
    U = ((112B - 74G - 38R + 128) >> 8) + 128, V likewise, each averaged
    over the 2x2 source block and clipped to [16,240].
    """

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("bgr_to_nv12 expects an [h,w,3] BGR frame")
    h, w = frame_bgr.shape[:2]
    if h % 2 or w % 2:
        raise ValueError("bgr_to_nv12 needs even height and width")
    bgr = frame_bgr.astype(np.int32)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    y = np.clip((66 * r + 129 * g + 25 * b + 16 * 256 + 128) >> 8, 16, 235)
    u = ((112 * b - 74 * g - 38 * r + 128) >> 8) + 128
    v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128

    def subsample(plane):
        quad = plane[::2, ::2] + plane[::2, 1::2] + plane[1::2, ::2] + plane[1::2, 1::2]
        return np.clip((quad + 2) >> 2, 16, 240)

    y = y.astype(np.uint8)
    uv = np.empty((h // 2, w), np.uint8)
    uv[:, 0::2] = subsample(u)
    uv[:, 1::2] = subsample(v)
    return np.concatenate([y.reshape(-1), uv.reshape(-1)]).tobytes()


def nv12_to_bgr(nv12, width, stride, height):
    """Stride-padded NV12 -> BGR uint8 ``[height,width,3]`` (BT.601 limited).

    R = (298(Y-16) + 409(V-128) + 128) >> 8; G and B with the -100/-208
    and +516 coefficients, chroma repeated to full resolution.
    """

    buf = np.frombuffer(nv12, dtype=np.uint8)
    expected = stride * height * 3 // 2
    if len(buf) < expected:
        raise ValueError(
            f"NV12 buffer holds {len(buf)} bytes, needs {expected} "
            f"(stride {stride} x height {height} x 1.5)"
        )
    if width % 2 or height % 2:
        raise ValueError("nv12_to_bgr needs even width and height")
    if not 0 < width <= stride:
        raise ValueError(f"width {width} must be positive and <= stride {stride}")
    y = buf[: stride * height].reshape(height, stride)[:, :width].astype(np.int32)
    uv = buf[stride * height : expected].reshape(height // 2, stride)[:, :width]
    cb = np.repeat(np.repeat(uv[:, 0::2].astype(np.int32), 2, axis=0), 2, axis=1) - 128
    cr = np.repeat(np.repeat(uv[:, 1::2].astype(np.int32), 2, axis=0), 2, axis=1) - 128
    y2 = (y - 16) * 298
    out = np.empty((height, width, 3), np.uint8)
    out[..., 2] = np.clip((y2 + 409 * cr + 128) >> 8, 0, 255)
    out[..., 1] = np.clip((y2 - 100 * cb - 208 * cr + 128) >> 8, 0, 255)
    out[..., 0] = np.clip((y2 + 516 * cb + 128) >> 8, 0, 255)
    return out


def pad_right(frame, coded_width):
    """Edge-replicate ``frame`` up to ``coded_width`` columns (no-op if wide).

    Replaces ``cv2.copyMakeBorder(..., cv2.BORDER_REPLICATE)``; the pad
    columns only exist to satisfy the hardware encoder's 8-pixel width
    multiple and are cropped again by the remux.
    """

    if frame.shape[1] >= coded_width:
        return frame
    pad = coded_width - frame.shape[1]
    fill = np.repeat(frame[:, -1:], pad, axis=1)
    return np.concatenate([frame, fill], axis=1)


# ======================================================================
# Prepared background asset: sidecar validation + hardware file decode.
# ======================================================================

def read_background_sidecar(data_dir):
    """Validate ``bg_video.json`` and the elementary stream it describes.

    The SHA-256 recorded at preparation time is verified against the actual
    file, so a stale sidecar or a partially copied stream fails loudly
    instead of decoding through the feeder's silent loop.
    """

    data_dir = Path(data_dir)
    stream = data_dir / "bg_video.h264"
    sidecar_path = data_dir / "bg_video.json"
    legacy = data_dir / "bg_video.mp4"
    if not sidecar_path.is_file() or not stream.is_file():
        if legacy.is_file():
            # The prepared asset is what is missing; the .mp4's own codec is
            # not inferred from its extension here.  (The sample's bundled
            # bg_video.mp4 is known MPEG-4 Part 2 — documented in the README
            # — and is not hardware-decodable on RDK either way.)
            raise BackgroundFormatError(
                f"No prepared background in {data_dir}: expected bg_video.h264 "
                "with its bg_video.json sidecar, but only a bg_video.mp4 file "
                "is present (the sample's bundled bg_video.mp4 is MPEG-4 Part "
                "2, which the RDK video decoder does not support). Prepare the "
                "hardware-decodable H.264 background once on a host (never "
                f"overwriting the original):\n  {HOST_PREP_COMMAND}\n"
                "then place bg_video.h264 and bg_video.json in the avatar "
                "data directory (see the sample README)."
            )
        raise BackgroundFormatError(
            f"No background asset in {data_dir}: expected the prepared "
            "bg_video.h264 with its bg_video.json sidecar "
            "(host preparation command in the sample README)"
        )
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BackgroundFormatError(f"Invalid bg_video.json: {error}") from error
    required = ("codec", "width", "height", "fps", "frames")
    missing = [key for key in required if key not in sidecar]
    if missing:
        raise BackgroundFormatError(f"bg_video.json lacks {missing}")
    if sidecar["codec"] != "h264":
        raise BackgroundFormatError(
            f"Background codec {sidecar['codec']!r} is not hardware-"
            "decodable; re-run the host preparation for H.264"
        )
    width, height = int(sidecar["width"]), int(sidecar["height"])
    frames = int(sidecar["frames"])
    coded_width = int(sidecar.get("coded_width", width))
    if min(width, height, frames, coded_width) < 1 or width % 2 or height % 2:
        raise BackgroundFormatError(f"Invalid bg_video.json geometry: {sidecar}")
    if width > coded_width or coded_width % 16:
        # The decoder's NV12 luma stride is align-16: a stream coded at an
        # 8- but not 16-pixel multiple yields a tight buffer the SDK reads
        # with a truncated/corrupted UV plane.
        raise BackgroundFormatError(
            f"coded_width {coded_width} must be a 16-pixel multiple covering "
            f"the visible width {width} (the decoder's NV12 luma stride is "
            "align-16)"
        )
    # The digest is mandatory: without it, a replaced or partially copied
    # stream could decode through the feeder's silent loop to a wrong count
    # (exactly the failure mode the sidecar exists to close, and the only
    # cross-check on targets whose decode() reports no frame count).
    digest = sidecar.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in digest)
    ):
        raise BackgroundFormatError(
            "bg_video.json must record the prepared stream's SHA-256 as 64 "
            "hex characters; re-run the host preparation to regenerate "
            "bg_video.h264 with its sidecar"
        )
    actual = hashlib.sha256(stream.read_bytes()).hexdigest()
    if actual != digest.lower():
        raise BackgroundFormatError(
            f"bg_video.h264 SHA-256 {actual} does not match the sidecar "
            f"{digest}; the stream or the sidecar is stale — re-run the "
            "host preparation"
        )
    return {
        "stream": stream,
        "width": width,
        "coded_width": coded_width,
        "height": height,
        "fps": int(sidecar["fps"]),
        "frames": frames,
    }


def decode_background(data_dir, max_frames=150):
    """Hardware-decode the prepared background into BGR frames (list).

    File mode through the SDK's own feeder: every returned buffer must be a
    complete frame at the coded geometry; empty buffers are skipped a
    bounded number of times (the feeder emits one right after open), the
    count stops at exactly the verified sidecar's frame count — never at
    the feeder's silent end-of-file loop — and the decoder is *always*
    closed.  On S100/S100P the SDK reports the frame count itself and a
    mismatch with the sidecar is an error.  Close failures count: a close
    that fails after a successful decode propagates, and a failure during
    decode never masks itself behind the cleanup error (the decode error
    wins).  The soft decode budget is an elapsed deadline checked between
    synchronous SDK calls, not a hard per-call cap.
    """

    # Sidecar validation first: a legacy or stale asset must fail with the
    # host preparation guidance on any machine, before the board SDK import.
    spec = read_background_sidecar(data_dir)
    from hobot_vio import libsrcampy  # board SDK; host tests inject frames

    width = spec["width"]
    coded_width = spec["coded_width"]
    height = spec["height"]
    want = min(max_frames, spec["frames"])
    frame_bytes = coded_width * height * 3 // 2
    decoder = libsrcampy.Decoder()
    frames = []
    primary_error = None
    try:
        ret = decoder.decode(str(spec["stream"]), 0, 1, coded_width, height)
        if isinstance(ret, (tuple, list)):
            if int(ret[0]) != 0:
                raise BackgroundFormatError(
                    f"Hardware decoder open failed: {list(ret)}"
                )
            if len(ret) > 1 and int(ret[1]) not in (0, spec["frames"]):
                raise BackgroundFormatError(
                    f"Decoder reports {int(ret[1])} frames but the verified "
                    f"sidecar holds {spec['frames']}"
                )
        elif ret not in (0, None):
            raise BackgroundFormatError(f"Hardware decoder open failed: {ret}")
        empty_skips = 0
        deadline = time.monotonic() + CALL_TIMEOUT_S * (want + 2)
        while len(frames) < want:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Hardware background decode stalled at {len(frames)}/{want} frames"
                )
            payload = decoder.get_img()
            if payload is None:
                raise BackgroundFormatError(
                    f"Background decode returned no frame at {len(frames)}/{want}"
                )
            data = bytes(payload)
            if len(data) == 0:
                empty_skips += 1
                if empty_skips > 4:
                    raise BackgroundFormatError(
                        "Background decoder kept returning empty buffers"
                    )
                continue
            if len(data) != frame_bytes:
                raise BackgroundFormatError(
                    f"Background frame {len(frames)} is {len(data)} bytes, "
                    f"expected {frame_bytes} (stride {coded_width} x {height})"
                )
            stride = len(data) * 2 // (3 * height)
            frames.append(nv12_to_bgr(data, width, stride, height))
    except BaseException as exc:  # noqa: BLE001 - kept, close runs, re-raised
        primary_error = exc
    # Close always runs.  The SDK accepts 0/None as success; a nonzero
    # status or an exception is a cleanup failure.  A primary decode error
    # always wins over the cleanup error; on a successful decode the close
    # failure itself propagates.
    close_error = None
    try:
        status = decoder.close()
    except Exception as exc:  # noqa: BLE001 - reported as the close failure
        close_error = RuntimeError(f"Hardware decoder close failed: {exc!r}")
    else:
        if status not in (0, None):
            close_error = RuntimeError(
                f"Hardware decoder close failed with status {status!r}"
            )
    if primary_error is not None:
        raise primary_error
    if close_error is not None:
        raise close_error
    return frames


# ======================================================================
# Encoder worker subprocess (ring/ACK protocol, hardware H.264 only).
# ======================================================================

def drain_encoder_packets(encoder, output, tries=3):
    """Collect any late encoder packets after the last frame (bounded)."""

    for _ in range(tries):
        packet = encoder.get_img()
        if packet is None:
            return
        output.write(bytes(packet))


def remux_command(elementary, audio, output, *, repeat=1, pad=0):
    """FFmpeg remux: H.264 copied verbatim, WAV encoded to AAC.

    ``-shortest`` is deliberately absent, and a host test executes this
    command against a real FFmpeg to pin that.  The pipeline derives the
    frame count as ``floor(audio_len * fps / sample_rate)``, so the audio
    is never shorter than the video by more than one frame period and the
    mux needs no trim.  With ``-shortest``, FFmpeg >= 6 (measured on 6.1.1
    and 9.0.2) discards the whole audio stream with exit status 0: the
    copied video packets end first and the still-queued audio is dropped
    on flush, producing a silent audio-less MP4 (FFmpeg 4.4 instead muxed
    the audio cut at the video end).
    """

    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-fflags", "+genpts",
        "-r", "30", "-i", str(elementary),
        "-stream_loop", str(repeat - 1),
        "-i", str(audio), "-c:v", "copy",
    ]
    # Limited-range BT.601 luma/chroma (see bgr_to_nv12); crop the pad
    # columns the 8-pixel encoder width multiple required.
    metadata = "video_full_range_flag=0:matrix_coefficients=5"
    if pad:
        metadata += f":crop_right={pad}"
    command += ["-bsf:v", "h264_metadata=" + metadata]
    command += ["-c:a", "aac", "-movflags", "+faststart", str(output)]
    return command


def verify_remuxed_tracks(output, *, ffprobe="ffprobe"):
    """Delivery check on the finished MP4, at the demuxer level only.

    FFmpeg exits 0 from a mux that dropped a whole stream (the silent,
    audio-less S600 delivery), so ``DONE`` may only follow an actual look
    at the product: an H.264 video stream and an AAC audio stream, each
    holding at least one packet.  ``-count_packets`` reads container
    metadata — nothing here decodes frames.
    """

    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error", "-count_packets",
                "-show_entries", "stream=codec_type,codec_name,nb_read_packets",
                "-of", "json", str(output),
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "ffprobe (shipped with FFmpeg) is required to verify the "
            "delivered MP4; cannot confirm both tracks"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe could not read {output}: {result.stderr.strip()}"
        )
    packets = {}
    for stream in json.loads(result.stdout).get("streams", []):
        packets[stream.get("codec_type")] = (
            stream.get("codec_name"), int(stream.get("nb_read_packets", 0))
        )
    video = packets.get("video")
    audio = packets.get("audio")
    if video is None or video[0] != "h264" or video[1] < 1:
        raise RuntimeError(
            f"delivered MP4 holds no H.264 video packets: {packets}"
        )
    if audio is None or audio[0] != "aac" or audio[1] < 1:
        raise RuntimeError(
            f"delivered MP4 holds no AAC audio packets: {packets}"
        )
    return packets


def worker_main():
    parser = argparse.ArgumentParser(description="Hardware H.264 encoder worker")
    parser.add_argument("--ring", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    from hobot_vio import libsrcampy  # board SDK only in the worker process

    coded_width = (args.width + 7) // 8 * 8
    pad = coded_width - args.width
    if args.height % 2:
        raise SystemExit(f"Encoder height must be even, got {args.height}")
    elementary = args.output + ".h264"
    encoder = libsrcampy.Encoder()
    code = encoder.encode(0, 1, coded_width, args.height, 16000)
    if code:
        encoder.close()
        raise SystemExit(f"Hardware encoder initialization failed: {code}")
    try:
        try:
            with open(args.ring, "r+b") as source:
                mapping = mmap.mmap(source.fileno(), 0)
                ring = np.ndarray(
                    (args.slots, args.height, args.width, 3),
                    dtype=np.uint8,
                    buffer=mapping,
                )
                try:
                    with open(elementary, "wb") as output:
                        print("READY", flush=True)
                        for line in sys.stdin:
                            if line.strip() == "END":
                                break
                            index = int(line)
                            if not 0 <= index < args.slots:
                                raise ValueError(f"Invalid frame slot: {index}")
                            nv12 = bgr_to_nv12(pad_right(ring[index], coded_width))
                            code = encoder.encode_file(nv12)
                            packet = encoder.get_img()
                            if code or packet is None:
                                raise RuntimeError(
                                    f"Hardware encoder did not return a frame: {code}"
                                )
                            output.write(packet)
                            print(f"ACK {index}", flush=True)
                        drain_encoder_packets(encoder, output)
                finally:
                    del ring
                    mapping.close()
        finally:
            encoder.close()
            # The parent normally removes the ring file itself; if it died
            # abruptly, stdin above already hit EOF (or an ACK just hit a
            # closed pipe) and this best-effort unlink is the only ring
            # cleanup left.  It sits in the finally so no protocol failure
            # can skip it; a normal parent's own unlink tolerates the file
            # already being gone.
            try:
                os.unlink(args.ring)
            except OSError:
                pass
        print(f"ENCODED {time.perf_counter()}", flush=True)
    except BrokenPipeError:
        # The reading end of stdout is gone: the parent died mid-protocol.
        # No ACK or DONE can ever be delivered, and no remux will consume
        # the partially written elementary stream -- remove it, keep the
        # dead pipe quiet at interpreter shutdown, and stop without
        # remuxing an output nobody asked for.
        try:
            os.unlink(elementary)
        except OSError:
            pass
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, AttributeError):
            pass
        raise SystemExit("hardware encoder worker: parent went away mid-encode")
    subprocess.run(
        remux_command(
            elementary, args.audio, args.output, repeat=args.repeat, pad=pad
        ),
        check=True,
        timeout=30,
    )
    try:
        verify_remuxed_tracks(args.output)
    except RuntimeError as error:
        # The MP4 and the elementary stream stay on disk for diagnostics.
        # No DONE is printed, so the parent reports a failed run instead of
        # accepting a silent delivery.
        raise SystemExit(f"hardware encoder worker: {error}") from error
    print("DONE", flush=True)
    # The elementary stream was remux scratch; the finished .mp4 replaces it.
    try:
        os.unlink(elementary)
    except OSError:
        pass


if __name__ == "__main__":
    worker_main()
