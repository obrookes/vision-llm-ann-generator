"""Pure-cv2/PIL frame decode + fps subsampling helper.

No torch/sam3 import here (only cv2, PIL, math) so this module runs on a CPU-only
login node with no torch installed, and is importable by sam3_runner.py (video mode),
annotate.py (frame counting) and tests alike.
"""
from __future__ import annotations

import math


def decode_frames(
    video_path: str,
    sample_fps: float | None,
    max_frames: int | None = None,
    extra_indices=None,
    stop_after: int | None = None,
    extra_only: bool = False,
):
    """Decode a video, keeping only the frames a sample_fps subsampling would keep, plus
    any explicitly requested extra source-frame indices.

    Source frame i is kept when floor(i * sample_fps / src_fps) is strictly greater than
    the bucket value of the last kept frame (frame 0 is always kept, since its bucket is
    0 > -1), OR when `i in extra_indices`. `last_bucket` only advances on an actual bucket
    increase (never on an extra-index-only keep), so the regular sampling grid is exactly
    the same set of indices regardless of which extra indices are requested. If sample_fps
    is None/0, or >= src_fps, every frame is kept (stride 1), and extra_indices is then a
    no-op (already a subset of "every frame").

    Skipped frames are only cap.grab()'d (cheap, no decode); kept frames are cap.retrieve()'d
    and converted BGR -> RGB -> PIL.Image. Stops early once max_frames frames have been kept.

    `extra_indices`: optional iterable of int source-frame indices to force-keep even if they
    don't fall on the sampling grid (e.g. frames a human annotator labelled). Indices at or
    beyond the source frame count (or past `stop_after`) are silently ignored.

    `extra_only`: keep ONLY the `extra_indices` frames (no sampling grid at all), and stop
    decoding after the largest of them. Used to run SAM3 on exactly the human-annotated
    frames. With no extra indices this keeps nothing.

    `stop_after`: optional int source-frame index; decoding stops once this source index has
    been grabbed, i.e. no frame with index > stop_after is considered. None (default, the
    existing behaviour) decodes to the end of the video.

    Returns (frame_indices, frames, src_fps):
        frame_indices: list[int], the source-video indices of the kept frames, in ascending
            order (extra indices are interleaved in order alongside the regular grid)
        frames: list[PIL.Image.Image], RGB
        src_fps: float, the source video's reported fps (falls back to 24.0 if cv2 reports 0)
    """
    import cv2
    from PIL import Image

    extra_set = set(extra_indices) if extra_indices else set()
    if extra_only:
        if not extra_set:
            return [], [], None
        last_extra = max(extra_set)
        stop_after = last_extra if stop_after is None else min(stop_after, last_extra)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    every_frame = not sample_fps or sample_fps >= src_fps

    frame_indices: list[int] = []
    frames: list = []
    last_bucket = -1
    i = 0
    try:
        while True:
            if stop_after is not None and i > stop_after:
                break

            ok = cap.grab()
            if not ok:
                break

            if extra_only:
                keep = i in extra_set
                bucket = last_bucket
            elif every_frame:
                keep = True
                bucket = i
            else:
                bucket = int(math.floor(i * sample_fps / src_fps))
                keep = bucket > last_bucket or i in extra_set

            if keep:
                ok2, frame_bgr = cap.retrieve()
                if not ok2:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                frame_indices.append(i)
                if not every_frame and bucket > last_bucket:
                    last_bucket = bucket
                if max_frames is not None and len(frame_indices) >= max_frames:
                    break

            i += 1
    finally:
        cap.release()

    return frame_indices, frames, src_fps
