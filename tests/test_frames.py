"""Tests for frames.decode_frames, incl. the extra_indices/stop_after additions.

Uses only cv2/numpy (via a tiny synthetic video written to a temp dir), matching the
CPU-only, no-torch dev environment.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from frames import decode_frames


def _write_synthetic_video(path: str, n_frames: int, fps: float, size=(32, 24)) -> None:
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("cv2 VideoWriter could not open (missing mp4v codec in this environment)")
    for i in range(n_frames):
        # Distinct per-frame content so we could tell frames apart if needed.
        frame = np.full((height, width, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    n_frames = 48
    fps = 24.0
    _write_synthetic_video(path, n_frames, fps)
    return str(path), n_frames, fps


def _expected_grid(n_frames: int, fps: float, sample_fps: float) -> list[int]:
    grid = []
    last_bucket = -1
    for i in range(n_frames):
        bucket = int(math.floor(i * sample_fps / fps))
        if bucket > last_bucket:
            grid.append(i)
            last_bucket = bucket
    return grid


def test_decode_frames_no_extra_matches_grid(synthetic_video):
    path, n_frames, fps = synthetic_video
    sample_fps = 6.0
    indices, frames, src_fps = decode_frames(path, sample_fps)
    expected = _expected_grid(n_frames, fps, sample_fps)
    assert indices == expected
    assert len(frames) == len(indices)


def test_decode_frames_with_extra_indices(synthetic_video):
    path, n_frames, fps = synthetic_video
    sample_fps = 6.0
    extra = {26, 75, 5}  # 75 is out of range for a 48-frame video and must be ignored

    grid = _expected_grid(n_frames, fps, sample_fps)
    indices, frames, src_fps = decode_frames(path, sample_fps, extra_indices=extra)

    expected = sorted(set(grid) | {5, 26})
    assert indices == expected
    assert len(frames) == len(indices)

    # The regular grid portion is unaffected by which extras were requested.
    indices_no_extra, _, _ = decode_frames(path, sample_fps)
    assert indices_no_extra == grid
    assert set(grid).issubset(set(indices))


def test_decode_frames_stop_after(synthetic_video):
    path, n_frames, fps = synthetic_video
    sample_fps = 6.0
    indices, frames, src_fps = decode_frames(path, sample_fps, stop_after=20)
    assert all(i <= 20 for i in indices)
    assert max(indices) <= 20


def test_decode_frames_max_frames_unchanged(synthetic_video):
    path, n_frames, fps = synthetic_video
    sample_fps = 6.0
    indices, frames, src_fps = decode_frames(path, sample_fps, max_frames=3)
    assert len(indices) == 3
    assert len(frames) == 3


def test_decode_frames_extra_only(synthetic_video):
    path, n_frames, fps = synthetic_video
    indices, frames, _ = decode_frames(path, 6.0, extra_indices={5, 21, 22, 999}, extra_only=True)
    assert indices == [5, 21, 22]
    assert len(frames) == 3


def test_decode_frames_extra_only_without_extras_keeps_nothing(synthetic_video):
    path, _, _ = synthetic_video
    indices, frames, _ = decode_frames(path, 6.0, extra_only=True)
    assert indices == [] and frames == []
