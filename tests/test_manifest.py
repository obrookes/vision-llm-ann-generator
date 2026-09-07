"""Tests for manifest.frames_by_video."""
from __future__ import annotations

from manifest import frames_by_video


def _write_csv(path, rows_text: str) -> str:
    path.write_text(rows_text)
    return str(path)


def test_frames_by_video_basic(tmp_path):
    csv_text = (
        "video_file,frame_idx\n"
        "dir/videoA.MP4,0.0\n"
        "dir/videoA.MP4,26.0\n"
        "otherdir/videoB.MP4,5.0\n"
    )
    csv_path = _write_csv(tmp_path / "annotations.csv", csv_text)

    result = frames_by_video(csv_path)

    assert result == {
        "videoA.MP4": {0, 26},
        "videoB.MP4": {5},
    }


def test_frames_by_video_tolerates_concatenated_headers(tmp_path):
    # Simulates a CSV built by cat-ing several per-video exports, each with its own header.
    csv_text = (
        "video_file,frame_idx\n"
        "videoA.MP4,10.0\n"
        "video_file,frame_idx\n"  # repeated header row, mid-file
        "videoA.MP4,20.0\n"
        "videoB.MP4,1.0\n"
    )
    csv_path = _write_csv(tmp_path / "annotations.csv", csv_text)

    result = frames_by_video(csv_path)

    assert result == {
        "videoA.MP4": {10, 20},
        "videoB.MP4": {1},
    }
