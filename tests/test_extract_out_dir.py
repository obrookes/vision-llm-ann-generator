"""Tests for `tools/extract_annotated_frames.py --out-dir`.

Runs only where `camtrapalign`/`pandas` are importable (the camtrap-align venv used by
`slurm/pss_p1_frames.sbatch STAGE=extract`), not in this repo's lightweight CPU-only dev
env. `_process_video_task` is monkeypatched so the test doesn't need real OCR/video
content -- it only exercises the --out-dir plumbing (where frame/index/videos/cache
files land), not OCR correctness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("camtrapalign")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import extract_annotated_frames as eaf  # noqa: E402

CONFIG = Path.home() / "camtrap-align" / "configs" / "pss_p1.yaml"


def _write_tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (32, 24))
    if not writer.isOpened():
        pytest.skip("cv2 VideoWriter could not open (missing mp4v codec in this environment)")
    for i in range(24):
        writer.write(np.full((24, 32, 3), i % 256, dtype=np.uint8))
    writer.release()


def _make_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "dataset"
    (ds / "videos").mkdir(parents=True)
    _write_tiny_video(ds / "videos" / "clipA.mp4")

    pd.DataFrame({
        "video_key": ["k1"],
        "annot_time": [""],
        "cam": ["1"],
        "transect": ["T1"],
        "video_id": ["v1"],
        "duration": ["1"],
        "match_strategy": [""],
    }).to_csv(ds / "frame_assignments.csv", index=False)

    pd.DataFrame({
        "video_key": ["k1"],
        "filepath": ["orig/clipA.mp4"],
        "local_name": ["clipA.mp4"],
        "mode": ["copy"],
        "bytes": [1],
    }).to_csv(ds / "videos" / "manifest.csv", index=False)

    return ds


def test_out_dir_routes_outputs_away_from_dataset_dir(tmp_path, monkeypatch):
    if not CONFIG.exists():
        pytest.skip(f"config not found: {CONFIG}")

    ds = _make_dataset(tmp_path)
    out = tmp_path / "full"

    seen_out_dirs = []

    def fake_process_video_task(task):
        seen_out_dirs.append(task["out_dir"])
        return {
            "file": task["file"], "video_key": task["video_key"], "rows": [],
            "summary": {"ok": 0, "verified": 0, "corrected": 0, "unverified": 0,
                        "not_found": 0, "out_of_range": 0},
            "start": None, "source": "none", "ocr_result": None,
            "status": "ok", "error": "",
        }

    monkeypatch.setattr(eaf, "_process_video_task", fake_process_video_task)
    monkeypatch.setattr(eaf, "_make_video_row", lambda task, r, overrides, species_col: {
        "video_key": task["video_key"], "file": task["file"],
    })

    monkeypatch.setattr(sys, "argv", [
        "extract_annotated_frames.py",
        "--dataset-dir", str(ds),
        "--out-dir", str(out),
        "--config", str(CONFIG),
        "--workers", "1",
    ])
    eaf.main()

    assert seen_out_dirs == [out]  # frame writer told to write under --out-dir, not --dataset-dir

    assert (out / "annotated_frames_index.csv").exists()
    assert (out / "videos.csv").exists()
    assert (out / "ocr_cache.json").exists()

    # --dataset-dir itself is untouched (frame_assignments.csv/manifest.csv are inputs only)
    assert not (ds / "annotated_frames_index.csv").exists()
    assert not (ds / "videos.csv").exists()
    assert not (ds / "ocr_cache.json").exists()


def test_default_out_dir_is_dataset_dir(tmp_path, monkeypatch):
    if not CONFIG.exists():
        pytest.skip(f"config not found: {CONFIG}")

    ds = _make_dataset(tmp_path)
    seen_out_dirs = []

    def fake_process_video_task(task):
        seen_out_dirs.append(task["out_dir"])
        return {
            "file": task["file"], "video_key": task["video_key"], "rows": [],
            "summary": {"ok": 0, "verified": 0, "corrected": 0, "unverified": 0,
                        "not_found": 0, "out_of_range": 0},
            "start": None, "source": "none", "ocr_result": None,
            "status": "ok", "error": "",
        }

    monkeypatch.setattr(eaf, "_process_video_task", fake_process_video_task)
    monkeypatch.setattr(eaf, "_make_video_row", lambda task, r, overrides, species_col: {
        "video_key": task["video_key"], "file": task["file"],
    })

    monkeypatch.setattr(sys, "argv", [
        "extract_annotated_frames.py",
        "--dataset-dir", str(ds),
        "--config", str(CONFIG),
        "--workers", "1",
    ])
    eaf.main()

    assert seen_out_dirs == [ds.resolve()]
    assert (ds / "annotated_frames_index.csv").exists()
