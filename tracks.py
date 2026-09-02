"""Tracks schema (schema_version=1), RLE mask codec, and JSON read/write.

Mask RLE: pycocotools is not available in this environment, so RLE encode/decode is
implemented in pure numpy here. The format matches COCO's *uncompressed* RLE
convention: column-major (Fortran order) run lengths over the flattened mask,
starting with a run of zeros (i.e. counts[0] is the length of the initial background
run, which may be 0), stored as a plain list of ints (not the LEB128-compressed
string pycocotools normally emits). This is a valid COCO RLE object
(`{"size": [h, w], "counts": [...]}`) that `pycocotools.mask.decode` can also read,
since pycocotools accepts uncompressed counts-as-list-of-ints as well as the
compressed string form.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def rle_encode(mask_bool: np.ndarray) -> dict:
    """COCO-compatible uncompressed RLE: {"size": [H, W], "counts": [int, ...]}."""
    h, w = mask_bool.shape
    flat = np.asfortranarray(mask_bool.astype(np.uint8)).reshape(-1, order="F")
    if flat.size == 0:
        return {"size": [h, w], "counts": []}
    # indices where the value changes
    change = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], change, [flat.size]))
    run_lengths = np.diff(boundaries).tolist()
    # counts must start with a run of the background (0) value; if the mask starts
    # with foreground (1), prepend a zero-length background run.
    if flat[0] == 1:
        run_lengths = [0] + run_lengths
    return {"size": [h, w], "counts": [int(c) for c in run_lengths]}


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        raise ValueError("compressed (string) RLE counts are not supported; only uncompressed int-list counts")
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    value = False  # counts[0] is always a background run
    for c in counts:
        if value and c:
            flat[pos:pos + c] = True
        pos += c
        value = not value
    return flat.reshape((h, w), order="F")


def stem_for(video_path: str, root: str | None = None) -> str:
    """Flattened relative-path stem: relative-to-root path with separators joined by '-',
    extension stripped. If root is None (or video_path is not under it), use just the filename
    stem.
    """
    p = Path(video_path)
    if root is not None:
        try:
            rel = p.resolve().relative_to(Path(root).resolve())
            parts = list(rel.with_suffix("").parts)
            return "-".join(parts)
        except ValueError:
            pass
    return p.stem


def write_tracks(path: str | Path, doc: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(doc, f)
    return path


def read_tracks(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def summarise(doc: dict) -> dict:
    frames = doc.get("frames", [])
    n_frames = len(frames)
    ids = set()
    max_concurrent = 0
    for fr in frames:
        objs = fr.get("objects", [])
        max_concurrent = max(max_concurrent, len(objs))
        for o in objs:
            ids.add(o["id"])
    return {"n_frames": n_frames, "n_tracks": len(ids), "max_concurrent": max_concurrent}
