# Output schema

`annotate.py` writes two kinds of files per run into the output directory (`--out`):

- `<stem>.json` — one tracks document per video (below).
- `<stem>.mp4` — an overlay video rendered from that document (unless `--no-overlay`).
- `index.jsonl` — one status line per processed video, appended as the run progresses.

`<stem>` is the flattened path (see `tracks.stem_for`): the video path relative to
`--root` with path separators joined by `-` and the extension stripped, or just the
filename stem if `--root` is not given or the video isn't under it. This avoids clip-id
collisions across cameras/missions in the corpus (e.g.
`mission_5_phase_3_pss-Videos_reference-Ref_cam146-02270454`).

## Tracks JSON (`<stem>.json`)

```json
{
  "schema_version": 1,
  "video": "/abs/path/to/source/video.MP4",
  "prompt": "animal",
  "checkpoint": "sam3-safari-pos.pt",
  "fps": 30.0,
  "width": 720,
  "height": 404,
  "n_frames": 1800,
  "frames": [
    {
      "frame": 0,
      "objects": [
        {
          "id": 3,
          "score": 0.91,
          "box_xywh": [120.5, 40.0, 80.0, 60.0],
          "mask_rle": {"size": [404, 720], "counts": [1234, 56, 789, ...]}
        }
      ]
    }
  ]
}
```

Field notes:

- `schema_version` — integer, currently `1`. Bump on any breaking change to this
  document's shape.
- `video` — absolute path to the source video as given on the command line (or resolved
  from `--video-dir`).
- `prompt` — the text prompt passed to SAM3 for this run.
- `checkpoint` — basename of the SAM3 checkpoint file used (not the full path, so the
  JSON stays portable across machines).
- `fps`, `width`, `height`, `n_frames` — probed from the source video with OpenCV
  (`n_frames` is the *source* frame count via `CAP_PROP_FRAME_COUNT`, which may exceed
  the number of entries in `frames` if `--max-frames` truncated tracking).
- `frames` — one entry per tracked frame, **in ascending frame order**. Frames with no
  detected objects are still listed, with `"objects": []` — do not assume a missing
  frame index means "no objects"; a genuinely absent frame index means tracking did not
  reach that frame (e.g. `--max-frames` truncation).
- `frames[].frame` — 0-based frame index into the source video.
- `frames[].objects[].id` — SAM3's per-video object id (stable across frames within one
  video; **not** unique across videos).
- `frames[].objects[].score` — SAM3's detection/tracking probability for this object on
  this frame (`null` if unavailable). If `--score-thresh` was set, only objects at or
  above that threshold are present.
- `frames[].objects[].box_xywh` — `[x, y, w, h]` in **absolute pixels** (top-left corner,
  width, height), converted from SAM3's normalised `[0, 1]` output using `width`/`height`
  above.
- `frames[].objects[].mask_rle` — COCO-style **uncompressed** RLE:
  `{"size": [h, w], "counts": [int, ...]}`, column-major (Fortran order) run lengths over
  the flattened mask, where `counts[0]` is always the length of the leading background
  run (possibly `0`). This is a valid COCO RLE object that `pycocotools.mask.decode` can
  read directly (pycocotools accepts uncompressed int-list counts, not only the
  compressed LEB128 string form); this repo decodes/encodes it with pure numpy in
  `tracks.py` since pycocotools isn't installed in the dev environment. May be absent if
  masks weren't produced for that object.

## `index.jsonl`

One JSON object per line, appended as each video finishes (successfully or not), so the
run is resumable and progress is visible while it's in flight:

```json
{"video": "...", "stem": "...", "status": "ok", "n_frames": 1800, "n_tracks": 4,
 "max_concurrent": 2, "seconds": 12.3, "prompt": "animal", "checkpoint": "sam3-safari-pos.pt"}
{"video": "...", "stem": "...", "status": "error", "error": "RuntimeError(...)",
 "prompt": "animal", "checkpoint": "sam3-safari-pos.pt"}
```

- `status` — `"ok"` or `"error"`.
- `error` — present only on `status: "error"`; `repr()` of the caught exception.
- `n_frames`, `n_tracks`, `max_concurrent` — present only on `status: "ok"`; from
  `tracks.summarise()`: number of frame entries written, number of distinct object ids
  seen across the whole video, and the largest number of objects present in any single
  frame.
- `seconds` — wall-clock time to process this one video (tracking + write + overlay
  render), present only on `status: "ok"`.

Resume behaviour: `annotate.py` skips a video if `<out>/<stem>.json` already exists,
unless `--overwrite` is given. This is checked by file existence, not by `index.jsonl`
contents, so a run can be resumed even if `index.jsonl` was lost or truncated.
