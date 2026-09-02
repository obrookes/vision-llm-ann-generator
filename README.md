# vision-llm-ann-generator

Runs SAM3 (official `facebookresearch/sam3`, SA-FARI fine-tuned weights) text-prompted
tracking over a directory/manifest of videos and writes, per video, an overlay video plus
a machine-readable tracks JSON. Partner repo to
[`vision-llm-ann-verifier`](../vision-llm-ann-verifier): this repo *produces* the
overlay clips + manifests that repo *reviews*.

## Layout

- `annotate.py` — main entry point: manifest/dir + prompt + checkpoint -> per-video
  overlay + tracks JSON + `index.jsonl`.
- `sam3_runner.py` — thin wrapper over the SAM3 video predictor (the only module that
  imports `sam3`/`torch`; everything else runs on CPU without them).
- `overlay.py` — render a tracks JSON over its source video (cv2, palette-per-id mask
  blend, box, `id N` label). Also runnable standalone on an existing tracks JSON.
- `tracks.py` — tracks schema (`schema_version: 1`), pure-numpy COCO-compatible RLE
  codec, JSON read/write, `stem_for()` path flattening, `summarise()`.
- `manifest.py` — build manifests: walk a video dir -> `videos.txt`, or list an
  `annotate.py` output dir's overlays -> a verifier-ready manifest.
- `container/sam3.def` — Apptainer definition (built by another agent).
- `slurm/` — `run.sh` / `submit.sbatch` / `build_sif.sbatch` (written by another agent).
- `manifests/` — `smoke.txt`, `dev8.txt`, `dev32.txt`.
- `outputs/` — gitignored; per-run subdirectories go here.
- `docs/OUTPUT_SCHEMA.md` — the tracks JSON / `index.jsonl` contract shared with the
  verifier.

## Setup

- Weights: copy `~/Unmarked-Anything/weights/sam3/*.pt` to
  `$SCRATCH/weights/sam3/` (default checkpoint used is `sam3-safari-pos.pt`).
- Container/env: see `container/sam3.def` and `slurm/` (built separately).

## Usage

CLI, on a GPU node (or inside the container):

```bash
python annotate.py --manifest manifests/dev32.txt --prompt animal --out outputs/dev32_animal \
    [--checkpoint $SCRATCH/weights/sam3/sam3-safari-pos.pt] [--score-thresh 0.5] \
    [--max-frames N] [--no-overlay] [--overwrite] [--root /scratch/.../vids]
# or: --video-dir DIR instead of --manifest
```

Resumable: re-running with the same `--out` skips videos whose `<out>/<stem>.json`
already exists (unless `--overwrite`). Per-video errors are logged to `index.jsonl` with
`status: "error"` and do not stop the run.

Hand off the rendered overlays to the verifier:

```bash
python manifest.py --from-outputs outputs/dev32_animal \
    --write ../vision-llm-ann-verifier/manifests/dev32_animal.txt
```

Build a manifest from a raw video directory:

```bash
python manifest.py --video-dir /scratch/.../vids --ext .MP4,.mp4,.avi --write manifests/all.txt
```

Render (or re-render) an overlay from an existing tracks JSON directly:

```bash
python overlay.py outputs/dev32_animal/<stem>.json --out /tmp/preview.mp4
```

## Slurm (planned knobs)

`slurm/submit.sbatch` runs `slurm/run.sh`, which reads:

- `MANIFEST` — manifest to shard across the job's array tasks.
- `PROMPT` — required text prompt.
- `CHECKPOINT` — default `$SCRATCH/weights/sam3/sam3-safari-pos.pt`.
- `OUT` — default `outputs/<manifest-stem>_<prompt>`.
- `SIF` — path to the built Apptainer image.
- `EXTRA_ARGS` — passed through to `annotate.py` verbatim.

Mirrors this repo's sibling verifier's `slurm/run_vllm.sh` conventions (thin sbatch
header, `awk 'NR % n == i'` sharding, `APPTAINERENV_*` exports, `TMPDIR=/tmp` forced
inside the container).

## Output contract

See `docs/OUTPUT_SCHEMA.md` for the full tracks JSON / `index.jsonl` field reference.
