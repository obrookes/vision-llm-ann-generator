#!/bin/bash
#SBATCH --job-name=pss-frames
#SBATCH --account=brics.b5bd
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm-pss-frames-%j.out
#
# CPU stages around the SAM3 job for a camtrap-align dataset dir (default pss_p1):
#   STAGE=extract  pytest, then OCR-verified annotated frame extraction (tools/extract_annotated_frames.py)
#   STAGE=export   per-frame seg_masks/ JSON + overlay/ PNG (tools/export_frame_masks.py)
# Env: DATASET, CONFIG, TRACKS_DIR, ONLY (substring filter), WORKERS.
#   STAGE=extract ONLY=04010054 sbatch slurm/pss_p1_frames.sbatch

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
set -euo pipefail

DATASET="${DATASET:-$SCRATCH/single-animal-distance/pss_p1}"
CONFIG="${CONFIG:-$HOME/camtrap-align/configs/pss_p1.yaml}"
TRACKS_DIR="${TRACKS_DIR:-$DATASET/sam3_image_animal}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
PY="$SCRATCH/envs/camtrap-align/bin/python"
ONLY_ARGS=()
[ -n "${ONLY:-}" ] && ONLY_ARGS=(--only "$ONLY")

echo "== pss_p1_frames: STAGE=$STAGE DATASET=$DATASET ONLY=${ONLY:-} =="
case "$STAGE" in
    extract)
        "$PY" -m pytest -q tests/
        "$PY" tools/extract_annotated_frames.py --dataset-dir "$DATASET" --config "$CONFIG" \
            --workers "$WORKERS" "${ONLY_ARGS[@]}" ${EXTRA_ARGS:-}
        ;;
    export)
        "$PY" tools/export_frame_masks.py --dataset-dir "$DATASET" --tracks-dir "$TRACKS_DIR" \
            "${ONLY_ARGS[@]}" ${EXTRA_ARGS:-}
        ;;
    *) echo "STAGE must be extract or export" >&2; exit 1 ;;
esac
