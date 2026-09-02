#!/bin/bash
#SBATCH --job-name=sam3-annotate
#SBATCH --account=brics.b5bd
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm-%A_%a.out
#SBATCH --array=0-0
#
# Usage:
#   MANIFEST=manifests/dev32.txt PROMPT=animal sbatch --array=0-3 slurm/submit.sbatch
#   PROMPT=chimpanzee sbatch slurm/submit.sbatch   # smoke.txt default manifest
# Each array task runs one SAM3 predictor on one GPU over its shard of the manifest.

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" && exec bash slurm/run.sh
