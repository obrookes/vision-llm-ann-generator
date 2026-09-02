#!/bin/bash
#SBATCH --job-name=build-sif
#SBATCH --account=brics.b5bd
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm-build-sif-%j.out
# Convert the sandbox (built on the login node from cached OCI layers) into a SIF on a compute node,
# where mksquashfs is not throttled by the login node's pids cgroup limit. No network needed.
#
# Usage:
#   NAME=sam3 sbatch slurm/build_sif.sbatch
#   SANDBOX=$SCRATCH/containers/sam3-sandbox sbatch slurm/build_sif.sbatch

NAME="${NAME:-sam3}"
SANDBOX="${SANDBOX:-$SCRATCH/containers/${NAME}-sandbox}"

if [ ! -d "$SANDBOX" ]; then
    echo "build_sif.sbatch: sandbox not found at $SANDBOX" >&2
    exit 1
fi

export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache APPTAINER_TMPDIR=$SCRATCH/apptainer-cache/tmp
apptainer build --force "$SCRATCH/containers/${NAME}.sif" "$SANDBOX" && ls -la "$SCRATCH/containers/"
