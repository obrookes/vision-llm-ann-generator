# container/

`sam3.def` builds an Apptainer image with the official `facebookresearch/sam3` predictor,
`pycocotools`, and `opencv-python-headless` on top of an NGC PyTorch base
(arm64, Python 3.12, torch >= 2.7, CUDA 12.x).

## Normal path

```
export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache
# Def-file builds (unlike plain docker:// pulls) chown files in the build temp dir, which Lustre
# refuses ("ownership change not allowed"), and the final rootfs->sandbox copy onto Lustre also
# failed ("archive/tar: missed writing ... unexpected EOF"). So build entirely on the login node's
# local disk (/local, 3 TB), then copy the finished sandbox to $SCRATCH as plain files.
export APPTAINER_TMPDIR=/local/user/$(id -u)/apptainer-tmp TMPDIR=/local/user/$(id -u)/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer build --sandbox /local/user/$(id -u)/sam3-sandbox container/sam3.def
cp -a /local/user/$(id -u)/sam3-sandbox "$SCRATCH/containers/sam3-sandbox"
NAME=sam3 sbatch slurm/build_sif.sbatch     # sandbox -> $SCRATCH/containers/sam3.sif on a compute node
```

`slurm/run.sh` looks for `$SCRATCH/containers/sam3.sif` first and falls back to
`$SCRATCH/containers/sam3-sandbox` if the SIF hasn't been built yet, so the sandbox alone is enough
to run jobs -- `build_sif.sbatch` just makes startup faster and avoids keeping an unpacked sandbox
around.

## Fallback: login-node cgroup kills the pip install

The login node's build cgroup is 4 GiB RAM / 500 pids. Pulling the NGC base image is usually fine
(it's just layer extraction), but `pip install git+https://github.com/facebookresearch/sam3.git`
compiles/pulls a fair number of dependencies and can be OOM- or pid-killed mid-`%post`, aborting the
whole `apptainer build --sandbox`.

If that happens, build a bare base-image sandbox (no `%post` pip step) and install SAM3 into a venv
on `$SCRATCH` instead, bound into the container at run time:

```
# 1. Sandbox with just the NGC base image (edit sam3.def to comment out the SAM3 pip install
#    under %post, or build a trimmed def with only the base image and no %post block).
apptainer build --sandbox "$SCRATCH/containers/sam3-sandbox" container/sam3.def

# 2. Create the venv on scratch (has its own cgroup headroom via apptainer exec) and install SAM3
#    into it, reusing the base image's site-packages (torch, CUDA libs) so it isn't reinstalled.
apptainer exec --bind /lus,/scratch,$HOME "$SCRATCH/containers/sam3-sandbox" \
    python3 -m venv --system-site-packages "$SCRATCH/envs/sam3"
apptainer exec --bind /lus,/scratch,$HOME "$SCRATCH/containers/sam3-sandbox" \
    "$SCRATCH/envs/sam3/bin/pip" install --no-cache-dir \
        "git+https://github.com/facebookresearch/sam3.git" pycocotools opencv-python-headless

# 3. At run time, use the venv's python instead of the container's:
apptainer exec --nv --bind /lus,/scratch,$HOME "$SCRATCH/containers/sam3-sandbox" \
    "$SCRATCH/envs/sam3/bin/python" annotate.py ...
```

`slurm/run.sh` invokes `python3` inside the container by default; if this fallback is used, set
`PY="$SCRATCH/envs/sam3/bin/python"` (mirrors the verifier repo's `run_vllm.sh` pattern of
preferring an explicit interpreter path when one exists) before running the job, or edit
`slurm/run.sh`'s invocation line to point at it.

Whichever path is actually used on this cluster, note it here (update this file) once confirmed --
the plan calls for documenting whichever approach worked.
