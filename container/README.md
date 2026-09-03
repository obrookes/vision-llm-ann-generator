# container/

`sam3.def` builds an Apptainer image with the official `facebookresearch/sam3` predictor,
`pycocotools`, and `opencv-python-headless` on top of an NGC PyTorch base
(arm64, Python 3.12, torch >= 2.7, CUDA 12.x).

## Normal path (confirmed 2026-09-03)

```
setsid -f nohup container/build_sandbox.sh > $SCRATCH/logs/build-sam3-sandbox.log 2>&1 < /dev/null
# ... ~25 min: pulls nvcr.io/nvidia/pytorch:25.06-py3, pip-installs sam3 (torch stays pinned at the
#     NGC build, 2.8.0a0+nv25.06), tars the sandbox to $SCRATCH/containers/sam3-sandbox.tar (25 GB)
NAME=sam3 sbatch slurm/build_sif.sbatch      # untars to node /tmp, writes $SCRATCH/containers/sam3.sif
```

Things that do NOT work on this login node, which is why the script looks the way it does:

- building straight onto `$SCRATCH`: def-file builds chown files in the build tmp dir
  ("ownership change not allowed"), and even with `TMPDIR` on local disk the final rootfs->sandbox
  copy onto Lustre dies ("archive/tar: missed writing ... unexpected EOF");
- `cp -a` of the finished 25 GB sandbox tree onto `$SCRATCH` (killed);
- `apptainer build sam3.sif <sandbox>` on the login node (mksquashfs dies in the session cgroup;
  apptainer 1.4.1 here has no `--mksquashfs-procs/-mem` flags to cap it).

Streaming `tar -cf` onto Lustre and unpacking on a compute node works. `slurm/run.sh` looks for
`$SCRATCH/containers/sam3.sif` first and falls back to `$SCRATCH/containers/sam3-sandbox` if you
chose to unpack the tarball there instead.

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
