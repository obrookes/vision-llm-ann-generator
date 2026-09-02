#!/usr/bin/env python3
"""Sanity-check the sam3 container: python/torch/cuda versions, sam3 import, and the signature of
build_sam3_video_predictor (to confirm the checkpoint kwarg name before wiring up sam3_runner.py).

Usage (inside the container):
    apptainer exec --nv sam3.sif python3 tools/container_check.py

No sam3 import at module level -- a failed import must not prevent the version info above it from
printing, and must exit non-zero rather than raising an unhandled traceback.
"""
from __future__ import annotations

import inspect
import sys


def main() -> int:
    print(f"python: {sys.version}")

    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda device: {torch.cuda.get_device_name(0)}")
            print(f"torch cuda version: {torch.version.cuda}")
    except Exception as e:
        print(f"torch import/check FAILED: {e!r}")
        return 1

    try:
        import sam3
        print(f"sam3 module: {getattr(sam3, '__file__', '<no __file__>')}")
    except Exception as e:
        print(f"import sam3 FAILED: {e!r}")
        return 1

    try:
        from sam3.model_builder import build_sam3_video_predictor
    except Exception as e:
        print(f"from sam3.model_builder import build_sam3_video_predictor FAILED: {e!r}")
        return 1

    print(f"build_sam3_video_predictor signature: {inspect.signature(build_sam3_video_predictor)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
