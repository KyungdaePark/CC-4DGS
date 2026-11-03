"""Make the CUDA extensions (``diff_gaussian_rasterization``, ``simple_knn``) importable.

Resolution order:

1. Already installed (e.g. a previous start of the Space) -> nothing to do.
2. A prebuilt wheel in ``wheels/`` that matches this Python/torch build.
3. Build from ``submodules/`` with a CUDA toolkit assembled from NVIDIA's
   redistributable archives (see ``cuda_toolchain.py``). No GPU is needed for
   this step, so it also works on ZeroGPU hardware before a GPU is attached.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

EXTENSIONS = {
    # import name -> source directory under submodules/
    "diff_gaussian_rasterization": "diff-gaussian-rasterization",
    "simple_knn": "simple-knn",
}


def _importable(name: str) -> bool:
    importlib.invalidate_caches()
    try:
        importlib.import_module(f"{name}._C")
        return True
    except ImportError:
        sys.modules.pop(name, None)
        return False


def _pip(*args: str, env: dict | None = None) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir", *args]
    print("[extensions] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _torch_tag() -> str:
    import torch

    return torch.__version__


def _install_prebuilt(wheel_dir: Path, name: str) -> bool:
    info_path = wheel_dir / "BUILD_INFO.json"
    if not info_path.exists():
        return False
    info = json.loads(info_path.read_text())
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if info.get("torch") != _torch_tag() or info.get("python") != py_tag:
        print(
            f"[extensions] prebuilt wheels target torch={info.get('torch')} {info.get('python')}, "
            f"runtime is torch={_torch_tag()} {py_tag}; building from source instead",
            flush=True,
        )
        return False
    dist = name.replace("-", "_")
    candidates = sorted(wheel_dir.glob(f"{dist}-*-{py_tag}-{py_tag}-*.whl"))
    if not candidates:
        return False
    _pip(str(candidates[-1]))
    return _importable(name)


def _build_from_source(repo_root: Path, name: str, subdir: str) -> bool:
    from .cuda_toolchain import cuda_env, ensure_cuda_toolkit

    cuda_home = ensure_cuda_toolkit()
    env = cuda_env(cuda_home)
    _pip("--no-build-isolation", str(repo_root / "submodules" / subdir), env=env)
    return _importable(name)


def ensure_extensions(repo_root: str | os.PathLike, wheel_dir: str | os.PathLike | None = None) -> None:
    """``repo_root`` holds ``submodules/``; ``wheel_dir`` defaults to ``<repo_root>/wheels``."""
    repo_root = Path(repo_root)
    wheel_dir = Path(wheel_dir) if wheel_dir is not None else repo_root / "wheels"
    for name, subdir in EXTENSIONS.items():
        if _importable(name):
            continue
        if wheel_dir.is_dir() and _install_prebuilt(wheel_dir, name):
            print(f"[extensions] installed prebuilt {name}", flush=True)
            continue
        if _build_from_source(repo_root, name, subdir):
            print(f"[extensions] built {name} from source", flush=True)
            continue
        raise RuntimeError(f"Could not install the CUDA extension '{name}'.")
