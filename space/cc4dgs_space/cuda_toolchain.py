"""Assemble a minimal CUDA toolkit from NVIDIA's official redistributable archives.

Hugging Face Spaces (including ZeroGPU) do not ship ``nvcc``. The pip package
``nvidia-cuda-nvcc-cu12`` for CUDA 12.4 contains only ``ptxas``, so it cannot be
used to build PyTorch CUDA extensions. This module downloads the
``cuda_nvcc``/``cuda_cudart``/``cuda_cccl``/``cuda_nvtx`` archives listed in
NVIDIA's ``redistrib_<version>.json`` manifest and merges them into a regular
``CUDA_HOME`` layout (``bin/``, ``include/``, ``lib64/``, ``nvvm/``).

No GPU is required to compile: the target architectures are selected with
``TORCH_CUDA_ARCH_LIST``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REDIST_ROOT = "https://developer.download.nvidia.com/compute/cuda/redist"
DEFAULT_CUDA_VERSION = "12.4.1"  # matches the torch==2.5.1 (cu124) wheel
COMPONENTS = ("cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_nvtx", "cuda_profiler_api")
PLATFORM = "linux-x86_64"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _merge_tree(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir() and not item.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
            _merge_tree(item, target)
        else:
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.move(str(item), str(target))


def ensure_cuda_toolkit(
    prefix: str | os.PathLike | None = None,
    cuda_version: str = DEFAULT_CUDA_VERSION,
) -> Path:
    """Return a CUDA_HOME containing nvcc, downloading it on first use."""
    prefix = Path(prefix or os.environ.get("CC4DGS_CUDA_PREFIX", Path.home() / ".cache" / "cc4dgs" / f"cuda-{cuda_version}"))
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.exists():
        return prefix

    with urllib.request.urlopen(f"{REDIST_ROOT}/redistrib_{cuda_version}.json", timeout=60) as resp:
        manifest = json.load(resp)

    prefix.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cc4dgs-cuda-") as tmpdir:
        tmpdir = Path(tmpdir)
        for component in COMPONENTS:
            entry = manifest[component][PLATFORM]
            archive = tmpdir / Path(entry["relative_path"]).name
            print(f"[cuda_toolchain] downloading {component} {manifest[component]['version']}", flush=True)
            _download(f"{REDIST_ROOT}/{entry['relative_path']}", archive)
            if _sha256(archive) != entry["sha256"]:
                raise RuntimeError(f"SHA256 mismatch for {archive.name}")
            extract_dir = tmpdir / component
            with tarfile.open(archive, "r:xz") as tar:
                tar.extractall(extract_dir)  # archive comes from NVIDIA and is hash-checked
            (root,) = [p for p in extract_dir.iterdir() if p.is_dir()]
            _merge_tree(root, prefix)
            archive.unlink()

    lib = prefix / "lib"
    lib64 = prefix / "lib64"
    if lib.is_dir() and not lib64.exists():
        lib64.symlink_to("lib")
    if not nvcc.exists():
        raise RuntimeError(f"nvcc was not found after extracting CUDA {cuda_version} into {prefix}")
    return prefix


def cuda_env(cuda_home: Path, arch_list: str | None = None) -> dict:
    """Environment variables for building torch CUDA extensions with ``cuda_home``."""
    env = dict(os.environ)
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_home / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["TORCH_CUDA_ARCH_LIST"] = arch_list or env.get("TORCH_CUDA_ARCH_LIST") or "8.0;8.6;8.9;9.0+PTX"
    env.setdefault("MAX_JOBS", str(max(1, (os.cpu_count() or 2))))
    return env


if __name__ == "__main__":
    import sys

    print(ensure_cuda_toolkit(sys.argv[1] if len(sys.argv) > 1 else None))
