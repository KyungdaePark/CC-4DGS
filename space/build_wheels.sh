#!/usr/bin/env bash
# Rebuild the prebuilt CUDA-extension wheels in space/wheels/ for the Space runtime.
#
# No GPU is needed. Requires Python 3.10 (the ZeroGPU Python), a C++ compiler
# and network access to pypi.org and developer.download.nvidia.com.
#
#   bash space/build_wheels.sh            # uses python3.10 from PATH (or uv)
#   PYTHON=/path/to/python3.10 bash space/build_wheels.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1}"
ARCHS="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0+PTX}"
WORK="${WORK:-$(mktemp -d -t cc4dgs-wheels-XXXX)}"
OUT="$HERE/wheels"

if [ ! -f "$REPO/submodules/simple-knn/setup.py" ]; then
  git -C "$REPO" submodule update --init --recursive
fi

if [ -z "${PYTHON:-}" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv -q -p 3.10 "$WORK/venv"
  else
    python3.10 -m venv "$WORK/venv"
  fi
  PYTHON="$WORK/venv/bin/python"
  "$PYTHON" -m ensurepip -q 2>/dev/null || true
fi

"$PYTHON" -m pip install -q --upgrade pip setuptools wheel ninja
"$PYTHON" -m pip install -q "$TORCH_SPEC" numpy

CUDA_HOME="$(cd "$HERE" && "$PYTHON" -m cc4dgs_space.cuda_toolchain "$WORK/cuda" | tail -1)"
export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCHS"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

mkdir -p "$OUT"
rm -f "$OUT"/*.whl
for pkg in simple-knn diff-gaussian-rasterization; do
  "$PYTHON" -m pip wheel --no-build-isolation --no-deps -w "$OUT" "$REPO/submodules/$pkg"
  rm -rf "$REPO/submodules/$pkg/build" "$REPO/submodules/$pkg"/*.egg-info
done

"$PYTHON" - "$OUT/BUILD_INFO.json" "$ARCHS" <<'PY'
import json, sys, torch
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
info = {"python": py, "torch": torch.__version__, "cuda": torch.version.cuda,
        "torch_cuda_arch_list": sys.argv[2], "built_with": "space/build_wheels.sh"}
open(sys.argv[1], "w").write(json.dumps(info, indent=2) + "\n")
print(info)
PY
ls -la "$OUT"
