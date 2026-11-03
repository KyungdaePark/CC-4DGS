"""Assemble the Hugging Face Space from this repository and (optionally) upload it.

The Space root needs the inference code of the repository next to the files in
``space/``. This script copies both into a staging folder and uploads it with
``huggingface_hub``, which stores binary files (wheels, images) through the Hub's
large-file storage automatically.

    # stage only (inspect the result, or push it yourself with git)
    python space/deploy.py --stage-dir /tmp/cc4dgs-space

    # create/update the Space and request ZeroGPU hardware
    HF_TOKEN=hf_xxx python space/deploy.py --space-id <user>/CC-4DGS
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parent
REPO = SPACE_DIR.parent

# Research code needed at inference time (training-only files are left out).
REPO_ITEMS = [
    "arguments",
    "gaussian_renderer",
    "scene",
    "utils",
    "tools",
    "submodules/diff-gaussian-rasterization",
    "submodules/simple-knn",
    "assets/cc4dgs_cdf.png",
    "LICENSE.md",
    "THIRD_PARTY_NOTICES.md",
]
SPACE_ITEMS = ["app.py", "README.md", "requirements.txt", "cc4dgs_space", "wheels"]
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "build", "dist", "*.egg-info", "*.so", "*.o", ".git", ".gitmodules", ".ninja_*",
)


def stage(dest: Path) -> Path:
    if not (REPO / "submodules/simple-knn/setup.py").exists():
        sys.exit("submodules/simple-knn is empty: run `git submodule update --init --recursive` first.")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel in REPO_ITEMS:
        src = REPO / rel
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest / rel, ignore=IGNORE)
        else:
            shutil.copy2(src, dest / rel)
    for rel in SPACE_ITEMS:
        src = SPACE_DIR / rel
        if src.is_dir():
            shutil.copytree(src, dest / rel, ignore=IGNORE)
        else:
            shutil.copy2(src, dest / rel)
    # Binary files must go through the Hub's large-file storage when pushed with git.
    lfs = ["*.whl", "*.png", "*.jpg", "*.mp4", "*.pth", "*.pt", "*.ply", "*.npz", "*.npy", "*.bin", "*.zip"]
    (dest / ".gitattributes").write_text("".join(f"{p} filter=lfs diff=lfs merge=lfs -text\n" for p in lfs))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space-id", help="<user or org>/<space name>; omit to only stage")
    ap.add_argument("--stage-dir", type=Path, help="staging folder (default: a temporary folder)")
    ap.add_argument("--hardware", default="zero-a10g", help="Space hardware flavor (ZeroGPU is 'zero-a10g')")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (default: $HF_TOKEN)")
    args = ap.parse_args()

    dest = stage(args.stage_dir or Path(tempfile.mkdtemp(prefix="cc4dgs-space-")) / "space")
    print(f"[deploy] staged Space files in {dest}")
    if not args.space_id:
        return

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=str(dest),
        commit_message="Deploy CC-4DGS viewer",
        delete_patterns=["*"],  # make the Space mirror the staged folder
    )
    print(f"[deploy] uploaded to https://huggingface.co/spaces/{args.space_id}")
    if args.hardware:
        try:
            api.request_space_hardware(args.space_id, hardware=args.hardware)
            print(f"[deploy] requested hardware '{args.hardware}'")
        except Exception as exc:  # e.g. ZeroGPU needs a PRO account or an organization plan
            print(f"[deploy] could not set hardware '{args.hardware}': {exc}\n"
                  "         Select 'ZeroGPU' in the Space settings instead.")


if __name__ == "__main__":
    main()
