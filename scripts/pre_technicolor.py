#!/usr/bin/env python3
"""
Minimal Technicolor preprocessing pipeline.

Reference implementation adapted from the public E-D3DGS project.
This script prepares the directory layout expected by our loader:

<scene>/
    cameras_parameters.txt
    images/camXX/0000.png ...
    colmap/
        input/camXX.png
        input.db
        manual/{cameras.txt, images.txt, points3D.txt}
        dense/workspace/sparse/{cameras.bin, images.bin, points3D.bin}
        dense/workspace/fused.ply

Usage:
    python scripts/pre_technicolor.py --videopath <path/to/scene> [--force]
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm


SCENE_FRAME_RANGES = {
    "Birthday": range(151, 201),  # inclusive ranges in the original scripts
    "Fabien": range(51, 101),
    "Painter": range(100, 150),
    "Painter_2": range(100, 150),
    "Theater": range(51, 101),
    "Train": range(151, 201),
}


def run(cmd, cwd=None) -> None:
    """Run a shell command with logging."""
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


class COLMAPDatabase(sqlite3.Connection):
    """Utility copied from COLMAP python helpers (see database.py)."""

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(CREATE_ALL)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, params.tobytes(), prior_focal_length),
        )
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id,
             prior_q[0], prior_q[1], prior_q[2], prior_q[3],
             prior_t[0], prior_t[1], prior_t[2]),
        )
        return cursor.lastrowid


# fmt: off
CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""
CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))"""
CREATE_ALL = CREATE_CAMERAS_TABLE + ";" + CREATE_IMAGES_TABLE
# fmt: on


def prepare_image_folders(scene_root: Path, frames: range, force: bool = False) -> None:
    """Reorganise *_undist_XXXXX_YY.png images."""
    image_dir = scene_root / "images"
    colmap_root = scene_root / "colmap"
    colmap_input = colmap_root / "input"

    existing = [path for path in (image_dir, colmap_root) if path.exists()]
    if existing and not force:
        paths = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Derived preprocessing outputs already exist:\n"
            f"{paths}\n"
            "Re-run with --force only if replacing them is intended."
        )
    for path in existing:
        shutil.rmtree(path)

    image_dir.mkdir(parents=True, exist_ok=True)
    colmap_input.mkdir(parents=True, exist_ok=True)

    # ensure cam directories exist
    for cam_idx in range(16):
        (image_dir / f"cam{cam_idx:02d}").mkdir(parents=True, exist_ok=True)

    if not any(scene_root.glob("*_undist_*_*.png")):
        raise FileNotFoundError("No *_undist_* images found in {}".format(scene_root))

    for frame_idx, offset in enumerate(tqdm(frames, desc="Copying frames")):
        matching = sorted(scene_root.glob(f"*_undist_{offset:05d}_*.png"))
        if not matching:
            raise FileNotFoundError(f"No images found for frame {offset:05d}")
        for src in matching:
            cam_token = src.name.split("_")[3]  # e.g. "09.png"
            cam_id = cam_token[:-4]
            dst = image_dir / f"cam{cam_id}" / f"{frame_idx:04d}.png"
            shutil.copyfile(src, dst)
            if frame_idx == 0:
                dst_colmap = colmap_input / f"cam{cam_id}.png"
                shutil.copyfile(src, dst_colmap)


def convert_model(scene_root: Path) -> None:
    """Populate manual images.txt/cameras.txt and initialise input.db."""
    colmap_root = scene_root / "colmap"
    manual_dir = colmap_root / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)

    images_txt = manual_dir / "images.txt"
    cameras_txt = manual_dir / "cameras.txt"
    points_txt = manual_dir / "points3D.txt"
    database_path = colmap_root / "input.db"
    if database_path.exists():
        database_path.unlink()

    db = COLMAPDatabase.connect(str(database_path))
    db.create_tables()

    camera_records = []
    images_records = []

    with open(scene_root / "cameras_parameters.txt", newline="") as f:
        reader = csv.reader(f, delimiter=" ")
        for row_idx, row in enumerate(reader):
            if row_idx == 0:
                continue
            row = [float(x) for x in row if x.strip()]
            fx = fy = row[0]
            cx, cy = row[1], row[2]
            qvec = row[5:9]
            tvec = row[9:12]
            cam_token = f"cam{row_idx-1:02d}"
            image_name = f"{cam_token}.png"
            camera_id = row_idx

            db.add_camera(1, 2048, 1088, np.array([fx, fy, cx, cy]), camera_id=camera_id)
            db.add_image(image_name, camera_id, prior_q=np.array(qvec), prior_t=np.array(tvec), image_id=camera_id)

            images_records.append(
                f"{camera_id} {' '.join(map(str, qvec))} {' '.join(map(str, tvec))} "
                f"{camera_id} {image_name}\n\n"
            )
            camera_records.append(f"{camera_id} PINHOLE 2048 1088 {fx} {fy} {cx} {cy}\n")

    with open(images_txt, "w") as f:
        f.writelines(images_records)
    with open(cameras_txt, "w") as f:
        f.writelines(camera_records)
    points_txt.touch()
    db.commit()
    db.close()


def run_colmap(scene_root: Path, gpu_index: int) -> None:
    """Run COLMAP SfM + dense reconstruction."""
    colmap_root = scene_root / "colmap"
    input_db = colmap_root / "input.db"
    images_dir = colmap_root / "input"
    manual_dir = colmap_root / "manual"
    distorted_dir = colmap_root / "distorted" / "sparse"
    workspace_dir = colmap_root / "dense" / "workspace"
    distorted_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    run(["colmap", "feature_extractor",
         "--database_path", str(input_db),
         "--image_path", str(images_dir)],
        cwd=scene_root)
    run(["colmap", "exhaustive_matcher",
         "--database_path", str(input_db)],
        cwd=scene_root)
    run(["colmap", "point_triangulator",
         "--database_path", str(input_db),
         "--image_path", str(images_dir),
         "--input_path", str(manual_dir),
         "--output_path", str(distorted_dir),
         "--Mapper.ba_global_function_tolerance=0.000001"],
        cwd=scene_root)
    run(["colmap", "image_undistorter",
         "--image_path", str(images_dir),
         "--input_path", str(distorted_dir),
         "--output_path", str(workspace_dir),
         "--output_type", "COLMAP"],
        cwd=scene_root)
    run(["colmap", "patch_match_stereo",
         "--workspace_path", str(workspace_dir),
         "--PatchMatchStereo.gpu_index", str(gpu_index)],
        cwd=scene_root)
    run(["colmap", "stereo_fusion",
         "--workspace_path", str(workspace_dir),
         "--output_path", str(workspace_dir / "fused.ply")],
        cwd=scene_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videopath", type=Path, required=True,
                        help="Path to Technicolor scene folder containing *_undist_* images.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index for COLMAP patch_match_stereo.")
    parser.add_argument("--force", action="store_true",
                        help="Replace existing derived images/ and colmap/ directories.")
    args = parser.parse_args()

    scene_root = args.videopath.resolve()
    if not scene_root.is_dir():
        raise FileNotFoundError(f"{scene_root} is not a directory")

    scene_name = scene_root.name
    if scene_name not in SCENE_FRAME_RANGES:
        raise ValueError(f"Unknown scene '{scene_name}'. Please extend SCENE_FRAME_RANGES.")

    print(f"Preparing Technicolor scene: {scene_name}")
    prepare_image_folders(scene_root, SCENE_FRAME_RANGES[scene_name], force=args.force)
    convert_model(scene_root)
    run_colmap(scene_root, args.gpu)
    print("Done. Dense workspace ready at", scene_root / "colmap" / "dense" / "workspace")


if __name__ == "__main__":
    main()
