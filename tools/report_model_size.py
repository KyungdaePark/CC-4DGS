#!/usr/bin/env python3
"""Report raw and deployable CC-4DGS checkpoint sizes."""

from __future__ import annotations

import argparse
from pathlib import Path


def file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def format_size(size: int) -> str:
    return f"{size / 1_000_000:.2f} MB ({size / (1024 * 1024):.2f} MiB; {size:,} bytes)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report canonical, Deform-module, and total deployable checkpoint sizes."
    )
    parser.add_argument("--model", type=Path, required=True, help="Trained model directory.")
    parser.add_argument("--iteration", type=int, required=True, help="Checkpoint iteration.")
    args = parser.parse_args()

    iteration_dir = args.model / "point_cloud" / f"iteration_{args.iteration}"
    if not iteration_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {iteration_dir}")

    raw_canonical = file_size(iteration_dir / "point_cloud.ply")
    model_state_names = ("deformation.pth", "FDHash.pth")
    deform = sum(file_size(iteration_dir / name) for name in model_state_names)
    compressed_canonical = tree_size(iteration_dir / "compressed")

    print(f"Checkpoint: {iteration_dir}")
    print(f"Raw canonical point cloud:       {format_size(raw_canonical)}")
    print(f"Deform-module storage:           {format_size(deform)}")
    print(f"Raw total model size:            {format_size(raw_canonical + deform)}")
    if compressed_canonical:
        print(f"CCA canonical-attribute payload: {format_size(compressed_canonical)}")
        print(f"Total deployable model size:     {format_size(compressed_canonical + deform)}")
    else:
        print(f"Compressed bundle not found: {iteration_dir / 'compressed'}")


if __name__ == "__main__":
    main()

