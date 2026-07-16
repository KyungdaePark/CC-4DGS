#!/usr/bin/env python3
"""Voxel downsample a dense point cloud."""

import argparse

import open3d as o3d


def downsample(input_path: str, output_path: str, target: int = 100_000) -> None:
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"Input points: {len(pcd.points)}")

    voxel_size = 0.001
    while len(pcd.points) > target:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled ({voxel_size:.4f}): {len(pcd.points)}")
        voxel_size += 0.005

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input fused point cloud (e.g. fused.ply)")
    parser.add_argument("output", help="Output path for the downsampled PLY")
    parser.add_argument("--target", type=int, default=100_000,
                        help="Maximum number of points after downsampling")
    args = parser.parse_args()
    downsample(args.input, args.output, args.target)


if __name__ == "__main__":
    main()

