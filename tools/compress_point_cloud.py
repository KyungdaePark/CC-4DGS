#!/usr/bin/env python3
"""
Utility to pack Gaussian point clouds (PLY) into a quantised NPZ archive
and optionally restore them back. The compression pipeline is used by the
CC-4DGS post-training CCA tooling.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from plyfile import PlyData, PlyElement


def _load_ply(path: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    names = list(vertex.dtype.names)
    arrays = {name: vertex[name].astype(np.float32).copy() for name in names}
    return arrays, names


def _write_ply(path: Path, arrays: Dict[str, np.ndarray], order: Iterable[str]) -> None:
    dtype = [(name, "<f4") for name in order]
    count = len(next(iter(arrays.values())))
    structured = np.empty(count, dtype=dtype)
    for name in order:
        structured[name] = arrays[name].astype(np.float32)
    vertex = PlyElement.describe(structured, "vertex")
    ply = PlyData([vertex], text=False)
    ply.write(str(path))


def _select_uint_dtype(bits: int) -> np.dtype:
    """Return the smallest unsigned dtype that can hold `bits` levels."""
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    raise ValueError(f"Quantisation with {bits} bits is not supported (max 32).")


def _quantise_linear(values: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly quantise along the last axis."""
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    spans = np.where(maxs > mins, maxs - mins, 1.0)
    levels = (1 << bits) - 1
    scaled = np.round((values - mins) / spans * levels)
    q_dtype = _select_uint_dtype(bits)
    quantised = scaled.clip(0, levels).astype(q_dtype)
    return quantised, mins.astype(np.float32), maxs.astype(np.float32)


def _dequantise_linear(quantised: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    spans = np.where(maxs > mins, maxs - mins, 1.0)
    values = mins + (quantised.astype(np.float32) / levels) * spans
    return values


def compress_point_cloud(
    input_path: Path,
    output_path: Path,
    *,
    pos_bits: int = 16,
    feature_bits: int = 8,
    coeff_bits: int = 12,
    pca_components: int = 16,
    drop_normals: bool = True,
    keep_percent: float = 1.0,
    top_k: int | None = None,
    sort_by: str = "opacity",
    min_field_value: float | None = None,
) -> None:
    arrays, original_names = _load_ply(input_path)
    count = len(next(iter(arrays.values())))

    if sort_by not in arrays:
        raise KeyError(f"Cannot sort by '{sort_by}' because the field is missing.")

    # Select subset to keep, default is keep all.
    keep_mask = np.ones(count, dtype=bool)
    if min_field_value is not None:
        keep_mask &= arrays[sort_by] >= min_field_value

    if top_k is not None and top_k < count:
        top_indices = np.argsort(arrays[sort_by])[::-1][:top_k]
        keep_mask = np.zeros(count, dtype=bool)
        keep_mask[top_indices] = True
    elif keep_percent < 1.0:
        keep_n = int(math.ceil(count * keep_percent))
        if keep_n <= 0:
            raise ValueError("keep_percent too small; nothing would be kept.")
        sorted_idx = np.argsort(arrays[sort_by])[::-1][:keep_n]
        keep_mask = np.zeros(count, dtype=bool)
        keep_mask[sorted_idx] = True

    if not keep_mask.all():
        for key in arrays:
            arrays[key] = arrays[key][keep_mask]

    pos_names = ["x", "y", "z"]
    missing = [name for name in pos_names if name not in arrays]
    if missing:
        raise KeyError(f"The point cloud is missing mandatory coordinate(s): {missing}")

    normal_names = [name for name in ("nx", "ny", "nz") if name in arrays]
    dropped_normals = normal_names.copy()
    rest_names = sorted(name for name in arrays if name.startswith("f_rest"))
    dynamic_names = [name for name in arrays if name == "dynamic"]
    other_names = [
        name
        for name in arrays
        if name not in pos_names + normal_names + rest_names + dynamic_names
    ]

    if drop_normals:
        for name in normal_names:
            arrays.pop(name, None)
        normal_names = []

    pos = np.stack([arrays[name] for name in pos_names], axis=1)
    rest = np.stack([arrays[name] for name in rest_names], axis=1) if rest_names else None
    others = np.stack([arrays[name] for name in other_names], axis=1) if other_names else None
    dynamic = (
        np.stack([arrays[name] for name in dynamic_names], axis=1)
        if dynamic_names
        else None
    )

    compressed: Dict[str, np.ndarray] = {}
    meta: Dict[str, object] = {
        "version": 1,
        "pos_names": pos_names,
        "rest_names": rest_names,
        "other_names": other_names,
        "normal_names": dropped_normals if drop_normals else normal_names,
        "original_names": original_names,
        "pos_bits": pos_bits,
        "feature_bits": feature_bits,
        "coeff_bits": coeff_bits,
        "pca_components": pca_components if rest is not None else 0,
        "sorted_by": sort_by,
        "kept_points": int(pos.shape[0]),
        "original_points": count,
        "keep_percent": keep_percent,
        "top_k": top_k,
        "min_field_value": min_field_value,
        "dynamic_names": dynamic_names,
    }

    pos_q, pos_min, pos_max = _quantise_linear(pos, pos_bits)
    compressed["pos"] = pos_q
    compressed["pos_min"] = pos_min
    compressed["pos_max"] = pos_max

    if others is not None:
        features_q, features_min, features_max = _quantise_linear(others, feature_bits)
        compressed["features"] = features_q
        compressed["features_min"] = features_min
        compressed["features_max"] = features_max
    else:
        compressed["features"] = np.empty((pos.shape[0], 0), dtype=np.uint8)
        compressed["features_min"] = np.empty(0, dtype=np.float32)
        compressed["features_max"] = np.empty(0, dtype=np.float32)

    rest_mode = "none"
    if rest is not None and pca_components > 0:
        rest_mode = "pca"
        mean = rest.mean(axis=0)
        centered = rest - mean
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(pca_components, vt.shape[0])
        coeff = u[:, :k] * s[:k]
        coeff_q, coeff_min, coeff_max = _quantise_linear(coeff, coeff_bits)
        compressed["coeff"] = coeff_q
        compressed["coeff_min"] = coeff_min
        compressed["coeff_max"] = coeff_max
        compressed["rest_mean"] = mean.astype(np.float32)
        compressed["rest_basis"] = vt[:k].astype(np.float32)
    elif rest is not None:
        rest_mode = "direct"
        meta["pca_components"] = 0
        rest_q, rest_min, rest_max = _quantise_linear(rest, coeff_bits)
        compressed["rest_direct"] = rest_q
        compressed["rest_direct_min"] = rest_min
        compressed["rest_direct_max"] = rest_max
        compressed["coeff"] = np.empty((pos.shape[0], 0), dtype=np.uint8)
        compressed["coeff_min"] = np.empty(0, dtype=np.float32)
        compressed["coeff_max"] = np.empty(0, dtype=np.float32)
        compressed["rest_mean"] = np.empty(0, dtype=np.float32)
        compressed["rest_basis"] = np.empty((0, 0), dtype=np.float32)
    else:
        meta["pca_components"] = 0
        compressed["coeff"] = np.empty((pos.shape[0], 0), dtype=np.uint8)
        compressed["coeff_min"] = np.empty(0, dtype=np.float32)
        compressed["coeff_max"] = np.empty(0, dtype=np.float32)
        compressed["rest_mean"] = np.empty(0, dtype=np.float32)
        compressed["rest_basis"] = np.empty((0, 0), dtype=np.float32)

    meta["rest_mode"] = rest_mode
    meta["rest_bits"] = coeff_bits

    if dynamic is not None:
        compressed["dynamic"] = dynamic.astype(np.float32)

    compressed["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(output_path, **compressed)


def decompress_point_cloud(input_path: Path, output_path: Path) -> None:
    packed = np.load(input_path, allow_pickle=False)
    meta = json.loads(str(packed["meta"]))

    pos_bits = int(meta["pos_bits"])
    feature_bits = int(meta["feature_bits"])
    coeff_bits = int(meta["coeff_bits"])

    pos = _dequantise_linear(packed["pos"], packed["pos_min"], packed["pos_max"], pos_bits)

    features = _dequantise_linear(packed["features"], packed["features_min"], packed["features_max"], feature_bits)
    coeff = packed["coeff"]
    if coeff.size:
        coeff = _dequantise_linear(coeff, packed["coeff_min"], packed["coeff_max"], coeff_bits)
    else:
        coeff = np.empty((pos.shape[0], 0), dtype=np.float32)

    rest_names: List[str] = meta["rest_names"]
    rest_mode = meta.get("rest_mode", "pca" if int(meta.get("pca_components", 0)) > 0 else "none")
    rest_bits = int(meta.get("rest_bits", coeff_bits))
    rest = None
    if rest_names and rest_mode == "pca":
        basis = packed["rest_basis"]
        mean = packed["rest_mean"]
        if coeff.size == 0:
            raise RuntimeError("Compressed file lacks PCA coefficients required to reconstruct f_rest.")
        rest = coeff @ basis + mean
    elif rest_names and rest_mode == "direct":
        rest = _dequantise_linear(
            packed["rest_direct"],
            packed["rest_direct_min"],
            packed["rest_direct_max"],
            rest_bits,
        )

    arrays: Dict[str, np.ndarray] = {}

    pos_names: List[str] = meta["pos_names"]
    for idx, name in enumerate(pos_names):
        arrays[name] = pos[:, idx]

    offset = 0
    other_names: List[str] = meta["other_names"]
    for name in other_names:
        arrays[name] = features[:, offset]
        offset += 1

    if rest is not None:
        for idx, name in enumerate(rest_names):
            arrays[name] = rest[:, idx]

    dynamic_names: List[str] = meta["dynamic_names"]
    if dynamic_names:
        dynamic_arr = packed["dynamic"]
        for idx, name in enumerate(dynamic_names):
            arrays[name] = dynamic_arr[:, idx]

    normal_names: List[str] = meta["normal_names"]
    for name in normal_names:
        arrays[name] = np.zeros(pos.shape[0], dtype=np.float32)

    original_names: List[str] = meta["original_names"]
    missing = [name for name in original_names if name not in arrays]
    if missing:
        raise KeyError(f"Failed to rebuild fields {missing}; metadata may be incompatible.")

    _write_ply(output_path, arrays, original_names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress or decompress CC-4DGS point clouds.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    comp = subparsers.add_parser("compress", help="Compress a binary PLY into a quantised NPZ archive.")
    comp.add_argument("input", type=Path)
    comp.add_argument("output", type=Path)
    comp.add_argument("--pos-bits", type=int, default=16)
    comp.add_argument("--feature-bits", type=int, default=8)
    comp.add_argument("--coeff-bits", type=int, default=12)
    comp.add_argument("--pca-components", type=int, default=16)
    comp.add_argument("--keep-percent", type=float, default=1.0)
    comp.add_argument("--top-k", type=int, default=None)
    comp.add_argument("--sort-by", type=str, default="opacity")
    comp.add_argument("--min-field-value", type=float, default=None)
    comp.add_argument("--keep-normals", action="store_false", dest="drop_normals")

    decomp = subparsers.add_parser("decompress", help="Rebuild a binary PLY from a compressed archive.")
    decomp.add_argument("input", type=Path)
    decomp.add_argument("output", type=Path)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "compress":
        compress_point_cloud(
            args.input,
            args.output,
            pos_bits=args.pos_bits,
            feature_bits=args.feature_bits,
            coeff_bits=args.coeff_bits,
            pca_components=args.pca_components,
            drop_normals=args.drop_normals,
            keep_percent=args.keep_percent,
            top_k=args.top_k,
            sort_by=args.sort_by,
            min_field_value=args.min_field_value,
        )
    elif args.command == "decompress":
        decompress_point_cloud(args.input, args.output)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
