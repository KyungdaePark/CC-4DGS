#!/usr/bin/env python3
"""Quantise+pack rest-free PLY point clouds with controllable lossy settings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import zstandard as zstd
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.lossless_pack import read_ply_arrays, write_ply


def _select_uint_dtype(bits: int) -> np.dtype:
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    raise ValueError("bits must be <=32")


def _quantise_linear(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    levels = (1 << bits) - 1
    scaled = np.round((values - mins) / span * levels)
    dtype = _select_uint_dtype(bits)
    quantised = scaled.clip(0, levels).astype(dtype)
    return quantised, mins.astype(np.float32), maxs.astype(np.float32)


def _dequantise_linear(q: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    span = np.where(maxs > mins, maxs - mins, 1.0)
    values = mins + (q.astype(np.float32) / levels) * span
    return values


def _mu_law_encode(values: np.ndarray, mu: float) -> np.ndarray:
    if mu <= 0:
        return values
    return np.sign(values) * (np.log1p(mu * np.abs(values)) / np.log1p(mu))


def _mu_law_decode(values: np.ndarray, mu: float) -> np.ndarray:
    if mu <= 0:
        return values
    return np.sign(values) * (np.power(1 + mu, np.abs(values)) - 1) / mu


def _compress_bytes(arr: np.ndarray, level: int = 9) -> bytes:
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(arr.tobytes())


def _decompress_bytes(blob: bytes, dtype: np.dtype, shape: Sequence[int]) -> np.ndarray:
    decompressor = zstd.ZstdDecompressor()
    raw = decompressor.decompress(blob)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


@dataclass
class GroupConfig:
    names: List[str]
    bits: int
    label: str
    compand_mu: float = 0.0


def pack_restfree(
    input_ply: Path,
    output_dir: Path,
    *,
    xyz_bits: int,
    scale_bits: int,
    opacity_bits: int,
    rotation_bits: int,
    dynamic_bits: int,
    dynamic_mu: float,
    quantize_groups: str,
) -> None:
    arrays, dtypes = read_ply_arrays(input_ply)
    count = len(next(iter(arrays.values())))
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = {item.strip() for item in quantize_groups.split(",") if item.strip()}
    if not requested:
        requested = set()

    groups: List[GroupConfig] = []
    xyz_names = ["x", "y", "z"]
    if all(name in arrays for name in xyz_names):
        groups.append(GroupConfig(xyz_names, xyz_bits, "xyz"))
    scale_names = sorted([name for name in arrays if name.startswith("scale_")])
    if scale_names:
        groups.append(GroupConfig(scale_names, scale_bits, "scale"))
    rot_names = sorted([name for name in arrays if name.startswith("rot_") or name.startswith("rot")])
    if rot_names:
        groups.append(GroupConfig(rot_names, rotation_bits, "rotation"))
    if "opacity" in arrays:
        groups.append(GroupConfig(["opacity"], opacity_bits, "opacity"))
    if "dynamic" in arrays:
        groups.append(GroupConfig(["dynamic"], dynamic_bits, "dynamic", compand_mu=dynamic_mu))

    quant_payload: Dict[str, bytes] = {}
    quant_meta: Dict[str, Dict[str, object]] = {}

    selected_groups = [cfg for cfg in groups if cfg.label in requested or not requested]
    for cfg in selected_groups:
        values = np.stack([arrays[name] for name in cfg.names], axis=1)
        if cfg.compand_mu > 0 and cfg.label == "dynamic":
            values = _mu_law_encode(values, cfg.compand_mu)
        q, mins, maxs = _quantise_linear(values, cfg.bits)
        dtype = q.dtype.str
        quant_meta[cfg.label] = {
            "names": cfg.names,
            "bits": cfg.bits,
            "min": mins.tolist(),
            "max": maxs.tolist(),
            "dtype": dtype,
            "shape": list(q.shape),
            "compand_mu": cfg.compand_mu,
        }
        quant_payload[cfg.label] = _compress_bytes(q)
        for name in cfg.names:
            arrays.pop(name, None)

    residual_payload = {}
    for name, values in arrays.items():
        # Keep remaining attributes in float32 so that not-selected groups stay lossless.
        residual_payload[name] = values.astype(np.float32)

    residual_path = output_dir / "residual.npz"
    np.savez_compressed(residual_path, **residual_payload)

    quant_dir = output_dir / "quantized"
    quant_dir.mkdir(parents=True, exist_ok=True)
    for label, blob in quant_payload.items():
        (quant_dir / f"{label}.bin").write_bytes(blob)

    meta = {
        "count": count,
        "dtypes": dtypes,
        "quantized": quant_meta,
        "residual_fields": list(residual_payload.keys()),
        "residual_dtype": "float32",
        "input": str(input_ply),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[restfree-pack] Packed {input_ply} -> {output_dir}")


def unpack_restfree_to_arrays(pack_dir: Path) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    meta = json.loads((pack_dir / "meta.json").read_text())
    count = meta["count"]
    quant_meta: Dict[str, Dict[str, object]] = meta["quantized"]

    arrays: Dict[str, np.ndarray] = {}
    quant_dir = pack_dir / "quantized"
    for label, cfg in quant_meta.items():
        blob = (quant_dir / f"{label}.bin").read_bytes()
        dtype = np.dtype(cfg["dtype"])
        shape = tuple(cfg["shape"])
        q = _decompress_bytes(blob, dtype, shape)
        mins = np.asarray(cfg["min"], dtype=np.float32)
        maxs = np.asarray(cfg["max"], dtype=np.float32)
        values = _dequantise_linear(q, mins, maxs, int(cfg["bits"]))
        mu = float(cfg.get("compand_mu") or 0.0)
        if mu > 0 and label == "dynamic":
            values = _mu_law_decode(values, mu)
        names = cfg["names"]
        for idx, name in enumerate(names):
            arrays[name] = values[:, idx].astype(np.float32)

    residual = np.load(pack_dir / "residual.npz")
    for name in residual.files:
        arr = residual[name]
        if arr.shape[0] != count:
            raise ValueError(f"Residual field {name} count mismatch.")
        arrays[name] = arr.astype(np.float32)

    return arrays, meta


def unpack_restfree(pack_dir: Path, output_ply: Path) -> None:
    arrays, meta = unpack_restfree_to_arrays(pack_dir)
    # Ensure all dtypes exist before writing.
    write_ply(output_ply, arrays, [tuple(dt) for dt in meta["dtypes"]])
    print(f"[restfree-pack] Unpacked {pack_dir} -> {output_ply}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantise+pack rest-free PLY files.")
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack", help="Pack a rest-free PLY into a quantised archive.")
    pack.add_argument("--input", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    pack.add_argument("--xyz-bits", type=int, default=14)
    pack.add_argument("--scale-bits", type=int, default=12)
    pack.add_argument("--opacity-bits", type=int, default=12)
    pack.add_argument("--rotation-bits", type=int, default=12)
    pack.add_argument("--dynamic-bits", type=int, default=16)
    pack.add_argument("--dynamic-mu", type=float, default=255.0, help="μ-law companding factor for dynamic logits (0 to disable).")
    pack.add_argument(
        "--quantize-groups",
        type=str,
        default="xyz,scale,opacity,rotation,dynamic",
        help="Comma-separated list of groups to quantise (others remain lossless).",
    )

    unpack = sub.add_parser("unpack", help="Reconstruct a PLY from a quantised archive.")
    unpack.add_argument("--input", type=Path, required=True, help="Directory produced by the pack command.")
    unpack.add_argument("--output", type=Path, required=True, help="Destination PLY path.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "pack":
        pack_restfree(
            input_ply=args.input,
            output_dir=args.output_dir,
            xyz_bits=args.xyz_bits,
            scale_bits=args.scale_bits,
            opacity_bits=args.opacity_bits,
            rotation_bits=args.rotation_bits,
            dynamic_bits=args.dynamic_bits,
            dynamic_mu=args.dynamic_mu,
            quantize_groups=args.quantize_groups,
        )
    elif args.command == "unpack":
        unpack_restfree(args.input, args.output)


if __name__ == "__main__":
    main()
