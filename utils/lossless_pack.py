#!/usr/bin/env python3
"""Shared helpers for lossless Morton+bitshuffle packing of PLY point clouds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import bitshuffle
import numpy as np
import zstandard as zstd
from plyfile import PlyData, PlyElement

CAST_DTYPES: Dict[str, np.dtype] = {
    "f32": np.float32,
    "f16": np.float16,
    "u16": np.uint16,
    "i16": np.int16,
}


def read_ply_arrays(path: Path) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, str]]]:
    """Return copies of all vertex attributes and their dtype codes."""
    ply = PlyData.read(path)
    elem = ply["vertex"]
    arrays: Dict[str, np.ndarray] = {}
    dtypes: List[Tuple[str, str]] = []
    for name in elem.data.dtype.names:
        arr = np.asarray(elem.data[name])
        arrays[name] = arr.astype(np.float32, copy=True)
        dtypes.append((name, "f4"))
    return arrays, dtypes


def write_ply(path: Path, arrays: Dict[str, np.ndarray], dtypes: List[Tuple[str, str]]) -> None:
    """Write arrays back to a binary little-endian PLY."""
    count = len(next(iter(arrays.values())))
    vertex = np.empty(count, dtype=np.dtype(dtypes))
    for name, _ in dtypes:
        vertex[name] = arrays[name]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def _expand_bits(v: np.ndarray) -> np.ndarray:
    x = v & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


def _morton_order(xyz: np.ndarray) -> np.ndarray:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    norm = np.clip((xyz - mins) / span, 0.0, 1.0)
    ints = np.clip((norm * ((1 << 21) - 1)).astype(np.uint32), 0, (1 << 21) - 1)
    codes = (_expand_bits(ints[:, 0]) << 2) | (_expand_bits(ints[:, 1]) << 1) | _expand_bits(ints[:, 2])
    return np.argsort(codes, kind="mergesort")


def _xor_delta(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    if out.size == 0:
        return out
    out[1:] = values[1:] ^ values[:-1]
    return out


def _xor_inverse(delta: np.ndarray) -> np.ndarray:
    if delta.size == 0:
        return delta
    out = delta.copy()
    for i in range(1, out.size):
        out[i] = out[i] ^ out[i - 1]
    return out


def _compress_bytes(data: bytes, block_size: int, level: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    shuffled = bitshuffle.bitshuffle(arr, block_size)
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(shuffled.tobytes())


def _decompress_bytes(data: bytes, block_size: int) -> bytes:
    decompressor = zstd.ZstdDecompressor()
    shuffled = decompressor.decompress(data)
    arr = np.frombuffer(shuffled, dtype=np.uint8)
    unshuffled = bitshuffle.bitunshuffle(arr, block_size)
    return unshuffled.tobytes()


def pack_ply(
    ply_path: Path,
    pack_dir: Path,
    *,
    block_size: int,
    level: int,
    use_morton: bool,
    cast_map: Dict[str, str],
    zero_mean_fields: set[str],
) -> None:
    """Pack a PLY into Morton+bitshuffle representation."""
    arrays, dtypes = read_ply_arrays(ply_path)
    pack_dir.mkdir(parents=True, exist_ok=True)
    xyz = np.stack([arrays["x"], arrays["y"], arrays["z"]], axis=1)
    order = _morton_order(xyz) if use_morton else np.arange(xyz.shape[0], dtype=np.uint32)
    metadata = {
        "count": int(xyz.shape[0]),
        "block_size": block_size,
        "zstd_level": level,
        "use_morton": use_morton,
        "fields": [],
        "dtypes": dtypes,
    }
    order_bytes = order.astype(np.uint32).tobytes()
    order_comp = _compress_bytes(order_bytes, block_size, level)
    (pack_dir / "order.bin").write_bytes(order_comp)
    arrays_reordered: Dict[str, np.ndarray] = {}
    for name in arrays:
        arrays_reordered[name] = arrays[name][order]
    for idx, (name, _) in enumerate(dtypes):
        arr = arrays_reordered[name]
        mean_val = None
        if name in zero_mean_fields:
            mean_val = float(arr.mean(dtype=np.float64))
            arr = arr - mean_val
        cast_code = cast_map.get(name)
        if cast_code:
            cast_dtype = CAST_DTYPES[cast_code]
            stored = arr.astype(cast_dtype)
        else:
            stored = arr
        raw_bytes = stored.tobytes()
        pad = (-len(raw_bytes)) % 4
        if pad:
            raw_bytes += b"\x00" * pad
        as_uint = np.frombuffer(raw_bytes, dtype=np.uint32)
        delta = _xor_delta(as_uint)
        comp = _compress_bytes(delta.tobytes(), block_size, level)
        field_file = f"field_{idx:03d}_{name}.bin"
        (pack_dir / field_file).write_bytes(comp)
        metadata["fields"].append(
            {
                "name": name,
                "file": field_file,
                "cast": cast_code,
                "pad": pad,
                "mean": mean_val,
            }
        )
    (pack_dir / "meta.json").write_text(json.dumps(metadata, indent=2))


def unpack_ply_to_arrays(pack_dir: Path) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, str]]]:
    """Return arrays reconstructed from a packed directory without writing to disk."""
    meta = json.loads((pack_dir / "meta.json").read_text())
    block_size = meta["block_size"]
    fields = meta["fields"]
    dtypes = [tuple(item) for item in meta["dtypes"]]
    order_comp = (pack_dir / "order.bin").read_bytes()
    order_bytes = _decompress_bytes(order_comp, block_size)
    order = np.frombuffer(order_bytes, dtype=np.uint32)
    arrays: Dict[str, np.ndarray] = {}
    for field in fields:
        name = field["name"]
        comp = (pack_dir / field["file"]).read_bytes()
        delta_bytes = _decompress_bytes(comp, block_size)
        delta = np.frombuffer(delta_bytes, dtype=np.uint32)
        values = _xor_inverse(delta)
        raw_bytes = values.tobytes()
        pad = field.get("pad", 0)
        if pad:
            raw_bytes = raw_bytes[:-pad]
        cast_code = field.get("cast")
        if cast_code:
            cast_dtype = CAST_DTYPES[cast_code]
            stored = np.frombuffer(raw_bytes, dtype=cast_dtype)
            arr_reordered = stored.astype(np.float32)
        else:
            arr_reordered = np.frombuffer(raw_bytes, dtype=np.float32)
        mean_val = field.get("mean")
        if mean_val is not None:
            arr_reordered = arr_reordered + mean_val
        arr = np.empty_like(arr_reordered)
        arr[order] = arr_reordered
        arrays[name] = arr
    return arrays, dtypes


def unpack_ply(pack_dir: Path, output_ply: Path) -> None:
    """Write a packed point cloud back to a PLY file."""
    arrays, dtypes = unpack_ply_to_arrays(pack_dir)
    write_ply(output_ply, arrays, dtypes)


__all__ = [
    "CAST_DTYPES",
    "pack_ply",
    "read_ply_arrays",
    "unpack_ply",
    "unpack_ply_to_arrays",
    "write_ply",
]
