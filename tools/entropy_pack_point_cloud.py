#!/usr/bin/env python3
"""
Entropy-coded variable-bit compressor for CC-4DGS point clouds.
This prototype quantises coordinates/features/rest attributes, then
encodes each dimension with canonical Huffman coding to produce a
compact bitstream. Decompression reconstructs the PLY exactly as
quantised (loss defined solely by the chosen bit depths).
"""

from __future__ import annotations

import argparse
import json
import math
import heapq
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import subprocess
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compress_point_cloud import (
    _load_ply,
    _write_ply,
    _quantise_linear,
    _dequantise_linear,
)


class BitWriter:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.current = 0
        self.nbits = 0

    def write(self, value: int, bits: int) -> None:
        for shift in range(bits - 1, -1, -1):
            bit = (value >> shift) & 1
            self.current = (self.current << 1) | bit
            self.nbits += 1
            if self.nbits == 8:
                self.buffer.append(self.current)
                self.current = 0
                self.nbits = 0

    def finish(self) -> Tuple[bytes, int]:
        pad = 0
        if self.nbits:
            pad = 8 - self.nbits
            self.current <<= pad
            self.buffer.append(self.current)
            self.current = 0
            self.nbits = 0
        return bytes(self.buffer), pad


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0

    def read_bit(self) -> int:
        if self.byte_pos >= len(self.data):
            raise EOFError("BitReader exhausted.")
        byte = self.data[self.byte_pos]
        bit = (byte >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit


def _build_huffman_lengths(values: np.ndarray, alphabet_size: int) -> List[int]:
    counts = np.bincount(values, minlength=alphabet_size).astype(np.int64)
    heap: List[Tuple[int, int, int | Tuple[int, int]]] = []
    counter = 0
    for symbol, freq in enumerate(counts):
        if freq > 0:
            heapq.heappush(heap, (freq, counter, symbol))
            counter += 1
    if not heap:
        raise ValueError("Empty frequency table; cannot build Huffman codes.")
    if len(heap) == 1:
        freq, _, symbol = heap[0]
        lengths = [0] * alphabet_size
        lengths[symbol] = 1
        return lengths
    while len(heap) > 1:
        freq1, _, node1 = heapq.heappop(heap)
        freq2, _, node2 = heapq.heappop(heap)
        heapq.heappush(heap, (freq1 + freq2, counter, (node1, node2)))
        counter += 1
    root = heap[0][2]
    lengths = [0] * alphabet_size

    def assign(node: int | Tuple[int, int], depth: int) -> None:
        if isinstance(node, int):
            lengths[node] = depth
            return
        left, right = node
        assign(left, depth + 1)
        assign(right, depth + 1)

    assign(root, 0)
    return lengths


class CanonicalHuffman:
    def __init__(self, lengths: List[int]) -> None:
        self.lengths = lengths
        self.code_map: Dict[int, Tuple[int, int]] = {}
        self.decode_tables: Dict[int, Dict[int, int]] = {}
        self.max_len = 0
        self._build()

    def _build(self) -> None:
        entries = [(l, sym) for sym, l in enumerate(self.lengths) if l > 0]
        if not entries:
            raise ValueError("No symbols to encode.")
        entries.sort()
        code = 0
        prev_len = entries[0][0]
        code <<= prev_len
        for length, symbol in entries:
            if length > prev_len:
                code <<= length - prev_len
                prev_len = length
            self.code_map[symbol] = (code, length)
            self.decode_tables.setdefault(length, {})[code] = symbol
            code += 1
        self.max_len = prev_len

    def encode(self, values: np.ndarray) -> Tuple[bytes, int]:
        writer = BitWriter()
        for val in values:
            code, length = self.code_map[val]
            writer.write(code, length)
        return writer.finish()

    def decode(self, data: bytes, count: int) -> np.ndarray:
        reader = BitReader(data)
        out = np.empty(count, dtype=np.int64)
        for idx in range(count):
            code = 0
            for length in range(1, self.max_len + 1):
                bit = reader.read_bit()
                code = (code << 1) | bit
                table = self.decode_tables.get(length)
                if table and code in table:
                    out[idx] = table[code]
                    break
            else:
                raise ValueError("Failed to decode symbol.")
        return out


def _encode_dimension(values: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray, bytes, int]:
    alphabet_size = 1 << bits
    lengths = _build_huffman_lengths(values, alphabet_size)
    codec = CanonicalHuffman(lengths)
    stream, pad = codec.encode(values)
    return np.array(lengths, dtype=np.uint8), np.array([values.size], dtype=np.int64), stream, pad


def _decode_dimension(stream: bytes, lengths: np.ndarray, count: int) -> np.ndarray:
    codec = CanonicalHuffman(lengths.tolist())
    return codec.decode(stream, count).astype(np.int64)


def _stack_fields(arrays: Dict[str, np.ndarray], names: Iterable[str]) -> np.ndarray:
    return np.stack([arrays[name] for name in names], axis=1)


def compress_point_cloud(
    input_path: Path,
    output_path: Path,
    *,
    pos_bits: int,
    feature_bits: int,
    rest_bits: int,
) -> None:
    print(f"[entropy] Loading {input_path} ...")
    arrays, original_names = _load_ply(input_path)
    pos_names = ["x", "y", "z"]
    rest_names = sorted([n for n in arrays if n.startswith("f_rest_")], key=lambda x: int(x.split("_")[-1]))
    other_names = [n for n in original_names if n not in pos_names + rest_names]

    pos = _stack_fields(arrays, pos_names)
    others = _stack_fields(arrays, other_names) if other_names else None
    rest = _stack_fields(arrays, rest_names) if rest_names else None

    meta: Dict[str, object] = {
        "version": 1,
        "original_names": original_names,
        "pos_names": pos_names,
        "rest_names": rest_names,
        "other_names": other_names,
        "pos_bits": pos_bits,
        "feature_bits": feature_bits,
        "other_bits": feature_bits,
        "rest_bits": rest_bits,
        "count": int(pos.shape[0]),
    }

    compressed: Dict[str, np.ndarray] = {}

    print("[entropy] Quantising positions ...")
    pos_q, pos_min, pos_max = _quantise_linear(pos, pos_bits)
    compressed["pos_min"] = pos_min
    compressed["pos_max"] = pos_max
    for dim in range(pos_q.shape[1]):
        lengths, counts, stream, pad = _encode_dimension(pos_q[:, dim].astype(np.int64), pos_bits)
        compressed[f"pos_lengths_{dim}"] = lengths
        compressed[f"pos_counts_{dim}"] = counts
        compressed[f"pos_stream_{dim}"] = np.frombuffer(stream, dtype=np.uint8)
        compressed[f"pos_pad_{dim}"] = np.array([pad], dtype=np.uint8)

    if others is not None:
        print("[entropy] Quantising other attributes ...")
        other_q, other_min, other_max = _quantise_linear(others, feature_bits)
        compressed["other_min"] = other_min
        compressed["other_max"] = other_max
        for dim in range(other_q.shape[1]):
            lengths, counts, stream, pad = _encode_dimension(other_q[:, dim].astype(np.int64), feature_bits)
            compressed[f"other_lengths_{dim}"] = lengths
            compressed[f"other_counts_{dim}"] = counts
            compressed[f"other_stream_{dim}"] = np.frombuffer(stream, dtype=np.uint8)
            compressed[f"other_pad_{dim}"] = np.array([pad], dtype=np.uint8)
    else:
        compressed["other_min"] = np.empty(0, dtype=np.float32)
        compressed["other_max"] = np.empty(0, dtype=np.float32)

    if rest is not None:
        print("[entropy] Quantising f_rest* ...")
        rest_q, rest_min, rest_max = _quantise_linear(rest, rest_bits)
        compressed["rest_min"] = rest_min
        compressed["rest_max"] = rest_max
        for dim in range(rest_q.shape[1]):
            lengths, counts, stream, pad = _encode_dimension(rest_q[:, dim].astype(np.int64), rest_bits)
            compressed[f"rest_lengths_{dim}"] = lengths
            compressed[f"rest_counts_{dim}"] = counts
            compressed[f"rest_stream_{dim}"] = np.frombuffer(stream, dtype=np.uint8)
            compressed[f"rest_pad_{dim}"] = np.array([pad], dtype=np.uint8)
    else:
        compressed["rest_min"] = np.empty(0, dtype=np.float32)
        compressed["rest_max"] = np.empty(0, dtype=np.float32)

    compressed["meta"] = np.array(json.dumps(meta))
    print(f"[entropy] Saving archive to {output_path} ...")
    np.savez_compressed(output_path, **compressed)
    print("[entropy] Compression done.")


def _decode_group(prefix: str, meta: Dict[str, object], packed) -> np.ndarray | None:
    names: List[str] = meta[prefix + "_names"]
    if not names:
        return None
    bits = int(meta[f"{prefix}_bits"])
    mins = packed[f"{prefix}_min"]
    maxs = packed[f"{prefix}_max"]
    decoded_cols = []
    for dim in range(len(names)):
        lengths = packed[f"{prefix}_lengths_{dim}"]
        counts = int(packed[f"{prefix}_counts_{dim}"][0])
        stream = bytes(packed[f"{prefix}_stream_{dim}"])
        values = _decode_dimension(stream, lengths, counts).astype(np.int64)
        decoded = _dequantise_linear(values.reshape(-1, 1), mins[dim : dim + 1], maxs[dim : dim + 1], bits)
        decoded_cols.append(decoded[:, 0])
    return np.stack(decoded_cols, axis=1)


def decompress_point_cloud(input_path: Path, output_path: Path) -> None:
    print(f"[entropy] Loading archive {input_path} ...")
    packed = np.load(input_path, allow_pickle=False)
    meta = json.loads(str(packed["meta"]))
    arrays: Dict[str, np.ndarray] = {}

    pos = _decode_group("pos", meta, packed)
    for idx, name in enumerate(meta["pos_names"]):
        arrays[name] = pos[:, idx]

    others = _decode_group("other", meta, packed)
    if others is not None:
        for idx, name in enumerate(meta["other_names"]):
            arrays[name] = others[:, idx]

    rest = _decode_group("rest", meta, packed)
    if rest is not None:
        for idx, name in enumerate(meta["rest_names"]):
            arrays[name] = rest[:, idx]

    print(f"[entropy] Writing reconstructed PLY to {output_path} ...")
    _write_ply(output_path, arrays, meta["original_names"])
    print("[entropy] Decompression done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entropy-coded point cloud compressor.")
    sub = parser.add_subparsers(dest="command", required=True)

    comp = sub.add_parser("compress")
    comp.add_argument("input", type=Path)
    comp.add_argument("output", type=Path)
    comp.add_argument("--pos-bits", type=int, default=16)
    comp.add_argument("--feature-bits", type=int, default=10)
    comp.add_argument("--rest-bits", type=int, default=12)
    comp.add_argument("--ram", action="store_true", help="After compression, render & metric the reconstructed PLY.")
    comp.add_argument("--model-path", type=Path)
    comp.add_argument("--dataset-path", type=Path)
    comp.add_argument("--config-path", type=Path)
    comp.add_argument("--iteration", type=int)
    comp.add_argument("--deformation-source", type=Path)
    comp.add_argument(
        "--keep-temp-ply",
        action="store_true",
        help="Keep the reconstructed PLY produced for --ram.",
    )

    decomp = sub.add_parser("decompress")
    decomp.add_argument("input", type=Path)
    decomp.add_argument("output", type=Path)
    decomp.add_argument("--ram", action="store_true", help="After decompression, render & metric.")
    decomp.add_argument("--model-path", type=Path)
    decomp.add_argument("--dataset-path", type=Path)
    decomp.add_argument("--config-path", type=Path)
    decomp.add_argument("--iteration", type=int)
    decomp.add_argument("--deformation-source", type=Path)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "compress":
        input_path = args.input
        default_deform = None
        if input_path.is_dir():
            ply_candidate = input_path / "point_cloud.ply"
            if not ply_candidate.exists():
                parser.error(f"Iteration folder {input_path} is missing point_cloud.ply")
            input_path = ply_candidate
            possible_def = input_path.parent / "deformation.pth"
            if possible_def.exists():
                default_deform = possible_def
        else:
            possible_def = input_path.parent / "deformation.pth"
            if possible_def.exists():
                default_deform = possible_def

        compress_point_cloud(
            input_path,
            args.output,
            pos_bits=args.pos_bits,
            feature_bits=args.feature_bits,
            rest_bits=args.rest_bits,
        )
        if args.ram:
            required = [args.model_path, args.dataset_path, args.config_path, args.iteration]
            if any(v is None for v in required):
                parser.error("--ram requires --model-path, --dataset-path, --config-path, --iteration")
            temp_ply = args.output.with_suffix(".ply")
            decompress_point_cloud(args.output, temp_ply)

            iter_dir = args.model_path / "point_cloud" / f"iteration_{args.iteration}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_ply, iter_dir / "point_cloud.ply")
            deform_src = args.deformation_source or default_deform or (input_path.parent / "deformation.pth")
            if not deform_src.exists():
                raise FileNotFoundError(f"Missing deformation checkpoint: {deform_src}")
            shutil.copy2(deform_src, iter_dir / "deformation.pth")
            render_cmd = [
                sys.executable,
                str(REPO_ROOT / "render.py"),
                "-s",
                str(args.dataset_path),
                "--configs",
                str(args.config_path),
                "--model_path",
                str(args.model_path),
                "--iteration",
                str(args.iteration),
                "--skip_train",
                "--skip_video",
            ]
            metrics_cmd = [
                sys.executable,
                str(REPO_ROOT / "metrics.py"),
                "-m",
                str(args.model_path),
            ]
            subprocess.run(render_cmd, cwd=REPO_ROOT, check=True)
            subprocess.run(metrics_cmd, cwd=REPO_ROOT, check=True)
            if not args.keep_temp_ply:
                temp_ply.unlink(missing_ok=True)
    elif args.command == "decompress":
        if args.ram:
            required = [args.model_path, args.dataset_path, args.config_path, args.iteration]
            if any(v is None for v in required):
                parser.error("--ram requires --model-path, --dataset-path, --config-path, --iteration")
            iter_dir = args.model_path / "point_cloud" / f"iteration_{args.iteration}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            target_ply = iter_dir / "point_cloud.ply"
            decompress_point_cloud(args.input, target_ply)
            if args.output.resolve() != target_ply.resolve():
                shutil.copy2(target_ply, args.output)
            deform_src = args.deformation_source
            if deform_src and deform_src.exists():
                shutil.copy2(deform_src, iter_dir / "deformation.pth")
            render_cmd = [
                sys.executable,
                str(REPO_ROOT / "render.py"),
                "-s",
                str(args.dataset_path),
                "--configs",
                str(args.config_path),
                "--model_path",
                str(args.model_path),
                "--iteration",
                str(args.iteration),
                "--skip_train",
                "--skip_video",
            ]
            metrics_cmd = [
                sys.executable,
                str(REPO_ROOT / "metrics.py"),
                "-m",
                str(args.model_path),
            ]
            subprocess.run(render_cmd, cwd=REPO_ROOT, check=True)
            subprocess.run(metrics_cmd, cwd=REPO_ROOT, check=True)
        else:
            decompress_point_cloud(args.input, args.output)


if __name__ == "__main__":
    main()
