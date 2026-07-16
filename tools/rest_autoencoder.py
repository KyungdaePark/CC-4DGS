#!/usr/bin/env python3
"""
Per-scene autoencoder for compressing f_rest* attributes.
Provides train / encode / decode commands.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from functools import partial
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import shutil
from tqdm.auto import tqdm

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional dependency
    zstd = None

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compress_point_cloud import _load_ply, _write_ply
from tools.entropy_pack_point_cloud import (
    compress_point_cloud as entropy_compress,
    decompress_point_cloud as entropy_decompress,
)
from tools import restfree_quant_pack

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
ANSI_BLUE = "\033[34m"
ANSI_RESET = "\033[0m"


def _require_zstd() -> None:
    if zstd is None:
        raise RuntimeError("zstandard module not available. Please install 'zstandard' to use npz_zstd compression.")


def _write_latents_pack(path: Path, payload: Dict[str, object], compression: str) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    data = buffer.getvalue()
    if compression == "npz":
        path.write_bytes(data)
    elif compression == "npz_zstd":
        _require_zstd()
        compressor = zstd.ZstdCompressor(level=10)
        path.write_bytes(compressor.compress(data))
    else:
        raise ValueError(f"Unsupported latents compression '{compression}'")


def _load_latents_pack(path: Path):
    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic == ZSTD_MAGIC:
        _require_zstd()
        compressed = path.read_bytes()
        decompressed = zstd.ZstdDecompressor().decompress(compressed)
        return np.load(io.BytesIO(decompressed), allow_pickle=False)
    return np.load(path, allow_pickle=False)


def _format_mb_colored(value: float) -> str:
    return f"{ANSI_BLUE}{value:.2f} MB{ANSI_RESET}"


def _select_uint_dtype(bits: int):
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    raise ValueError("quant_bits must be <=32")


def _initialise_codebook(data: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(data.shape[0], generator=generator, device=data.device)
    return data[perm[:k]].clone()


def _assign_points(data: torch.Tensor, centers: torch.Tensor, chunk_size: int) -> torch.Tensor:
    assignments = []
    n = data.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch = data[start:end]
        dists = torch.cdist(batch, centers)
        assignments.append(torch.argmin(dists, dim=1))
    return torch.cat(assignments)


def _update_centers(
    data: torch.Tensor,
    assignments: torch.Tensor,
    centers: torch.Tensor,
    chunk_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    k = centers.shape[0]
    new_centers = torch.zeros_like(centers)
    counts = torch.zeros(k, device=data.device, dtype=torch.float32)
    n = data.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch = data[start:end]
        idx = assignments[start:end]
        new_centers.index_add_(0, idx, batch)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    nonzero = counts > 0
    if nonzero.any():
        new_centers[nonzero] /= counts[nonzero].unsqueeze(1)

    zero_mask = ~nonzero
    if zero_mask.any():
        replace = data[torch.randint(0, n, (int(zero_mask.sum().item()),), device=data.device, generator=generator)]
        new_centers[zero_mask] = replace

    return new_centers


def _kmeans_codebook(
    data: np.ndarray,
    *,
    k: int,
    iters: int,
    chunk_size: int,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    tensor = torch.from_numpy(data).to(device)
    centers = _initialise_codebook(tensor, k, generator)
    assignments = torch.zeros(tensor.shape[0], dtype=torch.long, device=device)

    for _ in range(iters):
        assignments = _assign_points(tensor, centers, chunk_size)
        centers = _update_centers(tensor, assignments, centers, chunk_size, generator)

    assignments = _assign_points(tensor, centers, chunk_size)
    return centers.cpu().numpy().astype(np.float32), assignments.cpu().numpy()


def _resolve_ply(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "point_cloud.ply"
    if not path.exists():
        raise FileNotFoundError(f"PLY not found: {path}")
    return path


def _iteration_dir(model_path: Path, iteration: int) -> Path:
    return model_path / "point_cloud" / f"iteration_{iteration}"


def _strip_rest_fields(source_ply: Path, dest_ply: Path, rest_names: List[str]) -> None:
    arrays, names = _load_ply(source_ply)
    filtered = [name for name in names if name not in rest_names]
    data = {name: arrays[name] for name in filtered}
    _write_ply(dest_ply, data, filtered)
    print(f"[autoencoder-auto] Wrote rest-free PLY to {dest_ply}")


def _human_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _stack_fields(arrays: Dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    if not names:
        return np.empty((len(next(iter(arrays.values()))), 0), dtype=np.float32)
    return np.stack([arrays[name] for name in names], axis=1).astype(np.float32)


def _decode_latent_arrays(
    latents_path: Path,
    checkpoint: Path,
    *,
    chunk: int,
    device: str,
) -> Tuple[np.ndarray, List[str]]:
    with _load_latents_pack(latents_path) as latents_pack:
        meta = json.loads(str(latents_pack["meta"]))
        if meta.get("quant_bits"):
            shape = tuple(meta.get("latents_shape") or (meta["count"], meta["latent_dim"]))
            if meta.get("bitpack_format") == "bitplane_little":
                latents_q = _bitunpack_latents(
                    latents_pack["latents_bitpack"],
                    shape,
                    meta["quant_bits"],
                )
            else:
                latents_q = latents_pack["latents_q"]
            latents = _dequantise_latents(
                latents_q,
                latents_pack["latents_min"],
                latents_pack["latents_max"],
                meta["quant_bits"],
            )
        else:
            latents = latents_pack["latents"]
    model, target_mean, target_std, _, _, target_names, _ = _load_checkpoint(checkpoint, device)
    target_mean_t = torch.from_numpy(target_mean).to(device)
    target_std_t = torch.from_numpy(target_std).to(device)
    reconstructed = np.empty((latents.shape[0], len(target_names)), dtype=np.float32)
    chunk_size = latents.shape[0] if chunk <= 0 else chunk
    with torch.no_grad():
        for start in range(0, latents.shape[0], chunk_size):
            end = start + chunk_size
            batch = torch.from_numpy(latents[start:end].astype(np.float32)).to(device)
            rec = model.decode_batch(batch) * target_std_t + target_mean_t
            reconstructed[start:end] = rec.cpu().numpy()
    return reconstructed, target_names


def _expand_field_specs(arrays: Dict[str, np.ndarray], specs: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for spec in specs:
        if spec in arrays:
            expanded.append(spec)
            continue
        matches = [name for name in arrays if name.startswith(f"{spec}_")]
        if not matches:
            raise KeyError(f"Field spec '{spec}' not found in PLY attributes.")
        expanded.extend(sorted(matches, key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x))
    return expanded


class LatentDataset(Dataset):
    def __init__(
        self,
        target: np.ndarray,
        cond: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        cond_mean: np.ndarray,
        cond_std: np.ndarray,
    ) -> None:
        self.target = torch.from_numpy((target - target_mean) / target_std)
        if cond.size:
            self.cond = torch.from_numpy((cond - cond_mean) / cond_std)
        else:
            self.cond = torch.empty((target.shape[0], 0), dtype=torch.float32)

    def __len__(self) -> int:
        return self.target.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.cond.numel():
            return torch.cat([self.target[idx], self.cond[idx]], dim=0), self.target[idx]
        return self.target[idx], self.target[idx]


class RestAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0, output_dim: int | None = None):
        super().__init__()
        self.encoder = self._build_mlp(input_dim, latent_dim, hidden_dim, num_layers, dropout, final_activation=False)
        out_dim = input_dim if output_dim is None else output_dim
        self.decoder = self._build_mlp(latent_dim, out_dim, hidden_dim, num_layers, dropout, final_activation=False)

    @staticmethod
    def _build_mlp(inp: int, out: int, hidden: int, layers: int, dropout: float, final_activation: bool) -> nn.Sequential:
        blocks: List[nn.Module] = []
        last = inp
        for _ in range(layers):
            blocks.append(nn.Linear(last, hidden))
            blocks.append(nn.GELU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            last = hidden
        blocks.append(nn.Linear(last, out))
        if final_activation:
            blocks.append(nn.GELU())
        return nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(inputs)
        recon = self.decoder(z)
        return recon

    def encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode_batch(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def _quantise_latents(values: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    spans = np.where(maxs > mins, maxs - mins, 1.0)
    levels = (1 << bits) - 1
    scaled = np.round((values - mins) / spans * levels)
    dtype = _select_uint_dtype(bits)
    q = scaled.clip(0, levels).astype(dtype)
    return q, mins.astype(np.float32), maxs.astype(np.float32)


def _dequantise_latents(q: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    spans = np.where(maxs > mins, maxs - mins, 1.0)
    return mins + (q.astype(np.float32) / levels) * spans


def _bitpack_latents(q: np.ndarray, bits: int) -> np.ndarray:
    flat = q.reshape(-1).astype(np.uint32)
    count = flat.size
    bytes_per_plane = (count + 7) // 8
    packed = np.empty((bits, bytes_per_plane), dtype=np.uint8)
    for b in range(bits):
        plane = np.packbits(((flat >> b) & 1).astype(np.uint8), bitorder="little")
        if plane.size < bytes_per_plane:
            plane = np.pad(plane, (0, bytes_per_plane - plane.size), constant_values=0)
        packed[b] = plane
    return packed.reshape(-1)


def _bitunpack_latents(buffer: np.ndarray, shape: Sequence[int], bits: int) -> np.ndarray:
    count = int(np.prod(shape))
    bytes_per_plane = (count + 7) // 8
    expected = bytes_per_plane * bits
    if buffer.size != expected:
        raise ValueError(
            f"Bitpacked latents size mismatch: expected {expected} bytes, got {buffer.size}"
        )
    flat = np.zeros(count, dtype=np.uint32)
    for b in range(bits):
        start = b * bytes_per_plane
        plane = buffer[start : start + bytes_per_plane]
        bits_arr = np.unpackbits(plane, bitorder="little")[:count].astype(np.uint32)
        flat |= bits_arr << b
    dtype = _select_uint_dtype(bits)
    return flat.astype(dtype).reshape(shape)


def train_autoencoder(
    input_path: Path,
    output_dir: Path,
    *,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    target_fields: List[str],
    condition_fields: List[str],
    batch_size: int,
    steps: int,
    lr: float,
    device: str,
    log_every: int,
    seed: int,
    checkpoint_dtype: str,
) -> None:
    arrays, names = _load_ply(input_path)
    target_names = _expand_field_specs(arrays, target_fields)
    if not target_names:
        raise RuntimeError("No target fields selected for autoencoder.")
    rest = _stack_fields(arrays, target_names)
    for field in condition_fields:
        if field not in arrays:
            raise KeyError(f"Condition field '{field}' not found in PLY.")
    cond = _stack_fields(arrays, condition_fields)

    torch.manual_seed(seed)
    np.random.seed(seed)

    target_mean = rest.mean(axis=0, keepdims=True)
    target_std = rest.std(axis=0, keepdims=True)
    target_std = np.where(target_std < 1e-6, 1.0, target_std)

    if cond.size:
        cond_mean = cond.mean(axis=0, keepdims=True)
        cond_std = cond.std(axis=0, keepdims=True)
        cond_std = np.where(cond_std < 1e-6, 1.0, cond_std)
    else:
        cond_mean = np.zeros((1, 0), dtype=np.float32)
        cond_std = np.ones((1, 0), dtype=np.float32)

    dataset = LatentDataset(rest, cond, target_mean, target_std, cond_mean, cond_std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    iterator = cycle(loader)

    input_dim = rest.shape[1] + cond.shape[1]
    model = RestAutoencoder(input_dim, latent_dim, hidden_dim, num_layers, dropout, output_dim=rest.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    progress = tqdm(range(1, steps + 1), desc="[autoencoder] training", leave=True)
    for step in progress:
        inputs, targets = next(iterator)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        recon = model(inputs)
        loss = criterion(recon, targets)
        loss.backward()
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.6f}")
        # Progress bar already shows loss; suppress per-step prints for cleanliness.
    progress.close()

    raw_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if checkpoint_dtype == "float16":
        save_state = {k: v.half() for k, v in raw_state.items()}
    elif checkpoint_dtype == "float32":
        save_state = raw_state
    else:
        raise ValueError(f"Unsupported checkpoint dtype '{checkpoint_dtype}'")

    def _norm_array(arr: np.ndarray) -> np.ndarray:
        if checkpoint_dtype == "float16":
            return arr.astype(np.float16)
        return arr.astype(np.float32)

    ckpt = {
        "state_dict": save_state,
        "config": {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        },
        "normalizer": {
            "target_mean": _norm_array(target_mean),
            "target_std": _norm_array(target_std),
            "cond_mean": _norm_array(cond_mean),
            "cond_std": _norm_array(cond_std),
        },
        "target_names": target_names,
        "condition_fields": condition_fields,
        "steps": steps,
        "checkpoint_dtype": checkpoint_dtype,
    }
    ckpt_path = output_dir / "rest_autoencoder.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[autoencoder] Saved checkpoint to {ckpt_path}")


def _torch_load_trusted(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_checkpoint(path: Path, device: str):
    ckpt = _torch_load_trusted(path, map_location=device)
    cfg = ckpt["config"]
    ckpt_dtype = ckpt.get("checkpoint_dtype", "float32")
    state_dict = ckpt["state_dict"]
    if ckpt_dtype == "float16":
        state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
    model = RestAutoencoder(
        cfg["input_dim"],
        cfg["latent_dim"],
        cfg["hidden_dim"],
        cfg["num_layers"],
        cfg.get("dropout", 0.0),
        output_dim=ckpt["normalizer"]["target_mean"].shape[1],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    norm = ckpt["normalizer"]
    target_mean = norm["target_mean"].astype(np.float32)
    target_std = norm["target_std"].astype(np.float32)
    cond_mean = norm["cond_mean"].astype(np.float32)
    cond_std = norm["cond_std"].astype(np.float32)
    target_names = ckpt["target_names"]
    cond_fields = ckpt.get("condition_fields", [])
    return model, target_mean, target_std, cond_mean, cond_std, target_names, cond_fields


def encode_rest(
    input_path: Path,
    checkpoint: Path,
    output_path: Path,
    *,
    dtype: str,
    chunk: int,
    device: str,
    quant_bits: int | None,
    latents_compression: str = "npz",
    bitpack_latents: bool = False,
) -> None:
    input_path = _resolve_ply(input_path)
    arrays, names = _load_ply(input_path)
    model, target_mean, target_std, cond_mean, cond_std, target_names, cond_fields = _load_checkpoint(checkpoint, device)
    rest = _stack_fields(arrays, target_names)
    cond = _stack_fields(arrays, cond_fields)

    dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": np.float32,  # PyTorch lacks BF16 numpy
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype}")
    target_dtype = dtype_map[dtype]

    rest_mean_t = torch.from_numpy(target_mean).to(device)
    rest_std_t = torch.from_numpy(target_std).to(device)
    cond_mean_t = torch.from_numpy(cond_mean).to(device) if cond.size else None
    cond_std_t = torch.from_numpy(cond_std).to(device) if cond.size else None

    latents_list: List[np.ndarray] = []
    chunk_size = rest.shape[0] if chunk <= 0 else chunk
    with torch.no_grad():
        for start in range(0, rest.shape[0], chunk_size):
            end = start + chunk_size
            rest_batch = torch.from_numpy(rest[start:end]).to(device)
            rest_norm = (rest_batch - rest_mean_t) / rest_std_t
            if cond.size:
                cond_batch = torch.from_numpy(cond[start:end]).to(device)
                cond_norm = (cond_batch - cond_mean_t) / cond_std_t
                inputs = torch.cat([rest_norm, cond_norm], dim=1)
            else:
                inputs = rest_norm
            z = model.encode_batch(inputs).cpu().numpy()
            latents_list.append(z.astype(target_dtype))
    latents = np.concatenate(latents_list, axis=0)

    meta = {
        "checkpoint": str(checkpoint),
        "input_ply": str(input_path),
        "latent_dim": latents.shape[1],
        "dtype": dtype,
        "count": latents.shape[0],
        "target_names": target_names,
        "condition_fields": cond_fields,
        "latents_shape": latents.shape,
    }

    if quant_bits is not None:
        q, lat_min, lat_max = _quantise_latents(latents.astype(np.float32), quant_bits)
        meta["quant_bits"] = quant_bits
        if bitpack_latents:
            packed = _bitpack_latents(q, quant_bits)
            meta["bitpack_format"] = "bitplane_little"
            payload = {
                "latents_bitpack": packed,
                "latents_min": lat_min,
                "latents_max": lat_max,
                "meta": np.array(json.dumps(meta)),
            }
        else:
            payload = {
                "latents_q": q,
                "latents_min": lat_min,
                "latents_max": lat_max,
                "meta": np.array(json.dumps(meta)),
            }
    else:
        meta["quant_bits"] = None
        payload = {
            "latents": latents,
            "meta": np.array(json.dumps(meta)),
        }
    _write_latents_pack(output_path, payload, latents_compression)
    print(f"[autoencoder] Encoded latents saved to {output_path}")


def decode_rest(
    latents_path: Path,
    checkpoint: Path,
    source_ply: Path,
    output_ply: Path,
    *,
    chunk: int,
    device: str,
) -> None:
    with _load_latents_pack(latents_path) as latents_pack:
        meta = json.loads(str(latents_pack["meta"]))
        if meta.get("quant_bits"):
            shape = tuple(meta.get("latents_shape") or (meta["count"], meta["latent_dim"]))
            if meta.get("bitpack_format") == "bitplane_little":
                latents_q = _bitunpack_latents(
                    latents_pack["latents_bitpack"],
                    shape,
                    meta["quant_bits"],
                )
            else:
                latents_q = latents_pack["latents_q"]
            latents = _dequantise_latents(
                latents_q,
                latents_pack["latents_min"],
                latents_pack["latents_max"],
                meta["quant_bits"],
            )
        else:
            latents = latents_pack["latents"]
    model, target_mean, target_std, _, _, target_names, _ = _load_checkpoint(checkpoint, device)

    source_ply = _resolve_ply(source_ply)
    arrays, names = _load_ply(source_ply)
    count = latents.shape[0]
    names_out = list(names)
    for name in target_names:
        if name not in arrays:
            arrays[name] = np.zeros(count, dtype=np.float32)
            names_out.append(name)
    if latents.shape[0] != arrays[target_names[0]].shape[0]:
        raise ValueError("Latent count mismatch with source PLY.")

    target_mean_t = torch.from_numpy(target_mean).to(device)
    target_std_t = torch.from_numpy(target_std).to(device)

    reconstructed = np.empty((latents.shape[0], len(target_names)), dtype=np.float32)
    chunk_size = latents.shape[0] if chunk <= 0 else chunk
    with torch.no_grad():
        for start in range(0, latents.shape[0], chunk_size):
            end = start + chunk_size
            batch = torch.from_numpy(latents[start:end].astype(np.float32)).to(device)
            rec = model.decode_batch(batch) * target_std_t + target_mean_t
            reconstructed[start:end] = rec.cpu().numpy()

    for idx, name in enumerate(target_names):
        arrays[name] = reconstructed[:, idx]

    _write_ply(output_ply, arrays, names_out)
    print(f"[autoencoder] Reconstructed PLY saved to {output_ply}")


def _fit_residual_codebook(
    arrays: Dict[str, np.ndarray],
    latents_path: Path,
    checkpoint: Path,
    *,
    chunk: int,
    device: str,
    condition_fields: List[str],
    condition_weight: float,
    codebook_size: int,
    iters: int,
    seed: int,
    chunk_size: int,
    output_dir: Path,
) -> Tuple[Dict[str, Path], List[str]]:
    rest_hat, rest_names = _decode_latent_arrays(latents_path, checkpoint, chunk=chunk, device=device)
    rest_original = _stack_fields(arrays, rest_names)
    residual = rest_original - rest_hat
    residual_mean = residual.mean(axis=0, keepdims=True)
    residual_std = residual.std(axis=0, keepdims=True)
    residual_std = np.where(residual_std < 1e-6, 1.0, residual_std)
    residual_norm = (residual - residual_mean) / residual_std
    cond = _stack_fields(arrays, condition_fields)
    if cond.size:
        cond_mean = cond.mean(axis=0, keepdims=True)
        cond_std = cond.std(axis=0, keepdims=True)
        cond_std = np.where(cond_std < 1e-6, 1.0, cond_std)
        cond_norm = (cond - cond_mean) / cond_std
        fused = np.concatenate([residual_norm, condition_weight * cond_norm], axis=1)
    else:
        fused = residual_norm
    codebook_norm, assignments = _kmeans_codebook(
        fused,
        k=codebook_size,
        iters=iters,
        chunk_size=fused.shape[0] if chunk_size <= 0 else chunk_size,
        device=torch.device(device),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    codebook = np.zeros((codebook_size, residual.shape[1]), dtype=np.float32)
    for idx in range(codebook_size):
        mask = assignments == idx
        if mask.any():
            codebook[idx] = residual[mask].mean(axis=0)
        else:
            codebook[idx] = residual[rng.integers(0, residual.shape[0])]
    dtype = _select_uint_dtype(int(np.ceil(np.log2(max(codebook_size, 1)))) if codebook_size > 0 else 1)
    assign_quant = assignments.astype(dtype)
    codebook_path = output_dir / "residual_codebook.npy"
    assignments_path = output_dir / "residual_assignments.npy"
    np.save(codebook_path, codebook)
    np.save(assignments_path, assign_quant)
    return {"codebook": codebook_path, "assignments": assignments_path}, rest_names


def _apply_residual_to_ply(ply_path: Path, rest_names: List[str], codebook_path: Path, assignments_path: Path) -> None:
    if not codebook_path.exists() or not assignments_path.exists():
        return
    codebook = np.load(codebook_path)
    assignments = np.load(assignments_path)
    arrays, names = _load_ply(ply_path)
    residual = codebook[assignments]
    for idx, name in enumerate(rest_names):
        arrays[name] = arrays[name] + residual[:, idx]
    _write_ply(ply_path, arrays, names)


def _write_compressed_bundle(
    iteration_dir: Path,
    *,
    target_free_ply: Path,
    latents_path: Path,
    checkpoint_path: Path,
    quantize_groups: str,
    xyz_bits: int,
    scale_bits: int,
    opacity_bits: int,
    rotation_bits: int,
    dynamic_bits: int,
    dynamic_mu: float,
    residual_info: Dict[str, Path] | None = None,
) -> None:
    compressed_dir = iteration_dir / "compressed"
    if compressed_dir.exists():
        shutil.rmtree(compressed_dir)
    compressed_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = compressed_dir / "restfree"
    restfree_quant_pack.pack_restfree(
        input_ply=target_free_ply,
        output_dir=pack_dir,
        xyz_bits=xyz_bits,
        scale_bits=scale_bits,
        opacity_bits=opacity_bits,
        rotation_bits=rotation_bits,
        dynamic_bits=dynamic_bits,
        dynamic_mu=dynamic_mu,
        quantize_groups=quantize_groups,
    )
    shutil.copy2(latents_path, compressed_dir / "latents.npz")
    shutil.copy2(checkpoint_path, compressed_dir / "rest_autoencoder.pt")
    if residual_info:
        shutil.copy2(residual_info["codebook"], compressed_dir / "residual_codebook.npy")
        shutil.copy2(residual_info["assignments"], compressed_dir / "residual_assignments.npy")
    manifest = {
        "target_free_source": str(target_free_ply),
        "quantize_groups": quantize_groups,
        "xyz_bits": xyz_bits,
        "scale_bits": scale_bits,
        "opacity_bits": opacity_bits,
        "rotation_bits": rotation_bits,
        "dynamic_bits": dynamic_bits,
        "dynamic_mu": dynamic_mu,
    }
    (compressed_dir / "restfree_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[autoencoder-auto] Wrote compressed assets to {compressed_dir}")


def run_auto_pipeline(args) -> None:
    model_path = args.model_path
    src_dir = _iteration_dir(model_path, args.source_iteration)
    src_ply = _resolve_ply(src_dir)
    checkpoint_dir = model_path / "autoencoder" / f"iteration_{args.iteration}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[autoencoder-auto] Training on {src_ply} ...")
    train_autoencoder(
        src_ply,
        checkpoint_dir,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        target_fields=args.target_fields,
        condition_fields=args.condition_fields,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        log_every=args.log_every,
        seed=args.seed,
        checkpoint_dtype=args.checkpoint_dtype,
    )
    checkpoint = checkpoint_dir / "rest_autoencoder.pt"
    latents_path = checkpoint_dir / "latents.npz"
    print("[autoencoder-auto] Encoding latents ...")
    encode_rest(
        src_ply,
        checkpoint,
        latents_path,
        dtype=args.dtype,
        chunk=args.chunk,
        device=args.device,
        quant_bits=args.quant_bits,
        latents_compression=args.latents_compression,
        bitpack_latents=args.bitpack_latents,
    )
    residual_info = None
    residual_rest_names: List[str] = []
    if args.residual_codebook_size > 0:
        arrays_src, _ = _load_ply(src_ply)
        residual_info, residual_rest_names = _fit_residual_codebook(
            arrays_src,
            latents_path=latents_path,
            checkpoint=checkpoint,
            chunk=args.chunk,
            device=args.device,
            condition_fields=args.condition_fields,
            condition_weight=args.residual_condition_weight,
            codebook_size=args.residual_codebook_size,
            iters=args.residual_iters,
            seed=args.seed + 123,
            chunk_size=args.chunk,
            output_dir=checkpoint_dir,
        )
    model_state_names = ("deformation.pth", "FDHash.pth")
    target_dir = _iteration_dir(model_path, args.iteration)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_ply = target_dir / "point_cloud.ply"
    decode_source = target_ply
    if target_ply.exists() and not args.overwrite:
        print(f"[autoencoder-auto] Target PLY exists at {target_ply}; reusing source without overwrite.")
        decode_source = src_ply
    else:
        print("[autoencoder-auto] Decoding into target iteration ...")
        decode_rest(
            latents_path,
            checkpoint,
            src_ply,
            target_ply,
            chunk=args.chunk,
            device=args.device,
        )
        if residual_info:
            _apply_residual_to_ply(target_ply, residual_rest_names, residual_info["codebook"], residual_info["assignments"])
        copied_model_state = False
        for state_name in model_state_names:
            src_state = src_dir / state_name
            dst_state = target_dir / state_name
            if src_state.exists():
                shutil.copy2(src_state, dst_state)
                print(f"[autoencoder-auto] Copied model checkpoint to {dst_state}")
                copied_model_state = True
        if not copied_model_state:
            expected = ", ".join(str(src_dir / name) for name in model_state_names)
            print(f"[autoencoder-auto] Warning: no model checkpoint found; expected one of: {expected}")
    ckpt_meta = _torch_load_trusted(checkpoint, map_location="cpu")
    target_names = ckpt_meta["target_names"]
    target_free_path = checkpoint_dir / "point_cloud_without_targets.ply"
    _strip_rest_fields(decode_source, target_free_path, target_names)
    if args.write_compressed:
        _write_compressed_bundle(
            target_dir,
            target_free_ply=target_free_path,
            latents_path=latents_path,
            checkpoint_path=checkpoint,
            quantize_groups=args.restfree_groups,
            xyz_bits=args.restfree_xyz_bits,
            scale_bits=args.restfree_scale_bits,
            opacity_bits=args.restfree_opacity_bits,
            rotation_bits=args.restfree_rotation_bits,
            dynamic_bits=args.restfree_dynamic_bits,
            dynamic_mu=args.restfree_dynamic_mu,
            residual_info=residual_info,
        )

    share_artifact_path = target_free_path
    share_label = "target-free PLY"
    entropy_npz = None
    if args.entropy_share:
        entropy_npz = checkpoint_dir / "target_free_entropy.npz"
        entropy_compress(
            target_free_path,
            entropy_npz,
            pos_bits=args.entropy_pos_bits,
            feature_bits=args.entropy_feature_bits,
            rest_bits=args.entropy_rest_bits,
        )
        share_artifact_path = entropy_npz
        share_label = "target-free entropy NPZ"

    original_ply_size = _human_mb(src_ply)
    share_latents = _human_mb(latents_path)
    share_ckpt = _human_mb(checkpoint)
    share_ply = _human_mb(share_artifact_path)
    residual_codebook_mb = _human_mb(residual_info["codebook"]) if residual_info else 0.0
    residual_assign_mb = _human_mb(residual_info["assignments"]) if residual_info else 0.0
    total_share = share_latents + share_ckpt + share_ply + residual_codebook_mb + residual_assign_mb
    reduction = 100.0 * (1 - total_share / original_ply_size)
    print("[autoencoder-auto] Files to share:")
    print(f"  - {share_label}: {share_artifact_path} ({_format_mb_colored(share_ply)})")
    print(f"  - latents: {latents_path} ({_format_mb_colored(share_latents)})")
    print(f"  - checkpoint: {checkpoint} ({_format_mb_colored(share_ckpt)})")
    if residual_info:
        print(f"  - residual codebook: {residual_info['codebook']} ({_format_mb_colored(residual_codebook_mb)})")
        print(f"  - residual assignments: {residual_info['assignments']} ({_format_mb_colored(residual_assign_mb)})")
    print(
        f"  => Combined share size {_format_mb_colored(total_share)} vs original {_format_mb_colored(original_ply_size)} ({reduction:.1f}% smaller)"
    )
    deploy_iter = args.deploy_iteration or args.iteration
    deploy_dir = _iteration_dir(model_path, deploy_iter)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    deploy_ply = deploy_dir / "point_cloud.ply"
    deploy_source = target_free_path
    temp_entropy_ply = None
    if args.entropy_share:
        temp_entropy_ply = checkpoint_dir / "target_free_entropy_decompressed.ply"
        entropy_decompress(entropy_npz, temp_entropy_ply)
        deploy_source = temp_entropy_ply
    if deploy_ply.exists() and not args.overwrite:
        print(f"[autoencoder-auto] Deploy PLY exists at {deploy_ply}; skipping decode (use --overwrite to refresh).")
    else:
        decode_rest(
            latents_path,
            checkpoint,
            deploy_source,
            deploy_ply,
            chunk=args.chunk,
            device=args.device,
        )
        if residual_info:
            _apply_residual_to_ply(deploy_ply, residual_rest_names, residual_info["codebook"], residual_info["assignments"])
        copied_model_state = False
        for state_name in model_state_names:
            src_state = src_dir / state_name
            dst_state = deploy_dir / state_name
            if src_state.exists():
                shutil.copy2(src_state, dst_state)
                print(f"[autoencoder-auto] Copied model checkpoint to {dst_state}")
                copied_model_state = True
        if not copied_model_state:
            expected = ", ".join(str(src_dir / name) for name in model_state_names)
            print(f"[autoencoder-auto] Warning: no model checkpoint found; expected one of: {expected}")
    if temp_entropy_ply is not None:
        temp_entropy_ply.unlink(missing_ok=True)
    if args.run_render:
        required = [args.dataset_path, args.config_path]
        if any(v is None for v in required):
            raise ValueError("--run-render requires --dataset-path and --config-path")
        render_cmd = [
            sys.executable,
            str(REPO_ROOT / "render.py"),
            "-s",
            str(args.dataset_path),
            "--configs",
            str(args.config_path),
            "--model_path",
            str(model_path),
            "--iteration",
            str(deploy_iter),
            "--skip_train",
            "--skip_video",
        ]
        print("[autoencoder-auto] Running render.py ...")
        subprocess.run(render_cmd, check=True)
        if not args.skip_metrics:
            metrics_cmd = [
                sys.executable,
                str(REPO_ROOT / "metrics.py"),
                "-m",
                str(model_path),
                "--iteration",
                str(deploy_iter),
            ]
            print("[autoencoder-auto] Running metrics.py ...")
            subprocess.run(metrics_cmd, check=True)
    print(f"[autoencoder-auto] Finished. Iteration {args.iteration} ready at {target_dir}; deployed to iteration {deploy_iter}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autoencoder-based rest compression.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train autoencoder on f_rest*.")
    train.add_argument("input", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--latent-dim", type=int, default=12)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--num-layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument(
        "--condition-fields",
        type=str,
        nargs="*",
        default=[],
        help="Additional PLY attributes to concatenate as conditioning inputs.",
    )
    train.add_argument(
        "--target-fields",
        type=str,
        nargs="*",
        default=["f_rest"],
        help="Attribute groups to encode (prefix like f_rest, f_dc, or exact field names).",
    )
    train.add_argument("--batch-size", type=int, default=8192)
    train.add_argument("--steps", type=int, default=20000)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--device", type=str, default="cuda")
    train.add_argument("--log-every", type=int, default=100)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--checkpoint-dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="Precision used when serialising the autoencoder checkpoint.",
    )

    encode = sub.add_parser("encode", help="Encode rest vectors into latents.")
    encode.add_argument("input", type=Path)
    encode.add_argument("--checkpoint", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16"])
    encode.add_argument("--chunk", type=int, default=65536)
    encode.add_argument("--device", type=str, default="cuda")
    encode.add_argument("--quant-bits", type=int, default=None)
    encode.add_argument(
        "--latents-compression",
        type=str,
        choices=["npz", "npz_zstd"],
        default="npz",
        help="Container format for latents.npz (npz_zstd wraps the NPZ in Zstandard).",
    )
    encode.add_argument(
        "--bitpack-latents",
        action="store_true",
        help="Store quantised latents in a bit-packed representation.",
    )

    decode = sub.add_parser("decode", help="Decode latents and rebuild PLY.")
    decode.add_argument("latents", type=Path)
    decode.add_argument("--checkpoint", type=Path, required=True)
    decode.add_argument("--source", type=Path, required=True, help="Original PLY or iteration directory for non-rest attributes.")
    decode.add_argument("--output", type=Path, required=True)
    decode.add_argument("--chunk", type=int, default=65536)
    decode.add_argument("--device", type=str, default="cuda")
    decode.add_argument("--residual-dir", type=Path, default=None, help="Directory containing residual_codebook.npy and residual_assignments.npy.")

    auto = sub.add_parser("auto", help="Train + encode + decode directly into a target iteration.")
    auto.add_argument("--model-path", type=Path, required=True)
    auto.add_argument("--source-iteration", type=int, required=True)
    auto.add_argument("--iteration", type=int, required=True, help="Target iteration to write decoded PLY.")
    auto.add_argument("--latent-dim", type=int, default=12)
    auto.add_argument("--hidden-dim", type=int, default=256)
    auto.add_argument("--num-layers", type=int, default=2)
    auto.add_argument("--dropout", type=float, default=0.0)
    auto.add_argument("--condition-fields", type=str, nargs="*", default=[])
    auto.add_argument("--target-fields", type=str, nargs="*", default=["f_rest"])
    auto.add_argument("--batch-size", type=int, default=8192)
    auto.add_argument("--steps", type=int, default=20000)
    auto.add_argument("--lr", type=float, default=1e-3)
    auto.add_argument("--device", type=str, default="cuda")
    auto.add_argument("--log-every", type=int, default=100)
    auto.add_argument("--seed", type=int, default=42)
    auto.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16"])
    auto.add_argument("--chunk", type=int, default=65536, help="Chunk size for encode/decode.")
    auto.add_argument("--quant-bits", type=int, default=None)
    auto.add_argument(
        "--checkpoint-dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="Precision used to store rest_autoencoder.pt.",
    )
    auto.add_argument(
        "--latents-compression",
        type=str,
        choices=["npz", "npz_zstd"],
        default="npz",
        help="Container format for latents artifacts.",
    )
    auto.add_argument(
        "--bitpack-latents",
        action="store_true",
        help="Store quantised latents in a bit-packed representation.",
    )
    auto.add_argument("--overwrite", action="store_true", help="Overwrite existing target iteration PLY.")
    auto.add_argument("--residual-codebook-size", type=int, default=256)
    auto.add_argument("--residual-iters", type=int, default=20)
    auto.add_argument("--residual-condition-weight", type=float, default=0.2)
    auto.add_argument("--deploy-iteration", type=int, default=None, help="Iteration to publish final PLY (default: --iteration).")
    auto.add_argument("--entropy-share", action="store_true", help="Compress target-free PLY with entropy coding for sharing.")
    auto.add_argument("--entropy-pos-bits", type=int, default=18)
    auto.add_argument("--entropy-feature-bits", type=int, default=16)
    auto.add_argument("--entropy-rest-bits", type=int, default=16)
    auto.add_argument("--run-render", action="store_true", help="After deploying, run render.py and metrics.py.")
    auto.add_argument("--dataset-path", type=Path)
    auto.add_argument("--config-path", type=Path)
    auto.add_argument("--skip-metrics", action="store_true", help="Skip metrics.py invocation when --run-render is set.")
    auto.add_argument("--write-compressed", action="store_true", help="Emit compressed/ bundle (latents + restfree pack) for --compressed rendering.")
    auto.add_argument("--restfree-groups", type=str, default="scale,opacity,rotation,dynamic", help="Comma-separated attribute groups to restfree-quantise for compressed output.")
    auto.add_argument("--restfree-xyz-bits", type=int, default=14)
    auto.add_argument("--restfree-scale-bits", type=int, default=12)
    auto.add_argument("--restfree-opacity-bits", type=int, default=12)
    auto.add_argument("--restfree-rotation-bits", type=int, default=12)
    auto.add_argument("--restfree-dynamic-bits", type=int, default=16)
    auto.add_argument("--restfree-dynamic-mu", type=float, default=255.0, help="μ-law parameter for dynamic logits during compressed packing.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        input_ply = _resolve_ply(args.input)
        train_autoencoder(
            input_ply,
            args.output_dir,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            target_fields=args.target_fields,
            condition_fields=args.condition_fields,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            log_every=args.log_every,
            seed=args.seed,
            checkpoint_dtype=args.checkpoint_dtype,
        )
    elif args.command == "encode":
        encode_rest(
            args.input,
            args.checkpoint,
            args.output,
            dtype=args.dtype,
            chunk=args.chunk,
            device=args.device,
            quant_bits=args.quant_bits,
            latents_compression=args.latents_compression,
            bitpack_latents=args.bitpack_latents,
        )
    elif args.command == "decode":
        decode_rest(
            args.latents,
            args.checkpoint,
            args.source,
            args.output,
            chunk=args.chunk,
            device=args.device,
        )
        if args.residual_dir:
            codebook_path = args.residual_dir / "residual_codebook.npy"
            assignments_path = args.residual_dir / "residual_assignments.npy"
            if codebook_path.exists() and assignments_path.exists():
                _, _, _, _, _, target_names, _ = _load_checkpoint(args.checkpoint, args.device)
                _apply_residual_to_ply(args.output, target_names, codebook_path, assignments_path)
    elif args.command == "auto":
        run_auto_pipeline(args)
    else:
        parser.error(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
