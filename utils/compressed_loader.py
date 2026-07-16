"""Utilities for loading Gaussian models from compressed artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def _load_latent_matrix(latents_path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    from tools import rest_autoencoder as rest

    with rest._load_latents_pack(latents_path) as latents_pack:
        meta = json.loads(str(latents_pack["meta"]))
        if meta.get("quant_bits"):
            shape = tuple(meta.get("latents_shape") or (meta["count"], meta["latent_dim"]))
            if meta.get("bitpack_format") == "bitplane_little":
                latents_q = rest._bitunpack_latents(
                    latents_pack["latents_bitpack"],
                    shape,
                    meta["quant_bits"],
                )
            else:
                latents_q = latents_pack["latents_q"]
            latents = rest._dequantise_latents(
                latents_q,
                latents_pack["latents_min"],
                latents_pack["latents_max"],
                meta["quant_bits"],
            )
        else:
            latents = latents_pack["latents"]
    return latents.astype(np.float32), meta


def _apply_residual_to_arrays(arrays: Dict[str, np.ndarray], rest_names: Tuple[str, ...], codebook_path: Path, assignments_path: Path) -> None:
    if not codebook_path.exists() or not assignments_path.exists():
        return
    codebook = np.load(codebook_path)
    assignments = np.load(assignments_path)
    residual = codebook[assignments]
    for idx, name in enumerate(rest_names):
        base = arrays.get(name)
        if base is None:
            base = np.zeros(residual.shape[0], dtype=np.float32)
        arrays[name] = base + residual[:, idx]


def decode_compressed_iteration(
    model_path: Path,
    iteration: int,
    *,
    device: str = "cuda",
    chunk: int = -1,
) -> Dict[str, np.ndarray]:
    """Return field arrays reconstructed from compressed assets."""
    point_dir = model_path / "point_cloud" / f"iteration_{iteration}"
    compressed_dir = point_dir / "compressed"
    if not compressed_dir.exists():
        raise FileNotFoundError(f"No compressed data at {compressed_dir}")
    pack_dir_lossless = compressed_dir / "lossless"
    pack_dir_restfree = compressed_dir / "restfree"
    latents_path = compressed_dir / "latents.npz"
    checkpoint_path = compressed_dir / "rest_autoencoder.pt"
    if pack_dir_lossless.exists():
        from utils.lossless_pack import unpack_ply_to_arrays

        arrays, _ = unpack_ply_to_arrays(pack_dir_lossless)
    elif pack_dir_restfree.exists():
        from tools import restfree_quant_pack as restfree

        arrays, _ = restfree.unpack_restfree_to_arrays(pack_dir_restfree)
    else:
        raise FileNotFoundError(f"No lossless or restfree pack found in {compressed_dir}")
    latents, _ = _load_latent_matrix(latents_path)
    from tools import rest_autoencoder as rest

    model, target_mean, target_std, _, _, target_names, _ = rest._load_checkpoint(checkpoint_path, device)
    count = latents.shape[0]
    for name in target_names:
        if name not in arrays:
            arrays[name] = np.zeros(count, dtype=np.float32)
    target_mean_t = torch.from_numpy(target_mean).to(device)
    target_std_t = torch.from_numpy(target_std).to(device)
    recon = np.empty((latents.shape[0], len(target_names)), dtype=np.float32)
    chunk_size = latents.shape[0] if chunk <= 0 else chunk
    with torch.no_grad():
        for start in range(0, latents.shape[0], chunk_size):
            end = min(start + chunk_size, latents.shape[0])
            batch = torch.from_numpy(latents[start:end]).to(device)
            rec = model.decode_batch(batch) * target_std_t + target_mean_t
            recon[start:end] = rec.cpu().numpy()
    for idx, name in enumerate(target_names):
        arrays[name] = recon[:, idx]
    _apply_residual_to_arrays(
        arrays,
        tuple(target_names),
        compressed_dir / "residual_codebook.npy",
        compressed_dir / "residual_assignments.npy",
    )
    return arrays


def load_gaussians_from_compressed(
    gaussians,
    model_path: Path,
    iteration: int,
    *,
    device: str = "cuda",
    chunk: int = -1,
) -> None:
    """Populate a GaussianModel from compressed iteration assets."""
    arrays = decode_compressed_iteration(model_path, iteration, device=device, chunk=chunk)
    gaussians.load_from_arrays(arrays)
