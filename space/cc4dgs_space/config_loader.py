"""Minimal replacement for ``mmcv.Config.fromfile`` used by the scene configs.

The configs under ``arguments/`` are plain Python files that may declare
``_base_ = './default.py'`` and override nested dictionaries. mmcv 1.x is not
installable on the Python versions supported by ZeroGPU, so this module
implements the subset that the CC-4DGS configs use: ``_base_`` inheritance
(string or list) and recursive dictionary merging with ``_delete_`` support.
"""

from __future__ import annotations

import copy
import runpy
from pathlib import Path
from typing import Any, Dict

SECTIONS = ("OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams")


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("_delete_", False):
            out[key] = {k: v for k, v in value.items() if k != "_delete_"}
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    namespace = runpy.run_path(str(path))
    cfg = {k: v for k, v in namespace.items() if not k.startswith("__") and k != "_base_"}
    cfg = {k: v for k, v in cfg.items() if isinstance(v, (dict, list, tuple, str, int, float, bool, type(None)))}

    bases = namespace.get("_base_")
    if bases is None:
        return cfg
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for base in bases:
        merged = _merge(merged, load_config(path.parent / base))
    return _merge(merged, cfg)


def merge_into_args(args, cfg: Dict[str, Any]):
    """Same semantics as ``utils.params_utils.merge_hparams``."""
    for section in SECTIONS:
        for key, value in cfg.get(section, {}).items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args
