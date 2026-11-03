"""Inference-only runtime for CC-4DGS checkpoints.

Everything that does not need a GPU (parsing the PLY / CCA bundle, building the
CDF deformation network, camera trajectories) runs on CPU in the main process.
Only :func:`render_views` touches CUDA, so it is the single function that has to
run inside a ZeroGPU ``@spaces.GPU`` call.

The rendering itself reuses the repository's own ``gaussian_renderer.render`` and
``scene.deformation.deform_network`` so that the output matches ``render.py``.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import os
import re
import shutil
import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import sys


def _find_repo_root() -> Path:
    """Directory holding the research code (``arguments/``, ``scene/``, ...).

    That is the Space root once deployed, and the repository root (one level
    above ``space/``) when running from a git checkout.
    """
    env = os.environ.get("CC4DGS_REPO_ROOT")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "gaussian_renderer").is_dir() and (parent / "arguments").is_dir():
            return parent
    raise RuntimeError("Cannot locate the CC-4DGS sources; set CC4DGS_REPO_ROOT.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .config_loader import load_config, merge_into_args  # noqa: E402

CONFIG_DIRS = ("arguments/dynerf", "arguments/technicolor")
DEFAULT_CONFIG = "arguments/dynerf/default.py"
# Scenes whose cam09 is excluded by scene/neural_3D_dataset_NDC.py.
N3DV_CAM09_EXCLUDED = {"flame_steak", "cook_spinach", "cut_roasted_beef", "sear_steak"}
N3DV_IMG_WH = (1352, 1014)
WORK_ROOT = Path(os.environ.get("CC4DGS_WORK_DIR", Path(tempfile.gettempdir()) / "cc4dgs_space"))


# --------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------

def list_configs() -> List[str]:
    out = []
    for d in CONFIG_DIRS:
        for p in sorted((REPO_ROOT / d).glob("*.py")):
            out.append(str(p.relative_to(REPO_ROOT)))
    return out


def _config_for(path: str):
    """Return (lp, hp, op, pp) parameter groups with ``path`` merged in."""
    from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams

    parser = ArgumentParser()
    lp, op, pp, hp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser), ModelHiddenParams(parser)
    args = parser.parse_args([])
    cfg_path = REPO_ROOT / path
    if cfg_path.stat().st_size == 0:
        cfg_path = cfg_path.parent / "default.py"
    merge_into_args(args, load_config(cfg_path))
    lpe, hpe, ope, ppe = lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args)
    # Same overrides as render.py.
    hpe.h2m_render_mode = True
    hpe.smooth_cfg = dict(hpe.smooth_cfg, smooth=False)
    return lpe, hpe, ope, ppe


def guess_config(model_dir: Path) -> List[str]:
    """Configs ordered by how well their scene name matches the checkpoint path."""
    configs = list_configs()
    text = str(model_dir).lower()
    named = [c for c in configs if Path(c).stem.lower() in text and Path(c).stem != "default"]
    if "technicolor" in text or any("technicolor" in c for c in named):
        defaults = ["arguments/technicolor/default.py"]
    else:
        defaults = [DEFAULT_CONFIG]
    rest = [c for c in configs if c not in named and c not in defaults]
    return named + defaults + rest


# --------------------------------------------------------------------------------------
# Checkpoint discovery and loading (CPU only)
# --------------------------------------------------------------------------------------

def _iteration_of(path: Path) -> int:
    m = re.search(r"iteration_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def find_iteration_dir(root: Path, iteration: Optional[int] = None) -> Path:
    candidates = []
    for deform in root.rglob("deformation.pth"):
        d = deform.parent
        if (d / "point_cloud.ply").exists() or (d / "compressed").is_dir():
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Expected <scene>/point_cloud/iteration_<N>/ with deformation.pth "
            "and point_cloud.ply or a compressed/ bundle."
        )
    fine = [c for c in candidates if not c.name.startswith("coarse")] or candidates
    if iteration is not None and iteration >= 0:
        exact = [c for c in fine if _iteration_of(c) == iteration]
        if exact:
            return exact[0]
    return max(fine, key=_iteration_of)


def _read_ply_arrays(path: Path) -> Dict[str, np.ndarray]:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply.elements[0]
    return {name: np.asarray(vertex[name]) for name in vertex.data.dtype.names}


def _decode_compressed(iter_dir: Path) -> Dict[str, np.ndarray]:
    from utils.compressed_loader import decode_compressed_iteration

    # decode_compressed_iteration expects <model>/point_cloud/iteration_<N>/compressed.
    staging = Path(tempfile.mkdtemp(prefix="cca-", dir=WORK_ROOT))
    (staging / "point_cloud").mkdir()
    (staging / "point_cloud" / "iteration_0").symlink_to(iter_dir.resolve(), target_is_directory=True)
    try:
        return decode_compressed_iteration(staging, 0, device="cpu", chunk=262144)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _arrays_to_tensors(arrays: Dict[str, np.ndarray], sh_degree: int) -> Dict[str, torch.Tensor]:
    """CPU version of GaussianModel._assign_from_arrays."""

    def req(name):
        if name not in arrays:
            raise KeyError(f"Field '{name}' missing from point cloud.")
        return np.asarray(arrays[name], dtype=np.float32)

    def numbered(prefix):
        names = [n for n in arrays if n.startswith(prefix)]
        return sorted(names, key=lambda n: int(n.split("_")[-1]))

    n = req("x").shape[0]
    xyz = np.stack([req("x"), req("y"), req("z")], axis=1)
    f_dc = np.stack([req(f"f_dc_{i}") for i in range(3)], axis=1).reshape(n, 3, 1)
    rest_names = numbered("f_rest_")
    expected = 3 * (sh_degree + 1) ** 2 - 3
    if len(rest_names) != expected:
        raise ValueError(f"Expected {expected} f_rest_* fields, found {len(rest_names)}.")
    f_rest = np.stack([req(k) for k in rest_names], axis=1).reshape(n, 3, (sh_degree + 1) ** 2 - 1)
    dynamic = req("dynamic") if "dynamic" in arrays else np.zeros(n, dtype=np.float32)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    return {
        "xyz": t(xyz),
        "f_dc": t(f_dc).transpose(1, 2).contiguous(),
        "f_rest": t(f_rest).transpose(1, 2).contiguous(),
        "opacity": t(req("opacity")[:, None]),
        "scaling": t(np.stack([req(k) for k in numbered("scale_")], axis=1)),
        "rotation": t(np.stack([req(k) for k in numbered("rot")], axis=1)),
        "dynamic": t(dynamic[:, None]),
    }


def _build_deformation(hp, state: Dict[str, torch.Tensor], xyz: torch.Tensor):
    from scene.deformation import deform_network

    net = deform_network(hp, None)
    aabb_key = next((k for k in state if k.endswith("grid.aabb")), None)
    if aabb_key is not None:
        aabb = state[aabb_key]
        net.deformation_net.set_aabb(aabb[0].numpy(), aabb[1].numpy())
    else:  # pragma: no cover - old checkpoints
        net.deformation_net.set_aabb(xyz.max(0).values.numpy(), xyz.min(0).values.numpy())
    net.load_state_dict(state, strict=True)
    return net.eval()


def _read_white_background(model_root: Path) -> Optional[bool]:
    """Parse ``cfg_args`` (a Namespace repr) without eval()."""
    for cfg in model_root.rglob("cfg_args"):
        try:
            call = ast.parse(cfg.read_text().strip(), mode="eval").body
            for kw in getattr(call, "keywords", []):
                if kw.arg == "white_background":
                    return bool(ast.literal_eval(kw.value))
        except (SyntaxError, ValueError):
            continue
    return None


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@dataclass
class LoadedScene:
    key: str
    name: str
    config: str
    iter_dir: Path
    source: str  # "ply" or "cca"
    sh_degree: int
    dynamic_thro: float
    white_background: bool
    tensors: Dict[str, torch.Tensor]
    deformation: torch.nn.Module
    pipe: object
    storage: Dict[str, int] = field(default_factory=dict)
    config_matched_by: str = "user"  # "user", "name" or "architecture"

    @property
    def num_points(self) -> int:
        return int(self.tensors["xyz"].shape[0])

    @property
    def num_dynamic(self) -> int:
        return int((self.tensors["dynamic"] > self.dynamic_thro).sum())


_SCENES: Dict[str, LoadedScene] = {}
_CACHE_LIMIT = int(os.environ.get("CC4DGS_SCENE_CACHE", "3"))


def load_scene(model_root: str | os.PathLike, config: str = "auto", prefer_compressed: bool = True,
               iteration: Optional[int] = None) -> LoadedScene:
    model_root = Path(model_root)
    iter_dir = find_iteration_dir(model_root, iteration)
    use_cca = (iter_dir / "compressed").is_dir() and (prefer_compressed or not (iter_dir / "point_cloud.ply").exists())

    key_src = f"{iter_dir.resolve()}|{config}|{use_cca}"
    key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    if key in _SCENES:
        return _SCENES[key]

    candidates = guess_config(model_root) if config == "auto" else [config]
    named = {c for c in candidates if Path(c).stem.lower() in str(model_root).lower() and Path(c).stem != "default"}
    state = torch.load(iter_dir / "deformation.pth", map_location="cpu", weights_only=True)

    lp = hp = op = pp = deformation = None
    errors = []
    chosen = None
    for cand in candidates:
        lp, hp, op, pp = _config_for(cand)
        try:
            deformation = _build_deformation(hp, state, torch.zeros(1, 3))
            chosen = cand
            break
        except (RuntimeError, ValueError, KeyError) as exc:
            errors.append(f"{cand}: {str(exc).splitlines()[0]}")
    if chosen is None:
        raise RuntimeError("deformation.pth does not match any config:\n" + "\n".join(errors[:5]))

    arrays = _decode_compressed(iter_dir) if use_cca else _read_ply_arrays(iter_dir / "point_cloud.ply")
    tensors = _arrays_to_tensors(arrays, lp.sh_degree)

    white = _read_white_background(model_root)
    storage = {"deformation.pth": _dir_size(iter_dir / "deformation.pth")}
    if (iter_dir / "point_cloud.ply").exists():
        storage["point_cloud.ply"] = _dir_size(iter_dir / "point_cloud.ply")
    if (iter_dir / "compressed").is_dir():
        storage["compressed/ (CCA)"] = _dir_size(iter_dir / "compressed")

    scene = LoadedScene(
        key=key,
        name=model_root.name if model_root.name not in ("", ".") else iter_dir.parent.parent.name,
        config=chosen,
        iter_dir=iter_dir,
        source="cca" if use_cca else "ply",
        sh_degree=lp.sh_degree,
        dynamic_thro=float(op.dynamic_thro),
        white_background=lp.white_background if white is None else white,
        tensors=tensors,
        deformation=deformation,
        pipe=pp,
        storage=storage,
        config_matched_by="user" if config != "auto" else ("name" if chosen in named else "architecture"),
    )
    _SCENES[key] = scene
    while len(_SCENES) > _CACHE_LIMIT:
        _SCENES.pop(next(iter(_SCENES)))
    return scene


def get_scene(key: str) -> LoadedScene:
    if key not in _SCENES:
        raise KeyError("Scene is no longer loaded. Please load it again.")
    return _SCENES[key]


# --------------------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------------------

@dataclass
class ViewSpec:
    R: np.ndarray  # camera-to-world rotation (columns = camera axes), repo convention
    T: np.ndarray  # world-to-camera translation
    fovx: float
    fovy: float
    width: int
    height: int
    cxr: float = 0.0
    cyr: float = 0.0
    name: str = ""

    def scaled(self, scale: float) -> "ViewSpec":
        # Even sizes keep the H.264 (yuv420p) encoder from resizing frames.
        w = max(16, int(round(self.width * scale / 2)) * 2)
        h = max(16, int(round(self.height * scale / 2)) * 2)
        return ViewSpec(self.R, self.T, self.fovx, self.fovy, w, h, self.cxr, self.cyr, self.name)


@dataclass
class CameraRig:
    kind: str
    fixed: List[ViewSpec]
    path: List[ViewSpec]

    def describe(self) -> str:
        return f"{self.kind}: {len(self.fixed)} cameras, {len(self.path)}-view trajectory"


def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def look_at(pos, target, up) -> Tuple[np.ndarray, np.ndarray]:
    """OpenCV-convention camera (x right, y down, z forward) in the repo's (R, T) format."""
    z = _normalize(np.asarray(target, dtype=np.float64) - pos)
    x = _normalize(np.cross(-np.asarray(up, dtype=np.float64), z))
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    T = -R.T @ np.asarray(pos, dtype=np.float64)
    return R, T


def _focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def rig_from_poses_bounds(path: Path, scene_name: str = "", n_views: int = 300) -> CameraRig:
    """Same cameras/spiral as scene/neural_3D_dataset_NDC.py + format_render_poses."""
    from scene.neural_3D_dataset_NDC import get_spiral

    arr = np.load(path)
    poses = arr[:, :-2].reshape([-1, 3, 5])
    near_fars = arr[:, -2:]
    keep = list(range(len(poses)))
    if scene_name in N3DV_CAM09_EXCLUDED and len(poses) > 9:
        keep.remove(9)
    poses, near_fars = poses[keep], near_fars[keep]
    W, H = N3DV_IMG_WH
    focal = poses[0, 2, -1] / (2704 / W)
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    fovx, fovy = _focal2fov(focal, W), _focal2fov(focal, H)

    def to_view(p, name):
        R = -p[:3, :3].copy()
        R[:, 0] = -R[:, 0]
        T = -p[:3, 3].dot(R)
        return ViewSpec(R, T, fovx, fovy, W, H, name=name)

    cam_ids = [i for i in range(len(arr)) if i in keep]
    fixed = [to_view(p, f"cam{cid:02d}") for p, cid in zip(poses, cam_ids)]
    spiral = get_spiral(poses, near_fars, N_views=n_views)
    path_views = [to_view(p, f"spiral_{i:03d}") for i, p in enumerate(spiral)]
    return CameraRig("N3DV poses_bounds.npy", fixed, path_views)


def _read_colmap(sparse_dir: Path):
    from scene.colmap_loader import (
        read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text,
    )

    if (sparse_dir / "images.bin").exists():
        return read_extrinsics_binary(str(sparse_dir / "images.bin")), read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    return read_extrinsics_text(str(sparse_dir / "images.txt")), read_intrinsics_text(str(sparse_dir / "cameras.txt"))


def _spiral_from_views(views: Sequence[ViewSpec], xyz: np.ndarray, n_views: int) -> List[ViewSpec]:
    centers = np.stack([-v.R @ v.T for v in views])
    c = centers.mean(0)
    z = _normalize(np.mean([v.R[:, 2] for v in views], axis=0))
    up = _normalize(-np.mean([v.R[:, 1] for v in views], axis=0))
    depth = (xyz - c) @ z
    depth = depth[depth > 0]
    focus = float(np.percentile(depth, 50)) if depth.size else 1.0
    x_axis = _normalize(np.cross(-up, z))
    y_axis = np.cross(z, x_axis) * -1.0  # world "up" in the image plane
    offs = centers - c
    rx = float(np.percentile(np.abs(offs @ x_axis), 90)) if len(views) > 1 else 0.0
    ry = float(np.percentile(np.abs(offs @ y_axis), 90)) if len(views) > 1 else 0.0
    rx = rx if rx > 1e-6 else 0.05 * focus
    ry = ry if ry > 1e-6 else 0.05 * focus
    target = c + focus * z
    ref = views[0]
    out = []
    for i, th in enumerate(np.linspace(0, 2 * np.pi, n_views, endpoint=False)):
        pos = c + rx * math.cos(th) * x_axis + ry * math.sin(th) * y_axis + 0.05 * focus * math.sin(0.5 * th) * z
        R, T = look_at(pos, target, up)
        out.append(ViewSpec(R, T, ref.fovx, ref.fovy, ref.width, ref.height, name=f"spiral_{i:03d}"))
    return out


def rig_from_colmap(sparse_dir: Path, xyz: np.ndarray, n_views: int = 300) -> CameraRig:
    from scene.colmap_loader import qvec2rotmat

    extr, intr = _read_colmap(sparse_dir)
    views = []
    for key in sorted(extr, key=lambda k: extr[k].name):
        e = extr[key]
        cam = intr[e.camera_id]
        p = cam.params
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:  # SIMPLE_PINHOLE / SIMPLE_RADIAL / OPENCV ...: use the focal and principal point only
            fx = p[0]
            fy = p[1] if cam.model in ("OPENCV", "FULL_OPENCV") else p[0]
            cx, cy = (p[2], p[3]) if cam.model in ("OPENCV", "FULL_OPENCV") else (p[1], p[2])
        views.append(ViewSpec(
            np.transpose(qvec2rotmat(e.qvec)), np.asarray(e.tvec),
            _focal2fov(fx, cam.width), _focal2fov(fy, cam.height), cam.width, cam.height,
            cx / cam.width - 0.5, cy / cam.height - 0.5, name=os.path.splitext(e.name)[0],
        ))
    if not views:
        raise ValueError(f"No cameras in {sparse_dir}")
    return CameraRig("COLMAP sparse model", views, _spiral_from_views(views, xyz, n_views))


def rig_orbit(xyz: np.ndarray, n_views: int = 240, width: int = 960, height: int = 540,
              fov_deg: float = 60.0, elevation_deg: float = 15.0, distance_scale: float = 1.0) -> CameraRig:
    """Fallback when no camera file is available: orbit around the point cloud."""
    center = np.median(xyz, axis=0)
    radius = float(np.percentile(np.linalg.norm(xyz - center, axis=1), 60)) * 1.5 * distance_scale
    up = np.array([0.0, -1.0, 0.0])  # COLMAP worlds are usually y-down
    fovy = math.radians(fov_deg)
    fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
    el = math.radians(elevation_deg)

    def view(th, name):
        pos = center + radius * np.array([math.cos(el) * math.sin(th), -math.sin(el), -math.cos(el) * math.cos(th)])
        R, T = look_at(pos, center, up)
        return ViewSpec(R, T, fovx, fovy, width, height, name=name)

    fixed = [view(th, f"orbit_{int(math.degrees(th)):03d}deg") for th in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    path = [view(th, f"orbit_{i:03d}") for i, th in enumerate(np.linspace(0, 2 * np.pi, n_views, endpoint=False))]
    return CameraRig("orbit (no camera file found)", fixed, path)


def find_camera_rig(search_roots: Sequence[Path], scene: LoadedScene, n_views: int = 300) -> CameraRig:
    xyz = scene.tensors["xyz"].numpy()
    opaque = torch.sigmoid(scene.tensors["opacity"][:, 0]).numpy() > 0.5
    xyz_ref = xyz[opaque] if opaque.sum() > 100 else xyz
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        hits = sorted(root.rglob("poses_bounds.npy"))
        if hits:
            return rig_from_poses_bounds(hits[0], scene_name=_n3dv_name(scene, hits[0]), n_views=n_views)
        for pattern in ("images.bin", "images.txt"):
            for hit in sorted(root.rglob(pattern)):
                cams = hit.parent / pattern.replace("images", "cameras")
                if cams.exists():
                    return rig_from_colmap(hit.parent, xyz_ref, n_views=n_views)
    return rig_orbit(xyz_ref)


def _n3dv_name(scene: LoadedScene, poses_path: Path) -> str:
    text = f"{poses_path} {scene.iter_dir} {scene.config}".lower()
    for name in N3DV_CAM09_EXCLUDED | {"coffee_martini", "flame_salmon_1"}:
        if name in text:
            return name
    return ""


# --------------------------------------------------------------------------------------
# GPU rendering
# --------------------------------------------------------------------------------------

class _RenderCam:
    def __init__(self, v: ViewSpec, time: float, device: str):
        from utils.graphics_utils import getProjectionMatrix, getProjectionMatrixCV, getWorld2View2

        self.FoVx, self.FoVy = v.fovx, v.fovy
        self.image_width, self.image_height = v.width, v.height
        self.time = float(time)
        wv = torch.tensor(getWorld2View2(v.R, v.T)).transpose(0, 1)
        if abs(v.cxr) > 1e-8 or abs(v.cyr) > 1e-8:
            proj = getProjectionMatrixCV(0.01, 100.0, v.fovx, v.fovy, cx=v.cxr, cy=v.cyr).transpose(0, 1)
        else:
            proj = getProjectionMatrix(0.01, 100.0, v.fovx, v.fovy).transpose(0, 1)
        self.world_view_transform = wv.to(device)
        self.full_proj_transform = (wv.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0).to(device)
        self.camera_center = wv.inverse()[3, :3].to(device)


class _RenderGaussians:
    """The attribute surface of GaussianModel that gaussian_renderer.render() uses."""

    scaling_activation = staticmethod(torch.exp)
    rotation_activation = staticmethod(F.normalize)
    opacity_activation = staticmethod(torch.sigmoid)

    def __init__(self, scene: LoadedScene, device: str):
        t = {k: v.to(device, non_blocking=True) for k, v in scene.tensors.items()}
        self._xyz, self._features_dc, self._features_rest = t["xyz"], t["f_dc"], t["f_rest"]
        self._opacity, self._scaling, self._rotation, self._dynamic = t["opacity"], t["scaling"], t["rotation"], t["dynamic"]
        self.max_sh_degree = self.active_sh_degree = scene.sh_degree
        self.dynamic_thro = scene.dynamic_thro
        # Fresh copy per call: the CDF render-mode caches must never outlive a GPU session.
        self._deformation = copy.deepcopy(scene.deformation).to(device)
        self._features = torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_dynamic(self):
        return self._dynamic


def render_views(scene: LoadedScene, views: Sequence[ViewSpec], times: Sequence[float],
                 background: Optional[Sequence[float]] = None, device: str = "cuda") -> List[np.ndarray]:
    """Render ``views[i]`` at ``times[i]``; returns HxWx3 uint8 frames."""
    from gaussian_renderer import render

    if background is None:
        background = [1.0, 1.0, 1.0] if scene.white_background else [0.0, 0.0, 0.0]
    bg = torch.tensor(background, dtype=torch.float32, device=device)
    frames = []
    with torch.no_grad():
        pc = _RenderGaussians(scene, device)
        for v, t in zip(views, times):
            out = render(_RenderCam(v, t, device), pc, scene.pipe, bg, stage="fine", cam_type=None, compute_smooth=False)
            img = (out["render"].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).contiguous()
            frames.append(img.cpu().numpy())
        del pc
    return frames
