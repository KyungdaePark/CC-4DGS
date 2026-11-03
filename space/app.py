"""CC-4DGS viewer for Hugging Face Spaces (ZeroGPU).

Loads a trained CC-4DGS checkpoint (raw CDF ``point_cloud.ply`` or the CDF+CCA
``compressed/`` bundle, plus ``deformation.pth``) and renders novel views of the
dynamic scene at any time step. Only the rendering functions request a GPU.
"""

from __future__ import annotations

# `spaces` must be imported before torch so ZeroGPU can patch CUDA initialisation.
try:
    import spaces
except ImportError:  # running outside Hugging Face
    class _NoSpaces:
        @staticmethod
        def GPU(fn=None, **_kwargs):
            return fn if fn is not None else (lambda f: f)

    spaces = _NoSpaces()

import os
import shutil
import sys
import uuid
import zipfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from cc4dgs_space import runtime as rt  # noqa: E402
from cc4dgs_space.extensions import ensure_extensions  # noqa: E402

ensure_extensions(rt.REPO_ROOT, wheel_dir=APP_DIR / "wheels")

import gradio as gr  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

rt.WORK_ROOT.mkdir(parents=True, exist_ok=True)
MAX_ZIP_BYTES = int(os.environ.get("CC4DGS_MAX_ZIP_MB", "4096")) * 1024 * 1024
DEFAULT_REPO = os.environ.get("CC4DGS_HUB_REPO", "")
DEFAULT_SUBFOLDER = os.environ.get("CC4DGS_HUB_SUBFOLDER", "")
DEFAULT_REPO_TYPE = os.environ.get("CC4DGS_HUB_REPO_TYPE", "model")
BACKGROUNDS = {"Checkpoint default": None, "White": [1.0, 1.0, 1.0], "Black": [0.0, 0.0, 0.0]}


# --------------------------------------------------------------------------------------
# Checkpoint input
# --------------------------------------------------------------------------------------

def _safe_extract(zip_path: str, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        total = sum(i.file_size for i in zf.infolist())
        if total > MAX_ZIP_BYTES:
            raise gr.Error(f"Zip expands to {total / 1e9:.1f} GB, above the {MAX_ZIP_BYTES / 1e9:.1f} GB limit.")
        root = dest.resolve()
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(root) + os.sep) and target != root:
                raise gr.Error(f"Unsafe path in zip: {info.filename}")
        zf.extractall(dest)


def _model_dir_from_inputs(source, zip_file, repo_id, subfolder, repo_type, session_dir: Path) -> Path:
    if source == "Upload zip":
        if not zip_file:
            raise gr.Error("Upload a checkpoint .zip first.")
        # Name the folder after the zip so the scene name can select the config.
        dest = session_dir / Path(zip_file).stem
        _safe_extract(zip_file, dest)
        entries = [p for p in dest.iterdir() if not p.name.startswith(("__MACOSX", "."))]
        return entries[0] if len(entries) == 1 and entries[0].is_dir() else dest
    if not repo_id.strip():
        raise gr.Error("Enter a Hugging Face repo id (e.g. user/cc4dgs-checkpoints).")
    from huggingface_hub import snapshot_download

    sub = subfolder.strip().strip("/")
    local = snapshot_download(
        repo_id=repo_id.strip(),
        repo_type=repo_type,
        allow_patterns=[f"{sub}/**"] if sub else None,
        local_dir=str(rt.WORK_ROOT / "hub" / repo_id.strip().replace("/", "__")),
        token=os.environ.get("HF_TOKEN"),
    )
    model_dir = Path(local) / sub if sub else Path(local)
    if not model_dir.exists():
        raise gr.Error(f"Subfolder '{sub}' not found in {repo_id}.")
    return model_dir


def _fmt_bytes(n: int) -> str:
    return f"{n / 1e6:.2f} MB"


def load_checkpoint(source, zip_file, repo_id, subfolder, repo_type, camera_files, config, prefer_cca,
                    progress=gr.Progress()):
    session_dir = rt.WORK_ROOT / "sessions" / uuid.uuid4().hex[:12]
    session_dir.mkdir(parents=True, exist_ok=True)
    progress(0.1, desc="Fetching checkpoint")
    model_dir = _model_dir_from_inputs(source, zip_file, repo_id, subfolder, repo_type, session_dir)

    cam_dir = session_dir / "cameras"
    for f in camera_files or []:
        cam_dir.mkdir(exist_ok=True)
        shutil.copy(f, cam_dir / Path(f).name)

    progress(0.4, desc="Decoding Gaussians and building the CDF")
    try:
        scene = rt.load_scene(model_dir, config=config, prefer_compressed=prefer_cca)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc

    progress(0.8, desc="Preparing cameras")
    rig = rt.find_camera_rig([cam_dir, model_dir], scene)
    state = {
        "key": scene.key,
        "model_dir": str(model_dir),
        "config": scene.config,
        "prefer_cca": scene.source == "cca",
        "rig": rig,
        "out_dir": str(session_dir),
    }

    deploy = sum(v for k, v in scene.storage.items() if k != "point_cloud.ply") if scene.source == "cca" else sum(
        v for k, v in scene.storage.items() if k != "compressed/ (CCA)")
    storage = "\n".join(f"| `{k}` | {_fmt_bytes(v)} |" for k, v in scene.storage.items())
    status = (
        f"### Loaded `{scene.name}`\n"
        f"- checkpoint: `{scene.iter_dir.name}` · attributes from **{'CCA bundle' if scene.source == 'cca' else 'PLY'}**\n"
        f"- config: `{scene.config}`\n"
        f"- Gaussians: **{scene.num_points:,}** ({scene.num_dynamic:,} dynamic, deformed by the CDF)\n"
        f"- cameras: {rig.describe()}\n\n"
        f"| file | size |\n|---|---|\n{storage}\n| **rendered from** | **{_fmt_bytes(deploy)}** |"
    )
    if scene.config_matched_by == "architecture":
        status += ("\n\n> The config was picked because its CDF architecture matches `deformation.pth`; several "
                   "scene configs share an architecture but differ in deformation flags. Select the scene's config "
                   "explicitly if the motion looks wrong.")
    if rig.kind.startswith("orbit"):
        status += ("\n\n> No `poses_bounds.npy` or COLMAP `images.bin`/`cameras.bin` found, so an orbit around the "
                   "point cloud is used. Add the dataset's camera file for the original viewpoints.")
    names = [v.name for v in rig.fixed]
    return state, status, gr.update(choices=names, value=names[0]), gr.update(interactive=True), gr.update(interactive=True)


# --------------------------------------------------------------------------------------
# GPU work
# --------------------------------------------------------------------------------------

def _scene_from_state(state):
    try:
        return rt.get_scene(state["key"])
    except KeyError:
        # A fresh ZeroGPU worker or an evicted cache entry: rebuild from disk.
        return rt.load_scene(state["model_dir"], config=state["config"], prefer_compressed=state["prefer_cca"])


def _pick_view(state, mode, cam_name, path_pos, scale):
    rig = state["rig"]
    if mode == "Fixed camera":
        view = next((v for v in rig.fixed if v.name == cam_name), rig.fixed[0])
    else:
        idx = min(int(round(path_pos * (len(rig.path) - 1))), len(rig.path) - 1)
        view = rig.path[idx]
    return view.scaled(scale)


@spaces.GPU(duration=30)
def _render_frame_gpu(state, view, t, background):
    scene = _scene_from_state(state)
    return rt.render_views(scene, [view], [t], background=background)[0]


def render_frame(state, mode, cam_name, path_pos, t, scale, bg_name):
    if not state:
        raise gr.Error("Load a checkpoint first.")
    view = _pick_view(state, mode, cam_name, path_pos, scale)
    return _render_frame_gpu(state, view, float(t), BACKGROUNDS[bg_name])


def render_frame_if_loaded(state, mode, cam_name, path_pos, t, scale, bg_name):
    """Slider callback: silently ignore moves before a checkpoint is loaded."""
    if not state:
        return gr.update()
    return render_frame(state, mode, cam_name, path_pos, t, scale, bg_name)


def _video_duration(state, views, times, background, fps, out_path):
    pixels = views[0].width * views[0].height if views else 0
    return int(min(300, 20 + len(views) * (0.05 + pixels / 2.0e7)))


@spaces.GPU(duration=_video_duration)
def _render_video_gpu(state, views, times, background, fps, out_path):
    scene = _scene_from_state(state)
    frames = rt.render_views(scene, views, times, background=background)
    imageio.mimwrite(out_path, frames, fps=fps, quality=8, macro_block_size=1)
    return out_path


def render_video(state, mode, cam_name, path_pos, t, scale, bg_name, n_frames, fps, motion, sweep_time):
    if not state:
        raise gr.Error("Load a checkpoint first.")
    rig = state["rig"]
    n = int(n_frames)
    if motion == "Move along trajectory":
        idx = [int(round(i * (len(rig.path) - 1) / max(n - 1, 1))) for i in range(n)]
        views = [rig.path[i].scaled(scale) for i in idx]
    else:
        views = [_pick_view(state, mode, cam_name, path_pos, scale)] * n
    times = [i / max(n - 1, 1) for i in range(n)] if sweep_time else [float(t)] * n
    out_path = str(Path(state["out_dir"]) / f"render_{uuid.uuid4().hex[:8]}.mp4")
    return _render_video_gpu(state, views, times, BACKGROUNDS[bg_name], int(fps), out_path)


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

HEADER = """
# CC-4DGS: Computational Deformation and Point-Cloud Compression for Storage-Efficient Dynamic Gaussian Splatting
**IEEE TVCG 2026** · [arXiv](https://arxiv.org/abs/2609.02184) · [Code](https://github.com/KyungdaePark/CC-4DGS) ·
[Pretrained models](https://drive.google.com/drive/folders/1wSE9Nn9Xo8flSjNXRZsqgGDnOYsqV6qV?usp=drive_link)

Load a trained scene (a zip of the scene folder, or a folder in a Hugging Face repo) that contains
`point_cloud/iteration_<N>/deformation.pth` with `point_cloud.ply` (CDF) or `compressed/` (CDF+CCA).
Add the dataset's `poses_bounds.npy` (N3DV) or COLMAP `images.bin` + `cameras.bin` (Technicolor) to
render from the original cameras and spiral path.
"""

with gr.Blocks(title="CC-4DGS") as demo:
    state = gr.State(None)
    gr.Markdown(HEADER)
    with gr.Row():
        with gr.Column(scale=1):
            source = gr.Radio(["Upload zip", "Hugging Face Hub"], value="Hugging Face Hub" if DEFAULT_REPO else "Upload zip",
                              label="Checkpoint source")
            zip_file = gr.File(label="Scene folder (.zip)", file_types=[".zip"], type="filepath")
            with gr.Group():
                repo_id = gr.Textbox(label="Repo id", value=DEFAULT_REPO, placeholder="user/cc4dgs-checkpoints")
                subfolder = gr.Textbox(label="Subfolder (scene)", value=DEFAULT_SUBFOLDER,
                                       placeholder="CC-4DGS_N3DV_CDF_CCA/cut_roasted_beef")
                repo_type = gr.Radio(["model", "dataset"], value=DEFAULT_REPO_TYPE, label="Repo type")
            camera_files = gr.File(label="Camera files (optional): poses_bounds.npy, or images.bin + cameras.bin",
                                   file_count="multiple", type="filepath")
            config = gr.Dropdown(["auto"] + rt.list_configs(), value="auto", label="Scene config",
                                 info="'auto' matches the scene name and checks the deformation.pth architecture.")
            prefer_cca = gr.Checkbox(True, label="Use the CCA compressed bundle when available")
            load_btn = gr.Button("Load checkpoint", variant="primary")
            status = gr.Markdown()
        with gr.Column(scale=2):
            image = gr.Image(label="Render", type="numpy", format="png", interactive=False)
            t = gr.Slider(0.0, 1.0, value=0.0, step=1 / 299, label="Time t (normalised)")
            with gr.Row():
                mode = gr.Radio(["Trajectory", "Fixed camera"], value="Trajectory", label="Camera")
                cam_name = gr.Dropdown([], label="Fixed camera")
            path_pos = gr.Slider(0.0, 1.0, value=0.0, step=0.001, label="Position on trajectory")
            with gr.Row():
                scale = gr.Slider(0.25, 1.0, value=0.5, step=0.05, label="Resolution scale")
                bg_name = gr.Radio(list(BACKGROUNDS), value="Checkpoint default", label="Background")
            frame_btn = gr.Button("Render frame", interactive=False)
            with gr.Accordion("Video", open=True):
                with gr.Row():
                    n_frames = gr.Slider(10, 300, value=120, step=1, label="Frames")
                    fps = gr.Slider(10, 60, value=30, step=1, label="FPS")
                with gr.Row():
                    motion = gr.Radio(["Move along trajectory", "Hold current view"], value="Move along trajectory",
                                      label="Camera motion")
                    sweep_time = gr.Checkbox(True, label="Sweep time 0 → 1 (otherwise hold t)")
                video_btn = gr.Button("Render video", variant="primary", interactive=False)
                video = gr.Video(label="Video", autoplay=True)

    view_inputs = [state, mode, cam_name, path_pos, t, scale, bg_name]
    load_btn.click(
        load_checkpoint,
        [source, zip_file, repo_id, subfolder, repo_type, camera_files, config, prefer_cca],
        [state, status, cam_name, frame_btn, video_btn],
    ).success(render_frame, view_inputs, image)
    frame_btn.click(render_frame, view_inputs, image)
    for slider in (t, path_pos):
        slider.release(render_frame_if_loaded, view_inputs, image)
    video_btn.click(render_video, view_inputs + [n_frames, fps, motion, sweep_time], video)


if __name__ == "__main__":
    demo.queue(max_size=20).launch(allowed_paths=[str(rt.WORK_ROOT)])
