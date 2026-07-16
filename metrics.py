#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from typing import Optional
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    # Ensure deterministic, numeric ordering: 00012.png -> 12
    def _num_key(fname):
        stem = os.path.splitext(fname)[0]
        try:
            return int(stem)
        except Exception:
            return stem
    fnames = sorted(os.listdir(renders_dir), key=_num_key)

    for fname in fnames:
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def iter_image_pairs(renders_dir, gt_dir):
    def _num_key(fname):
        stem = os.path.splitext(fname)[0]
        try:
            return int(stem)
        except Exception:
            return stem

    for fname in sorted(os.listdir(renders_dir), key=_num_key):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        render_t = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt_t = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        yield fname, render_t, gt_t

def evaluate(model_paths, method_filter: Optional[str] = None, *, compressed_only: bool = False):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            methods = sorted(os.listdir(test_dir))
            for method in methods:
                if method == ".DS_Store":
                    continue
                if compressed_only and not method.endswith("_compressed"):
                    continue
                if method_filter and method_filter not in method:
                    continue
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                image_names = []
                # Prepare output directories for diff maps and plots
                diff_dir = method_dir / "diff_maps"
                plots_dir = method_dir / "plots"
                os.makedirs(diff_dir, exist_ok=True)
                os.makedirs(plots_dir, exist_ok=True)

                ssims = []
                psnrs = []
                # lpipss = []
                # lpipsa = []
                # ms_ssims = []
                # Dssims = []
                frame_indices = []
                num_images = len(os.listdir(renders_dir))
                for idx, (image_name, render, gt) in enumerate(tqdm(iter_image_pairs(renders_dir, gt_dir), total=num_images, desc="Metric evaluation progress")):
                    image_names.append(image_name)
                    # Primary metrics on float tensors
                    ssim_val = ssim(render, gt)
                    psnr_val = psnr(render, gt)
                    ssims.append(ssim_val.item())
                    psnrs.append(psnr_val.item())
                    # lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    # ms_ssims.append(ms_ssim(renders[idx], gts[idx],data_range=1, size_average=True ))
                    # lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                    # Dssims.append((1-ms_ssims[-1])/2)
                    # Diff map save (grayscale heatmap)
                    try:
                        with torch.no_grad():
                            diff = (render - gt).abs().mean(dim=1, keepdim=True)  # [1,1,H,W]
                            maxv = diff.max().clamp_min(1e-8)
                            diff_norm = (diff / maxv).squeeze().detach().cpu().numpy()
                            plt.imsave(diff_dir / image_name, diff_norm, cmap='magma')
                    except Exception:
                        pass
                    # Parse frame index from filename (e.g., '00012.png' -> 12)
                    try:
                        stem = os.path.splitext(image_name)[0]
                        frame_indices.append(int(stem))
                    except Exception:
                        frame_indices.append(idx)
                    del render, gt, ssim_val, psnr_val
                    if idx % 50 == 0:
                        torch.cuda.empty_cache()

                print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(float(np.mean(ssims)), ".5"))
                print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(float(np.mean(psnrs)), ".5"))
                # print("Scene: ", scene_dir,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                # print("Scene: ", scene_dir,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
                # print("Scene: ", scene_dir,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
                # print("Scene: ", scene_dir,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

                full_dict[scene_dir][method].update({
                                                        "SSIM": float(np.mean(ssims)),
                                                        "PSNR": float(np.mean(psnrs)),
                                                        }
                                                        # "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                                        # "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                        # "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                        # "D-SSIM": torch.tensor(Dssims).mean().item()},

                                                    )
                per_view_dict[scene_dir][method].update({
                                                            "SSIM": {name: ssim for ssim, name in zip(ssims, image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(psnrs, image_names)},
                                                            # "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            # "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                                            # "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                                            # "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},

                                                            }
                                                        )

                # Save per-method metric curves and plots
                try:
                    frames_np = [int(f) for f in frame_indices]
                    psnrs_np = list(psnrs)
                    ssims_np = list(ssims)
                    # Sort all series by frame index to avoid zig-zag plots
                    ordered = sorted(zip(frames_np, psnrs_np, ssims_np), key=lambda x: x[0])
                    frames_np, psnrs_np, ssims_np = [list(t) for t in zip(*ordered)]
                    curves = {
                        "frames": frames_np,
                        "PSNR": psnrs_np,
                        "SSIM": ssims_np,
                    }
                    with open(method_dir / "metrics_curve.json", 'w') as fp:
                        json.dump(curves, fp, indent=True)
                    # PSNR curve
                    plt.figure(figsize=(7, 3))
                    plt.plot(frames_np, psnrs_np, label='PSNR (dB)')
                    plt.xlabel('frame')
                    plt.ylabel('PSNR (dB)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'psnr_curve.png', dpi=150)
                    plt.close()
                    # SSIM curve
                    plt.figure(figsize=(7, 3))
                    plt.plot(frames_np, ssims_np, label='SSIM')
                    plt.xlabel('frame')
                    plt.ylabel('SSIM')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'ssim_curve.png', dpi=150)
                    plt.close()
                except Exception:
                    pass

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            
            print("Unable to compute metrics for model", scene_dir)
            raise e

def _add_interactive_tools():
    """No-op placeholder to keep source order tidy."""
    return None

def _interactive_view_for_method(method_dir: Path, title_prefix: str = ""):
    curves_path = method_dir / "metrics_curve.json"
    diff_dir = method_dir / "diff_maps"
    gt_dir = method_dir / "gt"
    renders_dir = method_dir / "renders"
    if not curves_path.exists():
        print(f"[interactive] skip (no metrics_curve.json): {method_dir}")
        return
    with open(curves_path, 'r') as fp:
        curves = json.load(fp)
    frames = curves.get('frames', [])
    psnrs = curves.get('PSNR', [])
    ssims = curves.get('SSIM', [])
    if len(frames)==0 or len(psnrs)==0:
        print(f"[interactive] empty curves: {method_dir}")
        return
    # sort by frame index
    ordered = sorted(zip(frames, psnrs, ssims if len(ssims)==len(frames) else [None]*len(frames)), key=lambda x: x[0])
    frames = [int(x[0]) for x in ordered]
    psnrs = [float(x[1]) for x in ordered]
    ssims = [float(x[2]) if x[2] is not None else None for x in ordered]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{title_prefix} {method_dir.name}")
    ax_psnr, ax_ssim = axes
    ax_psnr.plot(frames, psnrs, '-', lw=1.2)
    ax_psnr.set_ylabel('PSNR (dB)')
    if all(v is not None for v in ssims):
        ax_ssim.plot(frames, ssims, '-', lw=1.2)
    else:
        ax_ssim.text(0.5, 0.5, 'SSIM not available', ha='center', va='center', transform=ax_ssim.transAxes)
    ax_ssim.set_xlabel('frame')
    ax_ssim.set_ylabel('SSIM')

    # Inset for diff map
    box = fig.add_axes([0.72, 0.55, 0.25, 0.35])
    box.set_xticks([]); box.set_yticks([])

    def fmt_frame(n):
        try:
            return f"{int(n):05d}.png"
        except Exception:
            return str(n)

    def load_img(p: Path):
        try:
            return np.asarray(Image.open(p))
        except Exception:
            return None

    def update_inset(idx):
        box.cla(); box.set_xticks([]); box.set_yticks([])
        fname = fmt_frame(frames[idx])
        dm = load_img(diff_dir / fname)
        if dm is not None:
            box.imshow(dm)
            ssim_txt = (f", SSIM={ssims[idx]:.3f}" if ssims[idx] is not None else "")
            box.set_title(f"f={frames[idx]} | PSNR={psnrs[idx]:.2f}{ssim_txt}", fontsize=9)
        else:
            box.text(0.5,0.5, f"No diff {fname}", ha='center', va='center', transform=box.transAxes)
        fig.canvas.draw_idle()

    update_inset(0)

    def on_move(event):
        if event.inaxes not in (ax_psnr, ax_ssim):
            return
        if event.xdata is None:
            return
        x = float(event.xdata)
        idx = int(np.argmin(np.abs(np.asarray(frames) - x)))
        update_inset(idx)

    def on_click(event):
        if event.inaxes not in (ax_psnr, ax_ssim):
            return
        if event.xdata is None:
            return
        x = float(event.xdata)
        idx = int(np.argmin(np.abs(np.asarray(frames) - x)))
        fname = fmt_frame(frames[idx])
        fig2, axs = plt.subplots(1,3, figsize=(12,4))
        for a in axs: a.set_xticks([]); a.set_yticks([])
        gt = load_img(gt_dir / fname); rd = load_img(renders_dir / fname); dm = load_img(diff_dir / fname)
        if gt is not None: axs[0].imshow(gt); axs[0].set_title('GT')
        else: axs[0].text(0.5,0.5,'GT missing', ha='center', va='center', transform=axs[0].transAxes)
        if rd is not None: axs[1].imshow(rd); axs[1].set_title('Render')
        else: axs[1].text(0.5,0.5,'Render missing', ha='center', va='center', transform=axs[1].transAxes)
        if dm is not None: axs[2].imshow(dm); axs[2].set_title('Diff')
        else: axs[2].text(0.5,0.5,'Diff missing', ha='center', va='center', transform=axs[2].transAxes)
        fig2.suptitle(f"frame {frames[idx]}")
        fig2.tight_layout(); fig2.show()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.tight_layout()
    plt.show()


def interactive_only(model_paths, method_filter=None, *, compressed_only: bool = False):
    for scene_dir in model_paths:
        test_dir = Path(scene_dir) / 'test'
        if not test_dir.exists():
            print(f"[interactive] test dir missing: {test_dir}")
            continue
        for m in sorted(os.listdir(test_dir)):
            if (method_filter is not None) and (method_filter not in m):
                continue
            if compressed_only and not m.endswith("_compressed"):
                continue
            mdir = test_dir / m
            if (mdir / 'metrics_curve.json').exists():
                _interactive_view_for_method(mdir, title_prefix=str(Path(scene_dir).name))
            else:
                print(f"[interactive] skip(no curves): {mdir}")


def _build_html_for_method(method_dir: Path):
    curves_path = method_dir / 'metrics_curve.json'
    if not curves_path.exists():
        print(f"[html] skip(no curves): {method_dir}")
        return
    with open(curves_path, 'r') as fp:
        curves = json.load(fp)
    frames = curves.get('frames', [])
    psnrs = curves.get('PSNR', [])
    ssims = curves.get('SSIM', [])
    html_path = method_dir / 'interactive.html'
    diff_dir = 'diff_maps'
    gt_dir = 'gt'
    rd_dir = 'renders'
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Interactive Metrics - {method_dir.name}</title>
  <script src=\"https://cdn.plot.ly/plotly-2.26.0.min.js\"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 8px; }}
    .row {{ display:flex; gap:8px; }}
    .panel {{ border:1px solid #ccc; padding:4px; }}
    img {{ max-width:100%; height:auto; }}
  </style>
</head>
<body>
  <h3>{method_dir.parent.parent.name} / {method_dir.name}</h3>
  <div id=\"plot\" class=\"panel\" style=\"width:100%;height:420px;\"></div>
  <div class=\"row\">
    <div class=\"panel\" style=\"width:33%\"><h4>GT</h4><img id=\"gt\"/></div>
    <div class=\"panel\" style=\"width:33%\"><h4>Render</h4><img id=\"rd\"/></div>
    <div class=\"panel\" style=\"width:33%\"><h4>Diff</h4><img id=\"df\"/></div>
  </div>
  <script>
    const frames = {frames};
    const psnrs = {psnrs};
    const ssims = {ssims if ssims else 'null'};
    const plotDiv = document.getElementById('plot');
    const gt = document.getElementById('gt');
    const rd = document.getElementById('rd');
    const df = document.getElementById('df');
    function fmt(n) {{ return String(n).padStart(5,'0') + '.png'; }}
    function updateImages(idx) {{
      const f = fmt(frames[idx]);
      gt.src = '{gt_dir}/' + f;
      rd.src = '{rd_dir}/' + f;
      df.src = '{diff_dir}/' + f;
    }}
    const data = [
      {{ x: frames, y: psnrs, name:'PSNR', mode:'lines', line:{{width:1}} }},
    ];
    if (ssims) {{ data.push({{ x: frames, y: ssims, yaxis:'y2', name:'SSIM', mode:'lines', line:{{width:1}} }}); }}
    Plotly.newPlot(plotDiv, data, {{
      margin: {{t:20,r:50,l:50,b:40}},
      xaxis: {{ title:'frame' }},
      yaxis: {{ title:'PSNR (dB)' }},
      yaxis2: {{ title:'SSIM', overlaying:'y', side:'right' }}
    }});
    plotDiv.on('plotly_hover', (ev) => {{ if (ev.points&&ev.points.length) updateImages(ev.points[0].pointIndex); }});
    plotDiv.on('plotly_click', (ev) => {{ if (ev.points&&ev.points.length) updateImages(ev.points[0].pointIndex); }});
    updateImages(0);
  </script>
  <p style=\"color:#666\">Hover to preview; click to lock. Images load via relative paths.</p>
</body>
</html>
"""
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"[html] wrote {html_path}")


def build_html(model_paths, method_filter=None, *, compressed_only: bool = False):
    for scene_dir in model_paths:
        test_dir = Path(scene_dir) / 'test'
        if not test_dir.exists():
            print(f"[html] no test dir: {test_dir}")
            continue
        for m in sorted(os.listdir(test_dir)):
            if (method_filter is not None) and (method_filter not in m):
                continue
            if compressed_only and not m.endswith("_compressed"):
                continue
            _build_html_for_method(test_dir / m)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Metric evaluation / interactive viewer")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--interactive_only', action='store_true', help='Open interactive PSNR/SSIM viewer from saved curves without recomputing metrics')
    parser.add_argument('--method', type=str, default=None, help='Substring to filter methods (e.g., ours_best_24000)')
    parser.add_argument('--iteration', type=int, default=None, help='Evaluate only ours_best_<iteration>.')
    parser.add_argument('--compressed', action='store_true', help='Reserved for parity with render.py; renders should already reflect compressed assets.')
    parser.add_argument('--html', action='store_true', help='Write interactive HTML pages to each method dir (interactive.html)')
    parser.add_argument('--no_compute', action='store_true', help='Skip metric recompute; use saved curves')
    args = parser.parse_args()

    method_filter = args.method
    if args.iteration is not None:
        iter_token = f"ours_best_{args.iteration}"
        if args.compressed:
            iter_token = f"{iter_token}_compressed"
        method_filter = method_filter or iter_token
    compressed_only = bool(args.compressed)
    if compressed_only:
        print("[metrics] Using renders generated from compressed assets (flag acknowledged).")

    if args.interactive_only:
        interactive_only(args.model_paths, method_filter=method_filter, compressed_only=compressed_only)
    else:
        if not args.no_compute:
            evaluate(args.model_paths, method_filter=method_filter, compressed_only=compressed_only)
        if args.html:
            build_html(args.model_paths, method_filter=method_filter, compressed_only=compressed_only)
        else:
            interactive_only(args.model_paths, method_filter=method_filter, compressed_only=compressed_only)
