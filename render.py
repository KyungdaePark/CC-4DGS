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
import imageio
import numpy as np
import random
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams, OptimizationParams
from gaussian_renderer import GaussianModel
from utils.system_utils import searchForMaxIteration, find_best_iteration_from_log
from time import time
import threading
import concurrent.futures
import torch.cuda.nvtx as nvtx


def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def export_dynamic_static_ply(gaussians, dataset, scene):
    dyn_mask = (gaussians.get_dynamic > gaussians.dynamic_thro).squeeze(-1)
    dynamic_mask_np = dyn_mask.detach().cpu().numpy()
    static_mask_np = np.logical_not(dynamic_mask_np)

    if scene.loaded_iter is not None:
        base_dir_candidates = [
            os.path.join(dataset.model_path, "point_cloud", f"iteration_{scene.loaded_iter}"),
            os.path.join(dataset.model_path, "point_cloud", f"coarse_iteration_{scene.loaded_iter}"),
            os.path.join(dataset.model_path, "point_cloud"),
        ]
        suffix = f"iter{scene.loaded_iter}"
    else:
        base_dir_candidates = [os.path.join(dataset.model_path, "point_cloud"), dataset.model_path]
        suffix = "current"

    base_dir = None
    for candidate in base_dir_candidates:
        if os.path.isdir(candidate):
            base_dir = candidate
            break
    if base_dir is None:
        base_dir = base_dir_candidates[0]

    split_dir = os.path.join(base_dir, "dynamic_static_split")
    os.makedirs(split_dir, exist_ok=True)

    dynamic_path = os.path.join(split_dir, f"dynamic_points_{suffix}.ply")
    static_path = os.path.join(split_dir, f"static_points_{suffix}.ply")

    gaussians.save_ply(dynamic_path, mask=dynamic_mask_np)
    gaussians.save_ply(static_path, mask=static_mask_np)

    print(f"[render] Saved dynamic Gaussians to {dynamic_path}")
    print(f"[render] Saved static Gaussians to {static_path}")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, scene : Scene, selection_mask=None, output_suffix=None, render_args=None):
    print(f'Rendering set {name}')
    
    renders_dirname = "renders"
    if output_suffix:
        renders_dirname = f"{renders_dirname}_{output_suffix}"
    method_name = f"ours_best_{iteration}"
    if render_args and getattr(render_args, "compressed", False):
        method_name = f"{method_name}_compressed"
    render_path = os.path.join(model_path, name, method_name, renders_dirname)
    gts_path = os.path.join(model_path, name, method_name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print(f"[render] Number of Gaussians: {gaussians._xyz.shape[0]}")
    
    total_time = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        time1 = time()
        with nvtx.range("render_scene"):
            rendering = render(view, gaussians, pipeline, background, stage="fine", cam_type=cam_type, compute_smooth=False, dyn_mask2=selection_mask)
            torch.cuda.synchronize()  # Wait for GPU work before recording elapsed time.
        total_time += time() - time1
        
        rendering_images = rendering["render"]
        render_images.append(to8b(rendering_images).transpose(1,2,0))
        render_list.append(rendering_images)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)
    print(f'total_time : {total_time}')
    fps = len(views) / total_time
    print(f'fps : {fps}')

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)
    
    video_name = 'video_rgb.mp4' if not output_suffix else f'video_rgb_{output_suffix}.mp4'
    imageio.mimwrite(os.path.join(model_path, name, method_name, video_name), render_images, fps=30)

    
def render_sets(dataset : ModelParams, hyperparam, opt, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, render_args=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam, opt, dataset.model_path)
        use_compressed = bool(render_args and getattr(render_args, "compressed", False))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, use_compressed=use_compressed)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        selection_mask = None
        output_suffix = None
        if render_args:
            if getattr(render_args, "dynamic_only", False) and getattr(render_args, "static_only", False):
                raise ValueError("Both dynamic_only and static_only flags are set. This should have been caught earlier.")
            if getattr(render_args, "dynamic_only", False):
                selection_mask = (gaussians.get_dynamic > gaussians.dynamic_thro).squeeze(-1)
                selected = selection_mask.sum().item()
                total = selection_mask.shape[0]
                print(f"[render] Using dynamic-only mask: {selected} / {total} points")
                if selected == 0:
                    print("[render] Warning: dynamic-only mask removed all Gaussians.")
                output_suffix = "dynamic"
            elif getattr(render_args, "static_only", False):
                selection_mask = (gaussians.get_dynamic <= gaussians.dynamic_thro).squeeze(-1)
                selected = selection_mask.sum().item()
                total = selection_mask.shape[0]
                print(f"[render] Using static-only mask: {selected} / {total} points")
                if selected == 0:
                    print("[render] Warning: static-only mask removed all Gaussians.")
                output_suffix = "static"
        if render_args and getattr(render_args, "save_dynamic_static_ply", False):
            export_dynamic_static_ply(gaussians, dataset, scene)
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
                scene,
                selection_mask=selection_mask,
                output_suffix=output_suffix,
                render_args=render_args,
            )
        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
                scene,
                selection_mask=selection_mask,
                output_suffix=output_suffix,
                render_args=render_args,
            )
        if not skip_video:
            render_set(
                dataset.model_path,
                "video",
                scene.loaded_iter,
                scene.getVideoCameras(),
                gaussians,
                pipeline,
                background,
                cam_type,
                scene,
                selection_mask=selection_mask,
                output_suffix=output_suffix,
                render_args=render_args,
            )

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = True
            
if __name__ == "__main__":
    # Set up command line argument parser
    # parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    # pipeline = PipelineParams(parser)
    # hyperparam = ModelHiddenParams(parser)
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--skip_train", default=True, action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Skip rendering the dataset's interpolated video-camera trajectory.",
    )
    parser.add_argument("--configs", type=str)
    parser.add_argument("--compressed", action="store_true",
                        help="Load Gaussians from point_cloud/iteration_XXX/compressed instead of a PLY.")
    parser.add_argument("--dynamic_only", action="store_true",
                        help="Render only Gaussians classified as dynamic (dynamic score > threshold).")
    parser.add_argument("--static_only", action="store_true",
                        help="Render only Gaussians classified as static (dynamic score <= threshold).")
    parser.add_argument("--save_dynamic_static_ply", action="store_true",
                        help="Save dynamic/static Gaussian subsets as separate PLY files during rendering.")
    

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    if getattr(args, "dynamic_only", False) and getattr(args, "static_only", False):
        parser.error("Cannot use --dynamic_only and --static_only simultaneously.")
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    extracted_lp = lp.extract(args)
    extracted_hp = hp.extract(args)
    extracted_op = op.extract(args)
    extracted_pp = pp.extract(args)
    

    extracted_op.lambda_smooth = 0.0
    
    extracted_hp.smooth_cfg['smooth'] = False
    extracted_hp.h2m_render_mode = True

    
    print(f'rendering , {extracted_lp.model_path}')
    render_sets(extracted_lp, extracted_hp, extracted_op, args.iteration, 
                extracted_pp, args.skip_train, args.skip_test, 
                args.skip_video, render_args=args)
