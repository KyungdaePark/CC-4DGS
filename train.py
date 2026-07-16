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
import numpy as np
import random
import os, sys


import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
from scene.dataset_readers import is_technicolor_scene
import copy
from PIL import Image
import matplotlib.pyplot as plt

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
 
 
 
 
def get_lambda_smooth_weight(opt, iteration):
    base = getattr(opt, "lambda_smooth", 0.0)
    start = getattr(opt, "lambda_smooth_start_iter", 0)
    end = getattr(opt, "lambda_smooth_end_iter", 0)
    init = getattr(opt, "lambda_smooth_init", base)
    final = getattr(opt, "lambda_smooth_final", base)

    if end > start:
        if iteration <= start:
            return init
        if iteration >= end:
            return final
        alpha = (iteration - start) / float(end - start)
        return init + alpha * (final - init)
    return base


def train_dynamic(pipe ,opt, scene,  gaussians, tb_writer, args):

    gaussians.training_setup(opt, args)
    dy_iteration = 3000
    background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras()
    temp_list = [cam for cam in get_stamp_list(viewpoint_stack,0) if getattr(cam, "dyn_image", None) is not None]
    if not temp_list:
        print("[INFO] No dynamic ground-truth available; skipping dynamic training stage.")
        return
    viewpoint_stack = temp_list.copy()
    loss = torch.tensor(0.).cuda()
    progress_bar = tqdm(range(1, dy_iteration), desc="Dynamic Training progress")
    # train dynamic 
    for iteration in range(1,  dy_iteration + 1):

        gaussians.update_learning_dy_rate(iteration) 
        loss = torch.tensor(0.).cuda()

        if not viewpoint_stack:
            viewpoint_stack =  temp_list.copy()
  
        viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))

        dyn_image = getattr(viewpoint_cam, "dyn_image", None)
        if dyn_image is None:
            print("[WARN] Encountered camera without dynamic mask; skipping this sample.")
            if viewpoint_stack:
                continue
            progress_bar.close()
            return
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, bwd_dynamic=True, compute_smooth= False)
        dynamic_map = render_pkg["dynamic_map"]

        dyn_image = dyn_image.cuda()
        loss = scene.criterion(dynamic_map, dyn_image)


        loss.backward()



        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
                progress_bar.update(10)

            if iteration == dy_iteration:
                progress_bar.close()

            if iteration < dy_iteration:
                gaussians.dy_optimizer.step()
                gaussians.dy_optimizer.zero_grad(set_to_none = True)

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer, args):
    first_iter = 0

    gaussians.training_setup(opt, args)
    import re
    if checkpoint:
        # Support both file checkpoints (gaussians.capture tuples)
        # and directory checkpoints like point_cloud/{coarse,fine}_iteration_XXXX
        try:
            if os.path.isdir(checkpoint):
                print(f"[CKPT] Loading from directory: {checkpoint}")
                ply_path = os.path.join(checkpoint, "point_cloud.ply")
                if not os.path.exists(ply_path):
                    raise FileNotFoundError(f"Missing point_cloud.ply in {checkpoint}")
                gaussians.load_ply(ply_path)
                gaussians.load_model(checkpoint)
                # Try to infer iteration number from directory name suffix
                try:
                    base = os.path.basename(os.path.normpath(checkpoint))
                    m = re.search(r"_(\d+)$", base)
                    if m is not None:
                        first_iter = int(m.group(1))
                        print(f"[CKPT] Inferred iteration={first_iter} from '{base}'")
                except Exception:
                    pass
                # If we are in coarse stage but given a pre-trained dir, skip training coarse
                if stage == "coarse":
                    print("start from fine stage using loaded coarse checkpoint; skip coarse stage.")
                    return
            else:
                # Fallback to legacy .pth checkpoint
                if stage == "coarse" and stage not in checkpoint:
                    print("start from fine stage, skip coarse stage.")
                    # process is in the coarse stage, but start from fine stage
                    return
                if stage in checkpoint:
                    print(f'stage {stage} in checkpoint {checkpoint}.')
                    (model_params, first_iter) = torch.load(checkpoint)
                    gaussians.restore(model_params, opt)
        except Exception as e:
            print(f"[CKPT][WARN] Failed to load start_checkpoint '{checkpoint}': {e}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    viewpoint_stack = None

    final_iter = train_iter
    

    first_iter += 1
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()


    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=4,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0) # 20
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")


    best_mpsnr = -1
    best_iter = -1
    log_file_path = os.path.join(scene.model_path, "log_metrics.txt")
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch 
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=4,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        last_max_weight_list = []

        reg_smooth_total = None
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type, compute_smooth=hyper.compute_smooth)
            image, viewspace_point_tensor, visibility_filter, radii, max_weight_t = \
                        render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"],render_pkg["max_weight_t"]
            reg_val = render_pkg.get("reg_smooth", None)
            if reg_val is not None:
                reg_smooth_total = reg_val if reg_smooth_total is None else (reg_smooth_total + reg_val)
            images.append(image.unsqueeze(0))

            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            last_max_weight_list.append(max_weight_t)
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        last_max_weight = torch.cat(last_max_weight_list,0).max(dim=0).values
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Loss
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:]) * (1.0 - opt.lambda_dssim)

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        
        loss = Ll1
        ssim_term = torch.tensor(0.0, device=loss.device)
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            ssim_term = opt.lambda_dssim * (1.0-ssim_loss)
            loss += ssim_term

        lambda_smooth_weight = get_lambda_smooth_weight(opt, iteration)
        reg_smooth = None
        if reg_smooth_total is not None:
            reg_smooth = reg_smooth_total / float(len(viewpoint_cams))
            if lambda_smooth_weight > 0.0:
                loss = loss + lambda_smooth_weight * reg_smooth
        else:
            reg_smooth = torch.tensor(0.0, device=loss.device)
        
        loss.backward()
        if torch.isnan(loss).any():
            print("[WARN] Loss became NaN; restarting the training process.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        # iter_end.record()

        with torch.no_grad():
            # Progress bar
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "loss_total": f"{loss.item():.{7}f}",
                    "loss_l1": f"{Ll1.item():.6f}",
                    "loss_dssim": f"{ssim_term.item():.6f}",
                    "loss_reg": f"{(lambda_smooth_weight * reg_smooth).item():.6f}" if lambda_smooth_weight > 0 else "0.0",
                    "psnr": f"{psnr_:.{2}f}",
                    "point": f"{total_point}"
                })
                
                progress_bar.update(10)
            if iteration == final_iter - 1:
                progress_bar.close()

            # Log and save
            timer.pause()
            if  iteration == final_iter - 1 :
                print("The num of dynamic point: {}".format(torch.sum(gaussians.get_dynamic > gaussians.dynamic_thro)))
            # if stage == "fine" : #  or  stage=="coarse":
                # training_report(args, tb_writer, iteration, Ll1, loss, l1_loss,  testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)

            if iteration in testing_iterations and stage=="fine":
                torch.cuda.empty_cache()
                # 
                validation_configs = (
                    # {'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                    {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras())) ] },  # test all
                    # {'name': 'test', 'cameras' : [scene.getTestCameras()[0]  ] },
                    {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]}
                )

                for config in validation_configs:
                    if config['cameras'] and len(config['cameras']) > 0:
                        l1_test = 0.0
                        psnr_test = 0.0
                        for idx, viewpoint in enumerate(config['cameras']):
                            image = torch.clamp(render(viewpoint, scene.gaussians,stage=stage, cam_type=scene.dataset_type, *[pipe, background])["render"], 0.0, 1.0)
                            if scene.dataset_type == "PanopticSports":
                                gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                            else:
                                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                            l1_test += l1_loss(image, gt_image).mean().double()
                            # mask=viewpoint.mask
                            psnr_ = psnr(image, gt_image, mask=None).mean().double()                    
                            psnr_test += psnr_

                        psnr_test /= len(config['cameras'])
                        l1_test /= len(config['cameras'])          
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}, cameras num {}".format(iteration, config['name'], l1_test, psnr_test, len(config['cameras'])))
                        
                        if config['name'] == 'test':
                            with open(log_file_path, mode='a') as f:
                                f.write(f"{iteration}, {psnr_test.item():.4f}\n")
                        if psnr_test.item() > best_mpsnr:
                            best_mpsnr = psnr_test.item()
                            best_iter = iteration
                            # best_ckpt_path = os.path.join(scene.model_path, "chkpnt_best.pth")
                            # shutil.copy2(checkpoint_path, best_ckpt_path)
                            print(f"New Best mPSNR {psnr_test.item():.3f} at iter={best_iter}. Saved best model.\n")
                        
                        if tb_writer:
                            tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                            tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                if tb_writer:
                    tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
                    tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)

                torch.cuda.empty_cache()



            if (iteration in saving_iterations and stage == "fine"):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                gaussians._max_weight = torch.max(gaussians._max_weight, last_max_weight)

                if iteration >= opt.weight_prune_from_iter and iteration % opt.weight_prune_interval == 0 : 
                    gaussians.weight_prune(opt.weight_prune_threshold)
                
                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0: # and gaussians.get_xyz.shape[0]<360000:
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent)

                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 : # and gaussians.get_xyz.shape[0] > 200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)


                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()

                if iteration % 3000 == 0  and stage == "coarse": 
                    print("reset opacity")
                    gaussians.reset_opacity()
    
            # Optimizer step
            if iteration < final_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if gaussians.hash_optimizer is not None:
                    gaussians.hash_optimizer.step()  # It is only updated during the fine stage.
                    gaussians.hash_optimizer.zero_grad(set_to_none = True)


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname,args):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    dataset.model_path = args.model_path
    gaussians = GaussianModel(dataset.sh_degree, hyper, opt, dataset.model_path)
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()

    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                              checkpoint_iterations, checkpoint, debug_from,
                              gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer,args)
    if opt.train_dynamic:
        train_dynamic ( pipe , opt,  scene,  gaussians , tb_writer ,args)
    
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer, args)

def prepare_output_and_logger(expname):    
    
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# def training_report(args , tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):

   
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 3000, 5000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 3000, 5000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")


    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    technicolor_sparse = os.path.join(args.source_path, "colmap", "dense", "workspace", "sparse")
    if is_technicolor_scene(args.source_path) and os.path.exists(technicolor_sparse):
        if getattr(args, "resolution", -1) == -1:
            print("[INFO] Detected Technicolor dataset. Setting resolution=1.")
            args.resolution = 1

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args)

    # All done
    print("\nTraining complete.")
