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
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, bwd_dynamic = False, dyn_mask2= None, 
             override_color = None, stage="coarse", cam_type=None, compute_smooth=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix = viewpoint_camera.world_view_transform.cuda(),
            projmatrix = viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            bwd_depth= False,
            bwd_dynamic = bwd_dynamic,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features
    dynamics = pc.get_dynamic
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    reg_smooth_value = None

    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:

        dyn_mask = (pc.get_dynamic > pc.dynamic_thro).squeeze(1)
        means3D_final = means3D.clone()
        scales_final = scales.clone() if scales is not None else None
        rotations_final = rotations.clone() if rotations is not None else None
        opacity_final = opacity.clone()
        shs_final = shs.clone()

        if dyn_mask.any():
            time_dyn = torch.full((dyn_mask.sum().item(), 1), float(viewpoint_camera.time), dtype=means3D.dtype, device=means3D.device)
            dyn_xyz, dyn_scales, dyn_rots, dyn_opacity, dyn_shs, reg_val = pc._deformation(
                means3D[dyn_mask],
                scales[dyn_mask] if scales is not None else None,
                rotations[dyn_mask] if rotations is not None else None,
                opacity[dyn_mask],
                shs[dyn_mask],
                time_dyn,
                compute_smooth=compute_smooth
            )
            if compute_smooth and (reg_val is not None):
                reg_smooth_value = reg_val if reg_smooth_value is None else reg_smooth_value + reg_val
            means3D_final[dyn_mask] = dyn_xyz
            if dyn_scales is not None and scales_final is not None:
                scales_final[dyn_mask] = dyn_scales
            if dyn_rots is not None and rotations_final is not None:
                rotations_final[dyn_mask] = dyn_rots
            if dyn_opacity is not None:
                opacity_final[dyn_mask] = dyn_opacity
            if dyn_shs is not None:
                shs_final[dyn_mask] = dyn_shs

    
    else:
        raise NotImplementedError


    if scales_final is not None:
        scales_final = pc.scaling_activation(scales_final)  # exp
    if rotations_final is not None:
        rotations_final = pc.rotation_activation(rotations_final)  #  normalize
    opacity = pc.opacity_activation(opacity_final)  # sigmoid
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    if  dyn_mask2 !=None:  # our test code , not important
        rendered_image, radii, depth, dynamics_map, max_weight_t = rasterizer(
            means3D = means3D_final[dyn_mask2],
            means2D = means2D[dyn_mask2],
            shs = shs_final[dyn_mask2],
            colors_precomp = colors_precomp,
            opacities = opacity[dyn_mask2],
            scales = scales_final[dyn_mask2],
            rotations = rotations_final[dyn_mask2],
            dynamics = dynamics[dyn_mask2],
            cov3D_precomp = cov3D_precomp
            )
    else:
        rendered_image, radii, depth, dynamics_map, max_weight_t = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            dynamics = dynamics,
            cov3D_precomp = cov3D_precomp
        )
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "dynamic_map": dynamics_map,
            "max_weight_t":max_weight_t,
            "reg_smooth": reg_smooth_value
            }
