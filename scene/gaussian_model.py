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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from typing import Dict
# import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args, opt, model_path):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._deformation = deform_network(args, model_path)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._dynamic = torch.empty(0)
        self.hash_optimizer = None
        self.hash_scheduler_args = {}
        self._xyz_bound_min = None
        self._xyz_bound_max = None
        self._max_weight = torch.empty(0)
        self.dynamic_thro = opt.dynamic_thro
        self.activation = opt.activation
        self.n_neurons = opt.n_neurons

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        deform_state,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_dynamic(self):
        return self._dynamic
    
    @property
    def get_max_weight(self):
        return self._max_weight
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"[model] Number of points at initialization: {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._dynamic = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._max_weight = torch.zeros((self._xyz.shape[0]), device="cuda")  # Maximum contribution per Gaussian.

        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation = self._deformation.to("cuda")


    def get_xyz_bound(self, percentile = 100.0):
        with torch.no_grad():
            half_percentile = (100 - percentile) / 200
            self._xyz_bound_min = torch.quantile(self._xyz , half_percentile,dim=0)  # 86.6 % ?
            self._xyz_bound_max = torch.quantile(self._xyz , 1 - half_percentile,dim=0)
            return self._xyz_bound_min, self._xyz_bound_max
        
    def training_setup(self, training_args, args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init= training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final= training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
         


        deform_param_groups = {"default": [], "opacity": []}
        grid_params = []
        if hasattr(self._deformation, "get_mlp_parameter_groups"):
            deform_param_groups = self._deformation.get_mlp_parameter_groups()
        elif hasattr(self._deformation, "get_mlp_parameters"):
            deform_param_groups["default"] = list(self._deformation.get_mlp_parameters())
        if hasattr(self._deformation, "get_grid_parameters"):
            grid_params = list(self._deformation.get_grid_parameters())

        hash_param_groups = []
        if deform_param_groups.get("default"):
            hash_param_groups.append({'params': deform_param_groups["default"], 'lr': 0.0, "name": "deformation"})
        if grid_params:
            hash_param_groups.append({'params': grid_params, 'lr': 0.0, "name": "grid"})

        if hash_param_groups:
            self.hash_optimizer = torch.optim.Adam(hash_param_groups, lr=0.0, eps=1e-15)
            self.hash_scheduler_args = {}
            if deform_param_groups.get("default"):
                self.hash_scheduler_args["deformation"] = get_expon_lr_func(
                    lr_init=training_args.deformation_lr_init * self.spatial_lr_scale,
                    lr_final=training_args.deformation_lr_final * self.spatial_lr_scale,
                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                    max_steps=training_args.position_lr_max_steps,
                )
            if grid_params:
                self.hash_scheduler_args["grid"] = get_expon_lr_func(
                    lr_init=training_args.grid_lr_init * self.spatial_lr_scale,
                    lr_final=training_args.grid_lr_final * self.spatial_lr_scale,
                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                    max_steps=training_args.position_lr_max_steps,
                )
        else:
            self.hash_optimizer = None
            self.hash_scheduler_args = {}
        
        dynamics_l = [
             {'params': [self._dynamic], 'lr': 0.05, "name": "dynamic"},
        ]
        self.dy_optimizer =  torch.optim.Adam(dynamics_l, lr=0.0, eps=1e-15)
        self.dy_scheduler_args = get_expon_lr_func(lr_init= 0.05, lr_final=0.005,
                                                    lr_delay_mult=training_args.position_lr_delay_mult, max_steps=30000) 

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

        if self.hash_optimizer is not None:
            for param_group in self.hash_optimizer.param_groups:
                name = param_group.get("name", None)
                scheduler = self.hash_scheduler_args.get(name, None)
                if scheduler is not None:
                    param_group['lr'] = scheduler(iteration)

    def update_learning_dy_rate(self, iteration):
        for param_group in self.dy_optimizer.param_groups:
            if param_group["name"] == "dynamic":
                lr = self.dy_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('dynamic')
        return l
    def compute_deformation(self,time):
        return self._xyz
    # def save_ply_dynamic(path):
    #     for time in range(self._deformation.shape(-1)):
    #         xyz = self.compute_deformation(time)
    def load_model(self, path):
        deform_path = os.path.join(path, "deformation.pth")
        if os.path.exists(deform_path):
            print("loading deformation model from {}".format(deform_path))
            weight_dict = torch.load(deform_path, map_location="cuda")
            self._deformation.load_state_dict(weight_dict)
        else:
            print("No deformation weights found at {}; using initialization.".format(deform_path))

        self._deformation = self._deformation.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def save_hash(self, path):
        torch.save(self._deformation.state_dict(), os.path.join(path, "deformation.pth"))

    def save_ply(self, path, mask=None):

        mkdir_p(os.path.dirname(path))

        mask_np = None
        if mask is not None:
            mask_tensor = torch.as_tensor(mask, device=self._xyz.device)
            if mask_tensor.ndim > 1:
                mask_tensor = mask_tensor.squeeze()
            if mask_tensor.shape[0] != self._xyz.shape[0]:
                raise ValueError("Mask length must match number of Gaussians.")
            mask_np = mask_tensor.bool().detach().cpu().numpy()

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dynamic = self._dynamic.detach().cpu().numpy()

        if mask_np is not None:
            xyz = xyz[mask_np]
            normals = normals[mask_np]
            f_dc = f_dc[mask_np]
            f_rest = f_rest[mask_np]
            opacities = opacities[mask_np]
            scale = scale[mask_np]
            rotation = rotation[mask_np]
            dynamic = dynamic[mask_np]
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, dynamic), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def _assign_from_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        def _require(name: str) -> np.ndarray:
            if name not in arrays:
                raise KeyError(f"Field '{name}' missing from point cloud arrays.")
            return np.asarray(arrays[name], dtype=np.float32)

        xyz = np.stack((_require("x"), _require("y"), _require("z")), axis=1)
        opacities = _require("opacity")[..., np.newaxis]
        dynamic = _require("dynamic")[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        for idx in range(3):
            features_dc[:, idx, 0] = _require(f"f_dc_{idx}")

        extra_f_names = sorted([name for name in arrays.keys() if name.startswith("f_rest_")], key=lambda x: int(x.split('_')[-1]))
        expected = 3 * (self.max_sh_degree + 1) ** 2 - 3
        if len(extra_f_names) != expected:
            raise ValueError(f"Expected {expected} f_rest_* fields, found {len(extra_f_names)}.")
        features_extra = np.stack([_require(name) for name in extra_f_names], axis=1)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = sorted([name for name in arrays.keys() if name.startswith("scale_")], key=lambda x: int(x.split('_')[-1]))
        scales = np.stack([_require(name) for name in scale_names], axis=1)

        rot_names = sorted([name for name in arrays.keys() if name.startswith("rot")], key=lambda x: int(x.split('_')[-1]))
        rots = np.stack([_require(name) for name in rot_names], axis=1)

        self._dynamic = nn.Parameter(torch.tensor(dynamic, dtype=torch.float, device="cuda").requires_grad_(True))
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def load_ply(self, path) : 
        plydata = PlyData.read(path)
        arrays = {name: np.asarray(plydata.elements[0][name]) for name in plydata.elements[0].data.dtype.names}
        self._assign_from_arrays(arrays)

    def load_from_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        """Load Gaussian parameters directly from an attribute dictionary."""
        self._assign_from_arrays(arrays)


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_dy_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.dy_optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_points(self, mask):
        valid_points_mask = ~mask

        dy_optimizer_tensors = self._prune_dy_optimizer(valid_points_mask)
        self._dynamic = dy_optimizer_tensors["dynamic"]

        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._max_weight = self._max_weight[valid_points_mask]


        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def cat_tensors_to_dy_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.dy_optimizer.param_groups:
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def densification_postfix(self, new_dynamic, new_xyz, new_features_dc, new_features_rest, new_opacities, 
                              new_scaling, new_rotation, new_max_weight):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        # "deformation": new_deformation
       }
        dy = {        
            "dynamic":new_dynamic
        }

        dy_optimizable_tensors = self.cat_tensors_to_dy_optimizer(dy)
        self._dynamic = dy_optimizable_tensors["dynamic"]

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._max_weight = torch.cat((self._max_weight, new_max_weight), dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

        new_dynamic = self._dynamic[selected_pts_mask].repeat(N,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_max_weight = self._max_weight[selected_pts_mask].repeat(N)

        self.densification_postfix( new_dynamic ,new_xyz, new_features_dc, 
                                   new_features_rest, new_opacity, new_scaling, new_rotation,new_max_weight)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_dynamic = self._dynamic[selected_pts_mask]  
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_weight = self._max_weight[selected_pts_mask]

        self.densification_postfix(new_dynamic ,new_xyz, new_features_dc, new_features_rest, 
                                   new_opacities, new_scaling, new_rotation,new_max_weight)

    @property
    def get_aabb(self):
        return self._deformation.get_aabb
    
 
    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    def densify(self, max_grad, min_opacity, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    
    def weight_prune(self, weight_threshold):
        prune_mask = (self._max_weight < weight_threshold)
        self.prune_points(prune_mask)
        self._max_weight = self._max_weight * 0.0 # torch.zeros((self._xyz.shape[0]), device="cuda")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1



    # def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
    #     return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()
