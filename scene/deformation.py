import copy
import functools
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.cdf import ComputationalDeformationField

from utils.compute_regularization import (
    _compute_smooth_regularization_71,
)

def load_vertex_dhe_buffers(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if not hasattr(model, "voxel_3d"):
        print("Warning: load_vertex_dhe_buffers called on a model without voxel_3d (likely 4D grid). Skipping.")
        return

    for i, grid in enumerate(model.voxel_3d.grids):
        A_key = f"deformation_net.grid.voxel_3d.grids.{i}.vertex_dhe.A"
        b_key = f"deformation_net.grid.voxel_3d.grids.{i}.vertex_dhe.b"
        if A_key in checkpoint and b_key in checkpoint:
            grid.vertex_dhe.A.data.copy_(checkpoint[A_key])
            grid.vertex_dhe.b.data.copy_(checkpoint[b_key])
            print(f"Grid {i}: vertex_dhe A and b loaded.")
        else:
            print(f"Warning: {A_key} or {b_key} not found in checkpoint.")

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, args=None, model_path=None):
        super(Deformation, self).__init__()
        self.args = args
        self.D = D
        self.W = W
        self.no_grid = args.no_grid
        self.dhe = True
        self.smooth_cfg = args.smooth_cfg
        self.final_out_dim = None

        voxel_conf = copy.deepcopy(self.args.h2m_voxel_conf)
        time_conf = copy.deepcopy(self.args.h2m_time_conf)
        self.grid = ComputationalDeformationField(
            voxel_conf=voxel_conf,
            time_conf=time_conf,
            render_mode=self.args.h2m_render_mode
        )
        self.final_out_dim = self.grid.output_dim
        self.compute_smooth_regularization = functools.partial(_compute_smooth_regularization_71, self)        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        self.feature_out = [nn.Linear(self.final_out_dim, self.W)]
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    def query_time(self, xyz, scales_emb, rotations_emb, time_feature, time, need_normalize=True):
        if need_normalize:
            xyz = self.normalize_aabb(xyz, self.grid.get_aabb)
        xyz_t = torch.cat([xyz,time],-1)
        grid_feature = self.grid(xyz_t)
        hidden = grid_feature    
        hidden = self.feature_out(hidden) 
        return hidden
    
    @property
    def get_empty_ratio(self):
        return self.ratio

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None, compute_smooth=False):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb, compute_smooth)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    
    def forward_dynamic(self,
                        rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, 
                        time_feature, time_emb,
                        compute_smooth=True,
                        smooth_cfg=None
                        ):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        
        dx = None
        ds = None
        dr = None
        do = None
        dshs = None
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1) 
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        if self.args.no_ds :
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr
        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden)
        
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
            shs = torch.zeros_like(shs_emb)
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        out_pts, out_scales, out_rot, out_opacity, out_shs = (
            pts, scales, rotations, opacity, shs
        )
        reg_smooth = None
        if (self.smooth_cfg is not None) and compute_smooth:
            orig_cache = {
                "dx": dx,
                "ds": ds,
                "dr": dr,
                "do": do,
                "dshs": dshs,
            }
            reg_smooth = self.compute_smooth_regularization(
                rays_pts_emb[:, :3], scales_emb, rotations_emb, opacity_emb, shs_emb,
                time_emb[:, :1], out_pts, out_scales, out_rot, out_opacity, out_shs,
                self.smooth_cfg,
                orig_cache
            )
        return out_pts, out_scales, out_rot, out_opacity, out_shs, reg_smooth
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    
    def normalize_aabb(self, pts, aabb):
        norm = (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
        return (norm + 1) / 2  # [0,1]
    
    
class deform_network(nn.Module):
    def __init__(self, args, model_path) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        defor_depth= args.defor_depth
        self.deformation_net = Deformation(W=net_width, D=defor_depth, args=args, model_path=model_path)
        self.apply(initialize_weights)


    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, compute_smooth=False):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel, compute_smooth)
    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, compute_smooth=False):
        point_emb = point
        scales_emb = scales
        rotations_emb = rotations
        
        
        means3D, scales, rotations, opacity, shs, reg_smooth = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel, compute_smooth)
        return means3D, scales, rotations, opacity, shs, reg_smooth
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)

def poc_fre(input_data,poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb
